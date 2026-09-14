#!/usr/bin/env python3
"""
mujoco_rgbd_node.py
-------------------
Publish MuJoCo's simulated RGB-D camera onto the SAME ROS topics the real
Orbbec Femto Bolt uses, so nothing downstream can tell the difference.

    python3 scripts/mujoco_rgbd_node.py --check    no ROS, just verify + print
    python3 scripts/mujoco_rgbd_node.py            publish (needs roscore)

Topics published (names copied from OrbbecSDK_ROS1/src/ros_setup.cpp:1742,1766,
which builds them as  /<camera_name>/<stream>/image_raw  and  .../camera_info):

    /camera/color/image_raw     sensor_msgs/Image        rgb8
    /camera/color/camera_info   sensor_msgs/CameraInfo
    /camera/depth/image_raw     sensor_msgs/Image        16UC1, MILLIMETRES
    /camera/depth/camera_info   sensor_msgs/CameraInfo

THREE THINGS THIS FILE IS TRYING TO TEACH
=========================================

1. INTRINSICS DRIVE THE CAMERA, NOT THE OTHER WAY ROUND.
   A real camera has fx, fy, cx, cy measured by calibration. MuJoCo has no
   such knobs -- it has ONE number, `fovy`, the vertical field of view. So we
   go backwards, deriving fovy from the real fy:

       fy = 0.5 * height / tan(fovy/2)      =>   fovy = 2*atan(0.5*height/fy)

   Change ORBBEC below and the MuJoCo camera follows. Hardcode fovy instead
   and the two silently drift apart the first time you touch a resolution.

2. WHAT MUJOCO CANNOT MATCH. MuJoCo is an ideal pinhole camera:
     - principal point is ALWAYS the image centre; a real cx/cy is off-centre
     - pixels are always square, so fx == fy; a real sensor's differ slightly
     - no lens distortion at all
   We publish the REAL cx/cy/fx/fy in CameraInfo anyway, because that is what
   the real driver publishes and consumers calibrate against it. The residual
   mismatch is the honest sim-to-real gap. Section "check" prints its size.

3. THE OPTICAL FRAME FLIP -- the trap that costs people a day.
       MuJoCo camera:  +x right, +y UP,   -z forward   (OpenGL convention)
       ROS optical:    +x right, +y DOWN, +z forward   (REP-103)
   A 180-degree rotation about x. Get it wrong and the scene renders
   perfectly, looks completely plausible, and is vertically mirrored. A
   symmetric table will NOT reveal it -- which is why the test block sits
   off-centre.
"""

import math
import os

# MUST be set before mujoco is imported. Without it MuJoCo silently falls back
# to a windowed context, and headless rendering returns an all-black image with
# no error whatsoever.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco


# ----------------------------------------------------------------------
# Orbbec Femto Bolt intrinsics.
#
# >>> THESE ARE NOMINAL, NOT MEASURED. <<<
#
# femto_bolt.launch sets depth_width/height/fps to 0, meaning "let the SDK
# choose", so the true values are NOT recorded anywhere in the repo -- they
# come off the device at runtime. Below is the Femto Bolt's NFOV-unbinned mode
# (it uses the Azure Kinect DK depth sensor): 640x576, ~75 x 65 degrees.
#
# To replace them with the real thing, plug the camera in and run:
#
#     roslaunch orbbec_camera femto_bolt.launch camera_name:=camera
#     rostopic echo -n1 /camera/depth/camera_info
#
# then copy K = [fx, 0, cx, 0, fy, cy, 0, 0, 1] into the block below.
# ----------------------------------------------------------------------

class Intrinsics:
    def __init__(self, width, height, fx, fy, cx, cy, measured=False):
        self.width, self.height = width, height
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.measured = measured

    @property #calculated every time, not stored
    def fovy_deg(self):
        """The single number MuJoCo actually accepts, derived from fy."""
        return math.degrees(2.0 * math.atan(0.5 * self.height / self.fy))

    def mujoco_equivalent(self):
        """What MuJoCo will ACTUALLY render, given it is an ideal pinhole.

        Returns (fx, fy, cx, cy) as MuJoCo realises them -- square pixels and a
        dead-centre principal point. Differences against the real values are
        the sim-to-real gap and are reported by --check.
        """
        f = 0.5 * self.height / math.tan(0.5 * math.radians(self.fovy_deg))
        return f, f, 0.5 * (self.width - 1), 0.5 * (self.height - 1)


ORBBEC = Intrinsics(
    width=640, height=576,
    fx=504.0, fy=504.0,     # nominal NFOV unbinned
    cx=320.0, cy=288.0,
    measured=False,
)

COLOR_OPTICAL_FRAME = "camera_color_optical_frame"
DEPTH_OPTICAL_FRAME = "camera_depth_optical_frame"

CAMERA_NAME = "depth_cam"       # the <camera> in scene.py
PUBLISH_RATE = 30.0             # Hz


def render_rgbd(renderer, model, data, camera=CAMERA_NAME):
    """Return (rgb uint8 HxWx3, depth float32 HxW in metres, invalid mask).

    NOTE the renderer is passed IN, not constructed here: constructing a new
    Renderer per call allocates a new GL context, fine for a one-shot script
    and hopeless at 30 fps.
    """
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    # MuJoCo returns the far-plane distance for pixels that hit nothing. Left
    # in, that becomes a solid phantom wall of obstacles sitting at max range.
    far = float(model.vis.map.zfar * model.stat.extent)
    invalid = ~((depth > 0) & (depth < far * 0.99))
    return rgb, depth, invalid


def mujoco_cam_to_ros_optical(cam_xmat):
    """Convert MuJoCo's camera rotation into a ROS optical-frame rotation.

    cam_xmat maps MuJoCo-camera coords -> world. Post-multiplying by
    diag(1,-1,-1) first re-expresses ROS-optical axes as MuJoCo ones, giving
    optical -> world, which is what TF wants.
    """
    return cam_xmat.reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])


def mat_to_quat(m):
    """Rotation matrix -> (x, y, z, w). Written out rather than pulled from
    tf.transformations so this file has no ROS import in --check mode."""
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        return ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    if i == 0:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s,
                (m[2, 1] - m[1, 2]) / s)
    if i == 1:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s,
                (m[0, 2] - m[2, 0]) / s)
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s,
            (m[1, 0] - m[0, 1]) / s)


# ----------------------------------------------------------------------
# ROS publishing
# ----------------------------------------------------------------------

def make_camera_info(CameraInfo, intr, frame_id, stamp):
    """CameraInfo carrying the REAL intrinsics.

    We publish the measured fx/fy/cx/cy rather than MuJoCo's idealised ones,
    because that is what the real driver publishes and what any consumer
    calibrates against. D is zero -- MuJoCo has no lens distortion to report.
    """
    ci = CameraInfo()
    ci.header.stamp = stamp
    ci.header.frame_id = frame_id
    ci.width, ci.height = intr.width, intr.height
    ci.distortion_model = "plumb_bob"
    ci.D = [0.0] * 5
    ci.K = [intr.fx, 0.0, intr.cx,
            0.0, intr.fy, intr.cy,
            0.0, 0.0, 1.0]
    ci.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ci.P = [intr.fx, 0.0, intr.cx, 0.0,
            0.0, intr.fy, intr.cy, 0.0,
            0.0, 0.0, 1.0, 0.0]
    return ci
