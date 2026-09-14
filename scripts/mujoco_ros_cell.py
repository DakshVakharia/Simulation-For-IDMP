#!/usr/bin/env python3
"""
mujoco_ros_cell.py
-------------------
Combine mujoco_robot_sim.py and mujoco_rgbd_node.py into ONE process stepping
ONE MuJoCo world.

Why this file exists: run the robot and the camera as two separate ROS nodes,
each stepping its own independent MjModel, and the camera never sees the arm
or the clutter -- its point cloud disagrees with whatever /joint_states says
the arm is doing at that instant. This file steps a single shared workcell
(table, wall, clutter, UR5e -- built by scene.py) and serves both the robot
topics/services and the camera topics/TF out of the same physics state, so
everything downstream is looking at one consistent world.

    python3 scripts/mujoco_ros_cell.py --check     no ROS, just verify + print
    python3 scripts/mujoco_ros_cell.py             publish (needs roscore)
    python3 scripts/mujoco_ros_cell.py --viewer    publish AND open a window on
                                                    the same stepped world

Everything reused from the two original files is imported, not copied --
they remain the component implementations and their own --check runs are the
unit tests for joint ordering, ctrl-as-position-target, the optical frame
flip, etc. See their module docstrings for the traps; this file only adds:

1. HOME POSE SEEDING. scene's _compile_model strips ur5e.xml's own
   <keyframe> (it doesn't survive the extra clutter DOF -- see that module's
   docstring) and build_scene_xml() re-emits a scene-level "home" keyframe
   instead. mujoco_robot_sim.run_node resets to that keyframe by name. This
   file does the same lookup-by-name but writes qpos/ctrl directly (see
   HOME_QPOS_NATIVE below) so the seeding logic is visible and doesn't depend
   on the keyframe's exact free-body ordering matching -- setting the six
   robot joints explicitly is all that is actually needed to avoid the
   qpos=0/ctrl=0 servo yank.

2. TWO RATES, ONE LOOP. The robot loop runs at 125 Hz (matching
   mujoco_robot_sim's PUBLISH_RATE); the camera renders/publishes only every
   k-th iteration, k = round(125 / camera_rate). Two 640x576 renders per
   frame are expensive enough that doing them every physics tick would stall
   the joint-state rate the planner depends on.

3. TCP POSE VIA TF, NOT THE MUJOCO SITE. mujoco_robot_sim.read_tcp_pose
   returns the bare flange (attachment_site on wrist_3_link) -- there is no
   Robotiq Hand-E gripper in the MuJoCo model, so it cannot report tcp_link,
   ~65mm further out, which is what the real driver publishes and what
   RMP2 uses for goal convergence. Since robot_state_publisher (fed by the
   URDF and the /joint_states WE publish) already knows the full kinematic
   chain out to tcp_link, this node just asks TF for it. That means the
   gripper offset lives in exactly one place (the URDF) and never gets
   duplicated or drifts out of sync here.
"""

import argparse
import os
import sys

# MUST be set before mujoco is imported -- see the two originals for why.
# EGL is HEADLESS-ONLY: under it mujoco.viewer cannot open a window at all, so
# --viewer has to select the windowed GLFW backend instead. This is decided
# from raw argv because it must happen at import time, long before argparse
# runs. The two sibling modules imported below setdefault MUJOCO_GL to egl
# themselves, so setting it here first is what makes --viewer stick.
if "--viewer" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mujoco_robot_sim import (  # noqa: E402
    JOINT_NAMES_NATIVE, JOINT_NAMES_PUBLISH, joint_addrs,
    actuator_for_joint_ids, step_jog, read_joint_state_publish_order,
    read_tcp_pose,
)
from mujoco_rgbd_node import (  # noqa: E402
    ORBBEC, render_rgbd, mujoco_cam_to_ros_optical, mat_to_quat,
    make_camera_info, COLOR_OPTICAL_FRAME, DEPTH_OPTICAL_FRAME,
)
from scene import compile_scene_model, apply_environment_only  # noqa: E402


ROBOT_RATE = 125.0        # Hz, matches mujoco_robot_sim.PUBLISH_RATE
DEFAULT_CAMERA_RATE = 15.0  # Hz, ROS param ~camera_rate overrides this

# Viewer redraw rate. Syncing the window every one of the 125 robot loops
# measurably starves the publishers (125 -> 43 Hz joint_states, 15.7 -> 5.5 Hz
# depth, measured); at 30 Hz the window is still smooth to the eye and the
# published rates stay at their targets. Same every-k-th-loop trick the camera
# already uses.
VIEWER_RATE = 30.0

# The real cell's startup pose (matches the HOME_QPOS_ROBOT convention used
# elsewhere in this project), in NATIVE/kinematic joint order: shoulder_pan,
# shoulder_lift, elbow, wrist_1, wrist_2, wrist_3. There is no usable
# <keyframe> to reset to here (scene strips/replaces it, see module
# docstring), so this is seeded explicitly by name.
HOME_QPOS_NATIVE = [0.0, -1.57, 1.57, -1.57, -1.57, -1.57]

# HYPOTHESIS FIX, needs live verification in RViz -- see README.md §7.5 for
# the diagnosis. scene.py's table/camera layout was
# authored as an arbitrary MuJoCo world frame and never cross-checked
# against which way the real UR5e's true base_link +X actually points (the
# two OTHER 180-degree corrections in this stack -- ur5e.xml's own <body
# name="base" quat="0 0 0 -1">, and ur_description's base_link ->
# base_link_inertia fixed joint -- already cancel each other out correctly,
# so a residual whole-arm ~180-degree yaw between the rendered URDF arm and
# the published point cloud in RViz points at THIS layer instead). Only the
# camera TF needs it: robot_state_publisher renders the arm purely from
# /joint_states + the URDF, never touching MuJoCo's raw world coordinates,
# so this cannot affect the arm's own rendered shape/pose, only where the
# camera (and therefore the point cloud) is placed relative to base_link.
# Flip to False if enabling this makes the RViz mismatch worse instead of
# better -- that would mean the true error is a different axis/angle, not
# this one.
WORLD_YAW_180_Z = True


def _world_to_base_link(pos, rot_mat):
    """Re-express a MuJoCo world-frame position + 3x3 rotation matrix as
    though the world were yawed 180 degrees about Z before being called
    base_link. See WORLD_YAW_180_Z above."""
    if not WORLD_YAW_180_Z:
        return pos, rot_mat
    rz180 = np.diag([-1.0, -1.0, 1.0])
    return pos * np.array([-1.0, -1.0, 1.0]), rz180 @ rot_mat


# ----------------------------------------------------------------------
# Model build + home seeding
# ----------------------------------------------------------------------

def build_model():
    """Compile the single shared workcell, sized for the Orbbec renderer."""
    return compile_scene_model(include_robot=True, camera_fovy=ORBBEC.fovy_deg,
                                offwidth=ORBBEC.width, offheight=ORBBEC.height)


def seed_home(model, data, addrs, ctrl_ids):
    """Write HOME_QPOS_NATIVE into both qpos and the driving actuators' ctrl,
    looked up BY NAME, then mj_forward. Without also setting ctrl, the
    position servos see qpos at home but a target of 0 and immediately yank
    the arm back toward zero (same trap scene's own keyframe has to
    account for)."""
    for name, q in zip(JOINT_NAMES_NATIVE, HOME_QPOS_NATIVE):
        _, qpos_adr, _ = addrs[name]
        data.qpos[qpos_adr] = q
    for a, q in zip(ctrl_ids, HOME_QPOS_NATIVE):
        data.ctrl[a] = q
    mujoco.mj_forward(model, data)


def render_rgbd_masked(renderer, model, data, camera, scene_option=None):
    """Local variant of mujoco_rgbd_node.render_rgbd that threads a caller-
    owned MjvOption through update_scene, needed for ~env_only masking.
    render_rgbd itself calls update_scene(data, camera=...) with no
    scene_option, so this cannot just reuse it when a mask is active."""
    if scene_option is None:
        return render_rgbd(renderer, model, data, camera=camera)

    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    far = float(model.vis.map.zfar * model.stat.extent)
    invalid = ~((depth > 0) & (depth < far * 0.99))
    return rgb, depth, invalid


# ----------------------------------------------------------------------
# --check : everything verifiable without a roscore
# ----------------------------------------------------------------------

def run_check():
    model = build_model()
    data = mujoco.MjData(model)
    print("Model compiled: %d bodies, %d geoms, %d DOF, camera %dx%d"
          % (model.nbody, model.ngeom, model.nv, ORBBEC.width, ORBBEC.height))

    addrs = joint_addrs(model, JOINT_NAMES_NATIVE)
    ctrl_ids = actuator_for_joint_ids(model, [addrs[n][0] for n in JOINT_NAMES_NATIVE])
    seed_home(model, data, addrs, ctrl_ids)

    pos, quat_xyzw = read_tcp_pose(model, data)
    print("\nHome pose (native order): %s" % HOME_QPOS_NATIVE)
    print("Flange (attachment_site) pose at home:")
    print("  position          [%.4f %.4f %.4f]" % tuple(pos))
    print("  quaternion (xyzw) [%.4f %.4f %.4f %.4f]" % tuple(quat_xyzw))
    assert pos[2] > 0, "attachment_site below the table at home"

    # ctrl == qpos at home -> zero-velocity jog should not move anything.
    zero_vel = np.zeros(6)
    start = np.array([data.qpos[addrs[n][1]] for n in JOINT_NAMES_NATIVE])
    for _ in range(50):
        step_jog(model, data, ctrl_ids, zero_vel)
    end = np.array([data.qpos[addrs[n][1]] for n in JOINT_NAMES_NATIVE])
    drift = np.abs(end - start).max()
    print("\nHold-at-home drift over 50 steps with zero commanded velocity: "
          "%.6f rad (should be tiny)" % drift)
    assert drift < 1e-2, "arm drifted away from home with zero jog -- ctrl/qpos seeding is wrong"

    # A real jog command should still move it, proving actuators are live.
    joint_vel_native = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(50):
        step_jog(model, data, ctrl_ids, joint_vel_native)
    moved = np.array([data.qpos[addrs[n][1]] for n in JOINT_NAMES_NATIVE])
    assert moved[0] > end[0], "shoulder_pan did not move under a commanded velocity"
    print("Jog still moves the arm from home (shoulder_pan advanced by %.4f rad)."
          % (moved[0] - end[0]))

    # Render one frame to prove the camera sees the same world the arm is in
    # (i.e. this is genuinely one shared model, not two).
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_cam")
    assert cam_id >= 0, "no depth_cam in the combined scene"
    renderer = mujoco.Renderer(model, height=ORBBEC.height, width=ORBBEC.width)
    try:
        rgb, depth, invalid = render_rgbd_masked(renderer, model, data, "depth_cam")
        valid = ~invalid
        print("\nRendered one frame from the SAME model the arm was just jogged in:")
        print("  rgb   %s %s" % (rgb.shape, rgb.dtype))
        print("  depth %s %s  valid %d/%d px" % (depth.shape, depth.dtype, valid.sum(), valid.size))
        assert valid.sum() > 0, "no valid depth pixels rendered"

        env_opt = mujoco.MjvOption()
        apply_environment_only(env_opt, True)
        rgb_env, _, _ = render_rgbd_masked(renderer, model, data, "depth_cam", env_opt)
        assert not np.array_equal(rgb, rgb_env), \
            "env_only mask had no visible effect on the render"
        print("  env_only masking changes the render (OK)")
    finally:
        renderer.close()

    print("\nOK.")


# ----------------------------------------------------------------------
# ROS node
# ----------------------------------------------------------------------

class JogState:
    def __init__(self):
        self.armed = False
        self.target_vel_native = np.zeros(6)


def run_node(show_viewer=False):
    import rospy
    import tf2_ros
    from sensor_msgs.msg import JointState, Image, CameraInfo
    from geometry_msgs.msg import TransformStamped
    from ur_ros_driver.srv import StartJog, StartJogResponse
    from ur_ros_driver.msg import JogControl

    rospy.init_node("mujoco_ros_cell", anonymous=False)

    camera_rate = float(rospy.get_param("~camera_rate", DEFAULT_CAMERA_RATE))
    env_only = bool(rospy.get_param("~env_only", False))

    model = build_model()
    data = mujoco.MjData(model)

    addrs = joint_addrs(model, JOINT_NAMES_NATIVE)
    ctrl_ids = actuator_for_joint_ids(model, [addrs[n][0] for n in JOINT_NAMES_NATIVE])
    seed_home(model, data, addrs, ctrl_ids)

    jog = JogState()

    def handle_start_jog(req):
        jog.armed = bool(req.IO)
        if not jog.armed:
            jog.target_vel_native[:] = 0.0
        return StartJogResponse(success=True)

    def handle_jog_control(msg):
        if not jog.armed:
            rospy.logwarn("Jog is not activated !")
            return
        if msg.feature != 1:
            return  # only joint-speed jogging (feature 1) is implemented
        jog.target_vel_native = np.asarray(msg.vector, dtype=np.float64)

    rospy.Service("/ur_hardware_interface/start_jog", StartJog, handle_start_jog)
    rospy.Subscriber("/jog_control", JogControl, handle_jog_control)
    pub_joints = rospy.Publisher("/joint_states", JointState, queue_size=10)
    pub_tcp = rospy.Publisher("/ur_hardware_interface/tcp_pose", TransformStamped, queue_size=10)

    pub_rgb = rospy.Publisher("/camera/color/image_raw", Image, queue_size=1)
    pub_rgb_i = rospy.Publisher("/camera/color/camera_info", CameraInfo, queue_size=1, latch=True)
    pub_dep = rospy.Publisher("/camera/depth/image_raw", Image, queue_size=1)
    pub_dep_i = rospy.Publisher("/camera/depth/camera_info", CameraInfo, queue_size=1, latch=True)
    tf_bcast = tf2_ros.TransformBroadcaster()

    tf_buf = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buf)
    tcp_link_warned = {"done": False}

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_cam")
    renderer = mujoco.Renderer(model, height=ORBBEC.height, width=ORBBEC.width)
    env_opt = None
    if env_only:
        env_opt = mujoco.MjvOption()
        apply_environment_only(env_opt, True)

    steps_per_frame = max(1, int(round((1.0 / ROBOT_RATE) / model.opt.timestep)))
    camera_every = max(1, int(round(ROBOT_RATE / camera_rate)))
    viewer_every = max(1, int(round(ROBOT_RATE / VIEWER_RATE)))
    rospy.loginfo(
        "mujoco_ros_cell: robot %.0f Hz (%d physics steps/frame), camera every "
        "%d loops (~%.1f Hz), env_only=%s",
        ROBOT_RATE, steps_per_frame, camera_every,
        ROBOT_RATE / camera_every, env_only)

    viewer = None
    if show_viewer:
        # Same passive-viewer pattern the lesson scripts use: it hands the loop
        # back to us rather than taking the process over, so the ROS rate and
        # the stepping below stay in charge and the window only ever shows the
        # state we already stepped -- one world, one loop, drawn and published.
        viewer = mujoco.viewer.launch_passive(model, data)
        if env_opt is not None:
            # Match the published camera images: env_only is fixed for the
            # node's lifetime (read once from ~env_only above), so this is a
            # one-time mask, not a live toggle like scene.py's 'E' key.
            apply_environment_only(viewer.opt, True)
        rospy.loginfo("mujoco_ros_cell: viewer open -- close the window to stop the node")

    rate = rospy.Rate(ROBOT_RATE)
    loop_i = 0
    try:
        while not rospy.is_shutdown():
            if viewer is not None and not viewer.is_running():
                rospy.loginfo("mujoco_ros_cell: viewer closed, shutting down")
                break

            vel = jog.target_vel_native
            for _ in range(steps_per_frame):
                step_jog(model, data, ctrl_ids, vel)

            stamp = rospy.Time.now()

            pos, vel_out, eff = read_joint_state_publish_order(data, addrs)
            js = JointState()
            js.header.stamp = stamp
            js.name = list(JOINT_NAMES_PUBLISH)
            js.position = pos
            js.velocity = vel_out
            js.effort = eff
            pub_joints.publish(js)

            try:
                tf = tf_buf.lookup_transform("base_link", "tcp_link", rospy.Time())
                tcp_pos = (tf.transform.translation.x, tf.transform.translation.y,
                           tf.transform.translation.z)
                tcp_quat = (tf.transform.rotation.x, tf.transform.rotation.y,
                            tf.transform.rotation.z, tf.transform.rotation.w)
            except tf2_ros.TransformException as exc:
                if not tcp_link_warned["done"]:
                    rospy.logwarn_throttle(
                        5.0, "tcp_link TF unavailable (%s), falling back to the "
                        "MuJoCo flange (attachment_site) -- expect ~65mm error "
                        "vs the real gripper tip until robot_state_publisher is up", exc)
                    tcp_link_warned["done"] = True
                tcp_pos, tcp_quat = read_tcp_pose(model, data)

            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = "base_link"
            t.child_frame_id = "tcp_link"
            t.transform.translation.x = float(tcp_pos[0])
            t.transform.translation.y = float(tcp_pos[1])
            t.transform.translation.z = float(tcp_pos[2])
            t.transform.rotation.x = float(tcp_quat[0])
            t.transform.rotation.y = float(tcp_quat[1])
            t.transform.rotation.z = float(tcp_quat[2])
            t.transform.rotation.w = float(tcp_quat[3])
            pub_tcp.publish(t)

            if loop_i % camera_every == 0:
                rgb, depth, invalid = render_rgbd_masked(
                    renderer, model, data, "depth_cam", env_opt)

                cam_stamp = rospy.Time.now()

                depth_mm = np.where(invalid, 0, depth * 1000.0)
                depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)

                msg_rgb = Image()
                msg_rgb.header.stamp = cam_stamp
                msg_rgb.header.frame_id = COLOR_OPTICAL_FRAME
                msg_rgb.height, msg_rgb.width = ORBBEC.height, ORBBEC.width
                msg_rgb.encoding = "rgb8"
                msg_rgb.is_bigendian = 0
                msg_rgb.step = ORBBEC.width * 3
                msg_rgb.data = rgb.tobytes()
                pub_rgb.publish(msg_rgb)

                msg_dep = Image()
                msg_dep.header.stamp = cam_stamp
                msg_dep.header.frame_id = DEPTH_OPTICAL_FRAME
                msg_dep.height, msg_dep.width = ORBBEC.height, ORBBEC.width
                msg_dep.encoding = "16UC1"
                msg_dep.is_bigendian = 0
                msg_dep.step = ORBBEC.width * 2
                msg_dep.data = depth_mm.tobytes()
                pub_dep.publish(msg_dep)

                pub_rgb_i.publish(make_camera_info(CameraInfo, ORBBEC, COLOR_OPTICAL_FRAME, cam_stamp))
                pub_dep_i.publish(make_camera_info(CameraInfo, ORBBEC, DEPTH_OPTICAL_FRAME, cam_stamp))

                cam_pos, cam_mat = _world_to_base_link(
                    data.cam_xpos[cam_id], data.cam_xmat[cam_id].reshape(3, 3))
                rot = mujoco_cam_to_ros_optical(cam_mat)
                qx, qy, qz, qw = mat_to_quat(rot)
                for child in (DEPTH_OPTICAL_FRAME, COLOR_OPTICAL_FRAME):
                    tf_msg = TransformStamped()
                    tf_msg.header.stamp = cam_stamp
                    tf_msg.header.frame_id = "base_link"
                    tf_msg.child_frame_id = child
                    tf_msg.transform.translation.x = float(cam_pos[0])
                    tf_msg.transform.translation.y = float(cam_pos[1])
                    tf_msg.transform.translation.z = float(cam_pos[2])
                    tf_msg.transform.rotation.x = qx
                    tf_msg.transform.rotation.y = qy
                    tf_msg.transform.rotation.z = qz
                    tf_msg.transform.rotation.w = qw
                    tf_bcast.sendTransform(tf_msg)

            if viewer is not None and loop_i % viewer_every == 0:
                viewer.sync()

            loop_i += 1
            rate.sleep()
    finally:
        renderer.close()
        if viewer is not None:
            viewer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                     help="verify and print without touching ROS")
    ap.add_argument("--viewer", action="store_true",
                     help="also open an interactive MuJoCo window showing the "
                          "very same simulation being published (needs a display; "
                          "switches the GL backend from egl to glfw)")
    # parse_known_args, not parse_args: roslaunch appends __name:= and __log:=
    # to every node's argv, which parse_args rejects as unrecognized.
    args, _ = ap.parse_known_args()
    if args.check:
        run_check()
    else:
        run_node(show_viewer=args.viewer)


if __name__ == "__main__":
    main()
