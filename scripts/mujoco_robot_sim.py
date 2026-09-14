#!/usr/bin/env python3
"""
mujoco_robot_sim.py
--------------------
Impersonate the REAL UR5e + ur_ros_driver on the SAME ROS topics/services the
RMP2 motion planner already talks to, so the planner cannot tell it is driving
a MuJoCo simulation instead of hardware.

    python3 scripts/mujoco_robot_sim.py --check    no ROS, just verify + print
    python3 scripts/mujoco_robot_sim.py            publish (needs roscore)

ROS contract copied from real_robot_sim.py / ur5e_real.py:

    /ur_hardware_interface/start_jog   ur_ros_driver/StartJog   (service, in)
    /jog_control                       ur_ros_driver/JogControl (topic,  in)
    /joint_states                      sensor_msgs/JointState   (topic, out)
    /ur_hardware_interface/tcp_pose    geometry_msgs/TransformStamped (out)

FOUR THINGS THIS FILE IS TRYING TO TEACH
=========================================

1. THE JOINT-ORDERING LANDMINE -- the whole reason this header is long.
   Two different consumers read /joint_states two DIFFERENT ways:
     - robot_state_publisher (feeds IDMP's robot self-filter via TF) matches
       joints BY NAME.
     - RMP2's ur5e_real.py ignores `name` and indexes positionally, doing an
       unconditional swap of indices 0 and 2 on position/velocity/effort
       (see callback_joint_state: `temp_pos[[0,2]] = temp_pos[[2,0]]`).
   That swap only makes sense if the message is in ALPHABETICAL joint-name
   order (elbow, shoulder_lift, shoulder_pan, wrist_1, wrist_2, wrist_3),
   which is what ROS's joint_state_controller publishes by default -- indices
   0 and 2 of that ordering are exactly shoulder_pan and elbow, and swapping
   them lands on the standard kinematic order shoulder_pan, shoulder_lift,
   elbow, wrist_1, wrist_2, wrist_3.
   So we publish /joint_states in ALPHABETICAL order with `name` filled in to
   match. This satisfies both consumers: robot_state_publisher looks up by
   name, and RMP2's positional swap recovers the kinematic order. MuJoCo's own
   NATIVE joint order in ur5e.xml is the kinematic order already (shoulder_pan
   first) -- publishing that native order directly would silently scramble
   joints 0 and 2 for RMP2 with no error at all.

2. CTRL IS A POSITION TARGET, NOT A TORQUE OR A VELOCITY.
   The vendored ur5e.xml drives its six joints with position-servo <general>
   actuators (biastype="affine", strong gains). `data.ctrl[i]` is the target
   joint ANGLE. JogControl gives us a commanded joint VELOCITY, so to execute
   it we integrate it into the position target every physics step:
       data.ctrl[i] += commanded_velocity[i] * model.opt.timestep
   then call mj_step. This is the entire point of this rewrite: the file
   being replaced "simulated" the arm with pure numpy position += vel * dt
   and no physics at all, so it passed straight through tables. Here every
   commanded velocity is realised as actual contact-respecting dynamics.

3. NEVER ASSUME QPOS/QVEL INDEX i BELONGS TO JOINT i.
   Joint, qpos and dof addresses are looked up BY NAME with mj_name2id /
   jnt_qposadr / jnt_dofadr. Free joints and multi-dof joints would break a
   positional assumption; here it happens to be 1:1 for six hinge joints, but
   the lookup is written generally anyway so it stays correct if the scene
   changes.  Likewise the actuator that drives a given joint is found via
   `actuator_trnid`, not by assuming actuator order matches joint order
   (they happen to match in ur5e.xml, but nothing guarantees it).

4. STATE COMES BACK FROM THE SIMULATION, NEVER FROM OUR OWN INTEGRATION.
   Positions/velocities are read from data.qpos/data.qvel after mj_step, not
   accumulated by us (that was the other bug in the old file: it integrated
   position with a hardcoded 0.1s while separately integrating velocity with
   0.2s, two different deltas for one control period). Effort is stood in by
   data.qfrc_actuator -- there is no real force/torque sensor in this model,
   this is a reasonable proxy, not a measurement. The TCP pose is read from
   the `attachment_site` on wrist_3_link (world position + orientation),
   including a REAL quaternion via mat_to_quat -- the file being replaced
   left rotation as an all-zero quaternion, which isn't even a valid unit
   quaternion.
"""

import argparse
import os
import sys

# MUST be set before mujoco is imported. Without it MuJoCo silently falls back
# to a windowed context; headless rendering/stepping can behave oddly with no
# error whatsoever. We don't render here, but we match the sibling script for
# consistency and because run_node() may share a process with rendering nodes.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mujoco_rgbd_node import mat_to_quat  # noqa: E402


# ----------------------------------------------------------------------
# Joint ordering -- see trap (1) in the module docstring.
# ----------------------------------------------------------------------

# MuJoCo's native order in ur5e.xml (also the standard kinematic order, and
# the order RMP2's own forward kinematics expect, and the actuator order).
JOINT_NAMES_NATIVE = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# What we actually publish on /joint_states: alphabetical, matching what
# ROS's joint_state_controller publishes on real hardware, which is what
# RMP2's ur5e_real.py positional [0,2] swap is built to undo.
JOINT_NAMES_PUBLISH = sorted(JOINT_NAMES_NATIVE)

assert JOINT_NAMES_PUBLISH == [
    "elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# Index mapping: PUBLISH_FROM_NATIVE[i] is the native-order index that fills
# publish-order slot i. i.e. publish_array = native_array[PUBLISH_FROM_NATIVE]
PUBLISH_FROM_NATIVE = [JOINT_NAMES_NATIVE.index(n) for n in JOINT_NAMES_PUBLISH]

UR5E_XML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "assets", "robots", "ur5e", "ur5e.xml")

PUBLISH_RATE = 125.0  # Hz -- typical UR joint-state publish rate. Not
                       # load-bearing for correctness, just realism.


# ----------------------------------------------------------------------
# Model-derived lookups. These are the only functions that touch model
# structure by name -- everything above them and below them is just
# addresses and arrays.
# ----------------------------------------------------------------------

def joint_addrs(model, names):
    """name -> (joint id, qpos address, dof address), looked up BY NAME."""
    out = {}
    for n in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid < 0:
            raise RuntimeError("joint %r not found in model" % n)
        out[n] = (jid, int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]))
    return out


def actuator_for_joint_ids(model, joint_ids):
    """joint id -> driving actuator id, via the transmission target -- NOT by
    assuming actuator order matches joint order (see trap 3)."""
    jid_to_act = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
    return [jid_to_act[j] for j in joint_ids]


def load_check_model():
    """Bare vendored robot -- compiles standalone, sufficient to exercise
    joint lookup, actuator integration, qpos/qvel readback and TCP pose."""
    return mujoco.MjModel.from_xml_path(UR5E_XML_PATH)


def load_scene_model():
    """Full workcell (table, wall, clutter, ChArUco board, UR5e) built by the
    sibling scene script. This is the only line that differs between
    --check and the real node -- everything else is shared."""
    from scene import compile_scene_model
    return compile_scene_model(include_robot=True)


def step_jog(model, data, ctrl_ids, joint_vel_native):
    """Integrate one commanded native-order joint-velocity vector into the
    position-servo targets for one physics step, then advance the sim.
    See trap (2): ctrl is a target ANGLE, so velocity gets integrated in."""
    dt = model.opt.timestep
    for k, a in enumerate(ctrl_ids):
        data.ctrl[a] += float(joint_vel_native[k]) * dt
    mujoco.mj_step(model, data)


def read_joint_state_publish_order(data, addrs_native):
    """Return (position, velocity, effort) arrays in JOINT_NAMES_PUBLISH
    order, read straight from the simulation (trap 4)."""
    pos, vel, eff = [], [], []
    for name in JOINT_NAMES_PUBLISH:
        _, qpos_adr, dof_adr = addrs_native[name]
        pos.append(float(data.qpos[qpos_adr]))
        vel.append(float(data.qvel[dof_adr]))
        eff.append(float(data.qfrc_actuator[dof_adr]))
    return pos, vel, eff


def read_tcp_pose(model, data):
    """Position + REAL unit quaternion of attachment_site, world frame."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id < 0:
        raise RuntimeError("no site named 'attachment_site' in model")
    pos = data.site_xpos[site_id].copy()
    quat_xyzw = mat_to_quat(data.site_xmat[site_id].reshape(3, 3))
    return pos, quat_xyzw


# ----------------------------------------------------------------------
# --check : everything verifiable without a roscore
# ----------------------------------------------------------------------

def run_check():
    model = load_check_model()
    data = mujoco.MjData(model)

    print("Model compiled: %d bodies, %d geoms, %d DOF"
          % (model.nbody, model.ngeom, model.nv))

    addrs = joint_addrs(model, JOINT_NAMES_NATIVE)
    print("\nJoint addresses (looked up by name, not assumed positional):")
    for n in JOINT_NAMES_NATIVE:
        jid, qpos_adr, dof_adr = addrs[n]
        print("  %-22s id=%d  qpos_adr=%d  dof_adr=%d" % (n, jid, qpos_adr, dof_adr))

    print("\nNative (MuJoCo/kinematic) order: %s" % JOINT_NAMES_NATIVE)
    print("Publish (alphabetical)   order: %s" % JOINT_NAMES_PUBLISH)
    print("Index mapping publish[i] <- native[PUBLISH_FROM_NATIVE[i]]:")
    for i, n in enumerate(JOINT_NAMES_PUBLISH):
        print("  publish[%d]=%-22s <- native[%d]=%s"
              % (i, n, PUBLISH_FROM_NATIVE[i], JOINT_NAMES_NATIVE[PUBLISH_FROM_NATIVE[i]]))

    # ---- ordering round-trip: the test that actually proves the landmine
    # in the module docstring is handled, not just described. ----
    native_vals = {
        "shoulder_pan_joint": 0.1, "shoulder_lift_joint": 0.2,
        "elbow_joint": 0.3, "wrist_1_joint": 0.4,
        "wrist_2_joint": 0.5, "wrist_3_joint": 0.6,
    }
    for n in JOINT_NAMES_NATIVE:
        data.qpos[addrs[n][1]] = native_vals[n]
    mujoco.mj_forward(model, data)

    # Build the payload exactly as read_joint_state_publish_order does.
    publish_pos = [data.qpos[addrs[n][1]] for n in JOINT_NAMES_PUBLISH]
    print("\nDistinct native-order angles set: %s"
          % [native_vals[n] for n in JOINT_NAMES_NATIVE])
    print("Payload as it would be published (alphabetical order): %s" % publish_pos)

    # RMP2's ur5e_real.py callback_joint_state does exactly this swap.
    rmp2_view = np.asarray(publish_pos, dtype=np.float32)
    rmp2_view[[0, 2]] = rmp2_view[[2, 0]]
    expected_native = np.array([native_vals[n] for n in JOINT_NAMES_NATIVE], dtype=np.float32)
    print("After RMP2's [0,2] positional swap:                    %s" % rmp2_view.tolist())
    print("Original native-order values:                          %s" % expected_native.tolist())
    assert np.allclose(rmp2_view, expected_native), \
        "ordering round-trip FAILED -- RMP2 would receive scrambled joints"
    print("Ordering round-trip OK: RMP2's positional swap recovers native order.")

    # ---- home keyframe: attachment_site pose ----
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key_id >= 0, "no 'home' keyframe in model"
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    pos, quat_xyzw = read_tcp_pose(model, data)
    print("\nHome keyframe attachment_site pose:")
    print("  position          [%.4f %.4f %.4f]" % tuple(pos))
    print("  quaternion (xyzw) [%.4f %.4f %.4f %.4f]" % tuple(quat_xyzw))
    qnorm = float(np.linalg.norm(quat_xyzw))
    print("  quaternion norm   %.6f (should be ~1.0)" % qnorm)
    assert pos[2] > 0, "attachment_site below z=0 -- arm mounted through the table"
    assert abs(qnorm - 1.0) < 1e-3, "quaternion is not unit-norm"

    # ---- jog integration: command a known velocity, run ~1s, check motion ----
    ctrl_ids = actuator_for_joint_ids(model, [addrs[n][0] for n in JOINT_NAMES_NATIVE])
    joint_vel_native = np.array([0.10, -0.10, 0.05, 0.00, 0.00, 0.20])
    start_qpos = np.array([data.qpos[addrs[n][1]] for n in JOINT_NAMES_NATIVE])

    sim_seconds = 1.0
    n_steps = int(round(sim_seconds / model.opt.timestep))
    for _ in range(n_steps):
        step_jog(model, data, ctrl_ids, joint_vel_native)

    end_qpos = np.array([data.qpos[addrs[n][1]] for n in JOINT_NAMES_NATIVE])
    actual_delta = end_qpos - start_qpos
    expected_delta = joint_vel_native * sim_seconds

    print("\nJog integration over %.2fs (%d physics steps, dt=%.5f):"
          % (sim_seconds, n_steps, model.opt.timestep))
    print("  commanded velocity (rad/s)  %s" % joint_vel_native.tolist())
    print("  expected delta (rad)        %s" % expected_delta.tolist())
    print("  actual delta (rad)          %s" % actual_delta.tolist())

    for k in range(6):
        if abs(expected_delta[k]) < 1e-9:
            # Small drift from dynamic coupling with the OTHER moving joints
            # (gravity/inertial cross-terms) is expected and fine; it should
            # stay far below a real commanded motion of ~0.05-0.2 rad.
            assert abs(actual_delta[k]) < 1e-2, \
                "joint %d moved with zero commanded velocity" % k
        else:
            assert np.sign(actual_delta[k]) == np.sign(expected_delta[k]), \
                "joint %d moved the WRONG direction" % k
            ratio = actual_delta[k] / expected_delta[k]
            assert 0.5 < ratio < 1.5, \
                "joint %d moved %.1f%% of commanded amount" % (k, ratio * 100)
    print("  all joints moved in the commanded direction, roughly the "
          "commanded amount.")

    print("\nOK.")


# ----------------------------------------------------------------------
# ROS node
# ----------------------------------------------------------------------

class JogState:
    def __init__(self):
        self.armed = False
        self.target_vel_native = np.zeros(6)


def run_node():
    import rospy
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import TransformStamped
    from ur_ros_driver.srv import StartJog, StartJogResponse
    from ur_ros_driver.msg import JogControl

    rospy.init_node("mujoco_robot_sim", anonymous=False)

    model = load_scene_model()
    data = mujoco.MjData(model)

    addrs = joint_addrs(model, JOINT_NAMES_NATIVE)
    ctrl_ids = actuator_for_joint_ids(model, [addrs[n][0] for n in JOINT_NAMES_NATIVE])

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    jog = JogState()

    def handle_start_jog(req):
        # No real safety state machine in simulation -- just record the flag
        # and always report success, matching what the planner needs to see
        # to proceed (real_robot_sim.py does the same).
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

    steps_per_frame = max(1, int(round((1.0 / PUBLISH_RATE) / model.opt.timestep)))
    rospy.loginfo("mujoco_robot_sim: %d physics steps/frame at %.0f Hz, dt=%.5f",
                  steps_per_frame, PUBLISH_RATE, model.opt.timestep)

    rate = rospy.Rate(PUBLISH_RATE)
    while not rospy.is_shutdown():
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

        tcp_pos, tcp_quat = read_tcp_pose(model, data)
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "base_link"
        t.child_frame_id = "tool0"
        t.transform.translation.x = float(tcp_pos[0])
        t.transform.translation.y = float(tcp_pos[1])
        t.transform.translation.z = float(tcp_pos[2])
        t.transform.rotation.x = tcp_quat[0]
        t.transform.rotation.y = tcp_quat[1]
        t.transform.rotation.z = tcp_quat[2]
        t.transform.rotation.w = tcp_quat[3]
        pub_tcp.publish(t)

        rate.sleep()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                     help="verify and print without touching ROS")
    args = ap.parse_args()
    if args.check:
        run_check()
    else:
        run_node()


if __name__ == "__main__":
    main()
