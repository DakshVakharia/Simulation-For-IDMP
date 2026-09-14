"""
scene.py
--------
Populate the workcell: table, backdrop wall, tabletop clutter for the depth
camera to see, the UR5e, and a ChArUco calibration board at a hardcoded pose.

    python scripts/scene.py            open the viewer
    python scripts/scene.py --check    compile only, print stats

VIEWER KEY:  press 'E' to toggle "environment only" rendering (table, wall,
monitor, floor, and the robot itself -- everything except the movable
tabletop clutter) on and off. The same toggle is available programmatically
via apply_environment_only(), for the depth-camera ROS node to hide clutter
when it wants a clean background with the arm still visible.

THE TRAP THIS FILE HAD TO SOLVE:  build_scene_xml() returns a plain string,
which you'd naturally compile with mujoco.MjModel.from_xml_string(...). But
relative filenames inside an XML *string* have no file location to resolve
against, and the vendored assets/robots/ur5e/ur5e.xml has its own top-level
<compiler angle="radian" meshdir="assets" .../> that its 20 .obj meshes
depend on. Giving the PARENT scene its own <compiler meshdir=...> (needed, in
theory, for the ChArUco texture) does not "merge" with the child's the way
you'd hope -- it was tried empirically here and produced
"Error opening file .../ur5e/base_0.obj: No such file or directory": the
child's relative meshdir got resolved against the wrong base entirely.

The fix actually used: never give the parent scene a <compiler> element at
all. Every path this file writes into the XML (the ChArUco texture, the
<include> target) is either absolute or, when the robot is included, made
correct by colocation -- see _compile_model() below, which writes the
generated XML into assets/robots/ur5e/ (the same directory as ur5e.xml) and
compiles with from_xml_path instead of from_xml_string. That gives ur5e.xml's
own relative meshdir="assets" a real directory to resolve against, exactly as
it has when MuJoCo Menagerie's own scene.xml is used standalone. This was
verified by actually compiling both ways before picking one -- see the
module's git history / the task report for the failed attempts.

A parent <option> element, by contrast, was tested and found to merge fine
with ur5e.xml's own <option integrator="implicitfast"/> (different attributes,
no clash), so this file does add one, for gravity.

A THIRD trap, discovered only by actually compiling with the clutter's
<freejoint/>s in place: ur5e.xml carries its own <keyframe><key name="home"
qpos="six numbers".../></keyframe>. A keyframe's qpos length is validated
against the WHOLE compiled model's nq, not the included file's own nq, so the
moment this scene adds 5 free bodies (35 more DOF) the vendored "home"
keyframe becomes invalid ("expected length 41, got 6") and the model refuses
to compile -- whether or not anything ever reads that keyframe. MuJoCo also
refuses two keys with the same name, so you cannot just add a corrected
"home" alongside it. The fix: _compile_model() strips the vendored
<keyframe> out of ur5e.xml's text before writing the colocated scratch file,
and build_scene_xml() emits its own full-length "home" keyframe instead --
the same 6 robot angles plus each clutter body's resting (pos, identity
quat) as its 7-number freejoint entry, in the exact order those bodies are
declared (verified empirically: <include>'d content occupies the lower qpos
addresses, ahead of this file's own <worldbody> children). The replacement
key also needs its OWN "ctrl" attribute matching the robot's home angles --
omitting it (as a first pass here did) leaves data.ctrl at its zero default
while qpos starts at home, so the position actuators immediately drag the
arm from home back toward qpos=0, sweeping it through the tabletop clutter
during the very stability check meant to prove the clutter is undisturbed.
"""

import re
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer


# ----------------------------------------------------------------------
# Paths. Computed at runtime, like scripts/fetch_robot.py does -- never
# hardcoded, so this works from any checkout.
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UR5E_DIR = PROJECT_ROOT / "assets" / "robots" / "ur5e"
UR5E_XML = UR5E_DIR / "ur5e.xml"
CHARUCO_PNG = PROJECT_ROOT / "assets" / "charuco" / "charuco_6x9_sq32mm_mk24mm_DICT6X6_250.png"

# Name of the scratch copy of ur5e.xml (keyframe stripped, see module
# docstring) and of the generated scene, both written into UR5E_DIR so
# ur5e.xml's own relative meshdir="assets" resolves correctly.
UR5E_NOKEY_NAME = "_scene_ur5e_nokeyframe.xml"
SCENE_SCRATCH_NAME = "_scene_scratch.xml"

HOME_QPOS_ROBOT = "-1.5708 -1.5708 1.5708 -1.5708 -1.5708 0"


# ----------------------------------------------------------------------
# Geom groups. This is the mechanism behind the environment-only toggle:
# MuJoCo's MjvOption.geomgroup is a per-group visibility mask that both the
# interactive viewer and offscreen mujoco.Renderer respect, so hiding
# "everything but the environment" is one array write, not per-geom logic.
#
#   group 0  ENVIRONMENT   table, wall, monitor, floor -- never move
#   group 1  CLUTTER       glass, bottle, blocks, ChArUco board
#   group 2  robot visual  (from the vendored ur5e.xml, untouched)
#   group 3  robot collision (from the vendored ur5e.xml, untouched)
#   group 4  robot sites     (from the vendored ur5e.xml, untouched)
# ----------------------------------------------------------------------

GROUP_ENV = 0
GROUP_CLUTTER = 1
GROUP_ROBOT_VISUAL = 2

ENV_ONLY_TOGGLE_KEY = "E"


# ----------------------------------------------------------------------
# Table. Same z=0-at-the-top convention as Lesson 3, sized up so the robot's
# reach and every clutter object below fit on it without overhang.
# ----------------------------------------------------------------------

TABLE_LENGTH = 1.60      # along x
TABLE_WIDTH = 1.10       # along y
TABLE_HEIGHT = 0.75      # floor to top surface
SLAB_THICKNESS = 0.05
LEG_THICKNESS = 0.05
LEG_INSET = 0.08

# ----------------------------------------------------------------------
# Backdrop wall. The depth camera below sits at y = -0.75 looking toward the
# table at y = 0, so "behind the table" from the camera's viewpoint is +y --
# past the table's far edge at y = +TABLE_WIDTH/2. Sized generously (rather
# than trigonometrically exact for one fixed fovy) so it fills the camera's
# background regardless of small pose changes later.
# ----------------------------------------------------------------------

WALL_GAP = 0.05          # clearance behind the table's far edge
WALL_THICKNESS = 0.05
WALL_HEIGHT = 2.0        # measured up from the FLOOR, not the table top
WALL_MARGIN_X = 1.0      # extra width past each end of the table

# ----------------------------------------------------------------------
# The depth camera, carried over from Lesson 3's bench.
# ----------------------------------------------------------------------

CAMERA_POS = (0.75, -0.75, 0.65)
CAMERA_FOVY = 58

# ----------------------------------------------------------------------
# Tabletop clutter. Every object's SMALLEST dimension must stay >= 0.04 m --
# these get voxelized at 5 mm for the distance field, and anything thinner
# than that voxel-and-a-bit disappears instead of surviving as a recognizable
# cluster of points. Each object rests with its bottom face exactly on
# z = 0: a box/cylinder of half-height h therefore has its centre at z = h.
# ----------------------------------------------------------------------

GLASS_RADIUS = 0.040
GLASS_HALF_HEIGHT = 0.060
GLASS_POS_XY = (-0.35, -0.35)
GLASS_RGBA = "0.75 0.90 1.00 0.55"

BOTTLE_BODY_RADIUS = 0.040
BOTTLE_BODY_HALF_HEIGHT = 0.080
BOTTLE_NECK_RADIUS = 0.025
BOTTLE_NECK_HALF_HEIGHT = 0.030
BOTTLE_POS_XY = (-0.55, 0.15)
BOTTLE_RGBA = "0.10 0.45 0.20 1"

MONITOR_HALF_X = 0.175
MONITOR_HALF_Y = 0.025    # thin, but 0.05 m thick overall -- clears the 0.04 m floor
MONITOR_HALF_Z = 0.125
MONITOR_POS_XY = (0.0, 0.45)
MONITOR_RGBA = "0.08 0.08 0.09 1"

# Several blocks of differing sizes -- (half_x, half_y, half_z), xy position, colour.
BLOCK_SPECS = [
    dict(name="block_small", half=(0.060, 0.060, 0.060),
         pos_xy=(0.55, -0.35), rgba="0.85 0.25 0.20 1"),
    dict(name="block_medium", half=(0.080, 0.080, 0.080),
         pos_xy=(0.60, 0.35), rgba="0.95 0.55 0.10 1"),
    dict(name="block_flat", half=(0.100, 0.060, 0.050),
         pos_xy=(0.65, 0.0), rgba="0.55 0.25 0.75 1"),
]

# ----------------------------------------------------------------------
# ChArUco calibration board. Pose is COPIED VERBATIM from the real rig's
# poses.yaml (marker "aruco_1", parent frame base_link == our world origin /
# table top). Do not "fix" this transform -- it is deliberately hardcoded.
#
# Filename decodes the physical board: 6x9 squares, 32 mm squares ->
# 0.192 m x 0.288 m. MuJoCo quat order is (w, x, y, z).
# ----------------------------------------------------------------------

BOARD_SQUARES_X = 6
BOARD_SQUARES_Y = 9
BOARD_SQUARE_SIZE = 0.032
BOARD_WIDTH = BOARD_SQUARES_X * BOARD_SQUARE_SIZE     # 0.192 m, local x
BOARD_DEPTH = BOARD_SQUARES_Y * BOARD_SQUARE_SIZE     # 0.288 m, local y
BOARD_THICKNESS = 0.003

BOARD_POS = (0.395, 0.0, 0.047)
BOARD_QUAT_WXYZ = (0.0, 0.7071067811865476, 0.7071067811865475, 0.0)

# This quat is R = [[0,1,0],[1,0,0],[0,0,-1]]: the board's local +z axis
# points DOWN in world (maps to world -z). See the module report / --check
# output for whether the printed face still ends up visible from above --
# short answer: yes, because it is a BOX, not a plane, so both its top and
# bottom faces carry the texture; only a single-sided plane geom would go
# dark from above with this quat.


def _legs_xml(table_length, table_width, table_height, slab_thickness,
              leg_thickness, leg_inset, group):
    leg_length = table_height - slab_thickness
    leg_half_z = leg_length / 2
    leg_center_z = -slab_thickness - leg_half_z
    leg_x = table_length / 2 - leg_inset
    leg_y = table_width / 2 - leg_inset
    leg_half = leg_thickness / 2

    legs = []
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        legs.append(
            f'      <geom name="table_leg{i}" type="box" group="{group}" '
            f'pos="{sx * leg_x:.4f} {sy * leg_y:.4f} {leg_center_z:.4f}" '
            f'size="{leg_half:.4f} {leg_half:.4f} {leg_half_z:.4f}" '
            f'rgba="0.35 0.35 0.38 1"/>'
        )
    return "\n".join(legs)


def _blocks_xml(group):
    blocks = []
    for spec in BLOCK_SPECS:
        hx, hy, hz = spec["half"]
        x, y = spec["pos_xy"]
        blocks.append(f"""
    <body name="{spec['name']}" pos="{x:.4f} {y:.4f} {hz:.4f}">
      <freejoint/>
      <geom name="{spec['name']}_geom" type="box" group="{group}"
            size="{hx:.4f} {hy:.4f} {hz:.4f}" rgba="{spec['rgba']}"/>
    </body>""")
    return "".join(blocks)


def build_scene_xml(include_robot: bool = True, camera_fovy=CAMERA_FOVY,
                     offwidth=None, offheight=None) -> str:
    """Return the full workcell MJCF as a string.

    See the module docstring for why this deliberately emits NO <compiler>
    element: every path here is either absolute (ChArUco texture) or, for the
    robot include, made to work by how the caller compiles the string (see
    _compile_model), not by anything declared inside this XML.

    camera_fovy overrides the depth_cam vertical field of view (degrees).
    offwidth/offheight, when both given, declare the offscreen framebuffer
    size on the <global> visual tag -- required whenever a caller will build
    a mujoco.Renderer larger than MuJoCo's 640x480 default (see
    mujoco_rgbd_node.build_scene for the same override and why).
    """
    slab_half_z = SLAB_THICKNESS / 2
    slab_center_z = -slab_half_z
    legs_xml = _legs_xml(TABLE_LENGTH, TABLE_WIDTH, TABLE_HEIGHT,
                         SLAB_THICKNESS, LEG_THICKNESS, LEG_INSET, GROUP_ENV)

    wall_half_x = (TABLE_LENGTH + 2 * WALL_MARGIN_X) / 2
    wall_half_y = WALL_THICKNESS / 2
    wall_half_z = WALL_HEIGHT / 2
    wall_pos_y = TABLE_WIDTH / 2 + WALL_GAP + wall_half_y
    wall_pos_z = -TABLE_HEIGHT + wall_half_z

    bottle_x, bottle_y = BOTTLE_POS_XY
    bottle_body_top_z = 2 * BOTTLE_BODY_HALF_HEIGHT
    bottle_neck_center_z = bottle_body_top_z + BOTTLE_NECK_HALF_HEIGHT

    blocks_xml = _blocks_xml(GROUP_CLUTTER)

    include_xml = f'  <include file="{UR5E_NOKEY_NAME}"/>\n' if include_robot else ""

    if offwidth is not None and offheight is not None:
        global_xml = ('<global azimuth="120" elevation="-20" '
                      f'offwidth="{int(offwidth)}" offheight="{int(offheight)}"/>')
    else:
        global_xml = '<global azimuth="120" elevation="-20"/>'

    # Full-length "home" keyframe: 6 robot angles + one (pos, identity quat)
    # 7-vector per freejoint body, in worldbody declaration order -- see the
    # module docstring for why the vendored keyframe cannot simply be reused.
    keyframe_xml = ""
    if include_robot:
        free_body_home = [
            (GLASS_POS_XY[0], GLASS_POS_XY[1], GLASS_HALF_HEIGHT),
            (bottle_x, bottle_y, BOTTLE_BODY_HALF_HEIGHT),
        ] + [(spec["pos_xy"][0], spec["pos_xy"][1], spec["half"][2]) for spec in BLOCK_SPECS]
        free_qpos = " ".join(
            f"{x:.4f} {y:.4f} {z:.4f} 1 0 0 0" for x, y, z in free_body_home
        )
        keyframe_xml = f"""
  <keyframe>
    <key name="home" qpos="{HOME_QPOS_ROBOT} {free_qpos}" ctrl="{HOME_QPOS_ROBOT}"/>
  </keyframe>"""

    return f"""
<mujoco model="scene">

  <option gravity="0 0 -9.81"/>
{include_xml}
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    {global_xml}
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7"
             rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.1"/>

    <texture type="2d" name="charuco_tex" file="{CHARUCO_PNG}"/>
    <material name="charuco_mat" texture="charuco_tex" specular="0" shininess="0"/>
  </asset>

  <worldbody>

    <light pos="0 0 2.5" dir="0 0 -1" directional="true"/>

    <geom name="floor" type="plane" group="{GROUP_ENV}" pos="0 0 {-TABLE_HEIGHT}"
          size="4 4 0.05" material="groundplane"/>

    <!-- The table -- static (no joint), top surface at z = 0. -->
    <body name="table" pos="0 0 0">
      <geom name="table_top" type="box" group="{GROUP_ENV}"
            pos="0 0 {slab_center_z:.4f}"
            size="{TABLE_LENGTH / 2:.4f} {TABLE_WIDTH / 2:.4f} {slab_half_z:.4f}"
            rgba="0.72 0.60 0.44 1"/>
{legs_xml}
    </body>

    <!-- Backdrop wall, past the table's far (+y) edge from the camera's
         viewpoint, so the distance field gets a real background instead of
         empty space. Static, part of the environment. -->
    <body name="wall" pos="0 {wall_pos_y:.4f} {wall_pos_z:.4f}">
      <geom name="wall_geom" type="box" group="{GROUP_ENV}"
            size="{wall_half_x:.4f} {wall_half_y:.4f} {wall_half_z:.4f}"
            rgba="0.55 0.55 0.58 1"/>
    </body>

    <!-- Monitor: listed with the "clutter" in the task write-up, but it
         never moves, so by the environment/non-environment split it belongs
         with the table and wall -- group GROUP_ENV, no freejoint. -->
    <body name="monitor" pos="{MONITOR_POS_XY[0]:.4f} {MONITOR_POS_XY[1]:.4f} {MONITOR_HALF_Z:.4f}">
      <geom name="monitor_geom" type="box" group="{GROUP_ENV}"
            size="{MONITOR_HALF_X:.4f} {MONITOR_HALF_Y:.4f} {MONITOR_HALF_Z:.4f}"
            rgba="{MONITOR_RGBA}"/>
    </body>

    <site name="robot_mount" pos="0 0 0" size="0.035" rgba="0.9 0.2 0.2 1"/>

    <camera name="depth_cam" pos="{CAMERA_POS[0]} {CAMERA_POS[1]} {CAMERA_POS[2]}"
            fovy="{camera_fovy}" mode="targetbody" target="table"/>

    <!-- Drinking glass: one cylinder, free to be nudged by the arm. -->
    <body name="glass" pos="{GLASS_POS_XY[0]:.4f} {GLASS_POS_XY[1]:.4f} {GLASS_HALF_HEIGHT:.4f}">
      <freejoint/>
      <geom name="glass_geom" type="cylinder" group="{GROUP_CLUTTER}"
            size="{GLASS_RADIUS:.4f} {GLASS_HALF_HEIGHT:.4f}" rgba="{GLASS_RGBA}"/>
    </body>

    <!-- Bottle: body cylinder + a narrower neck cylinder, one rigid body. -->
    <body name="bottle" pos="{bottle_x:.4f} {bottle_y:.4f} {BOTTLE_BODY_HALF_HEIGHT:.4f}">
      <freejoint/>
      <geom name="bottle_body_geom" type="cylinder" group="{GROUP_CLUTTER}"
            size="{BOTTLE_BODY_RADIUS:.4f} {BOTTLE_BODY_HALF_HEIGHT:.4f}" rgba="{BOTTLE_RGBA}"/>
      <geom name="bottle_neck_geom" type="cylinder" group="{GROUP_CLUTTER}"
            pos="0 0 {bottle_neck_center_z - BOTTLE_BODY_HALF_HEIGHT:.4f}"
            size="{BOTTLE_NECK_RADIUS:.4f} {BOTTLE_NECK_HALF_HEIGHT:.4f}" rgba="{BOTTLE_RGBA}"/>
    </body>
{blocks_xml}

    <!-- ChArUco board: hardcoded pose from the real rig's poses.yaml
         (aruco_1, parent base_link == this world origin). NOT a freejoint --
         a calibration target is bolted at a known, fixed transform. -->
    <body name="charuco_board" pos="{BOARD_POS[0]} {BOARD_POS[1]} {BOARD_POS[2]}"
          quat="{BOARD_QUAT_WXYZ[0]} {BOARD_QUAT_WXYZ[1]:.16f} {BOARD_QUAT_WXYZ[2]:.16f} {BOARD_QUAT_WXYZ[3]}">
      <geom name="charuco_geom" type="box" group="{GROUP_CLUTTER}"
            size="{BOARD_WIDTH / 2:.4f} {BOARD_DEPTH / 2:.4f} {BOARD_THICKNESS / 2:.4f}"
            material="charuco_mat"/>
    </body>

  </worldbody>
{keyframe_xml}
</mujoco>
"""


def _compile_model(xml: str, include_robot: bool) -> mujoco.MjModel:
    """Compile the generated XML, working around the meshdir and keyframe traps.

    Without the robot there is nothing relative to resolve (the ChArUco
    texture path is absolute) and no keyframe-length conflict, so
    from_xml_string is fine. With the robot: write a copy of ur5e.xml with
    its incompatible <keyframe> stripped out, and the generated scene, both
    into ur5e.xml's own directory, and compile the scene from that path --
    see the module docstring for why both steps are necessary.
    """
    if not include_robot:
        return mujoco.MjModel.from_xml_string(xml)

    nokey_path = UR5E_DIR / UR5E_NOKEY_NAME
    scratch = UR5E_DIR / SCENE_SCRATCH_NAME
    ur5e_text = UR5E_XML.read_text()
    ur5e_nokey = re.sub(r"<keyframe>.*?</keyframe>\s*", "", ur5e_text, flags=re.S)
    assert "<keyframe" not in ur5e_nokey, "failed to strip ur5e.xml's keyframe"

    nokey_path.write_text(ur5e_nokey)
    scratch.write_text(xml)
    try:
        return mujoco.MjModel.from_xml_path(str(scratch))
    finally:
        scratch.unlink(missing_ok=True)
        nokey_path.unlink(missing_ok=True)


def compile_scene_model(include_robot: bool = True, camera_fovy=CAMERA_FOVY,
                         offwidth=None, offheight=None) -> mujoco.MjModel:
    """Build AND compile the workcell in one call -- the entry point other
    scripts should use.

    Do NOT do from_xml_string(build_scene_xml()) instead: the generated MJCF
    <include>s the vendored ur5e.xml, whose meshdir only resolves relative to
    a real file on disk, so compiling from a bare string cannot find the
    meshes. _compile_model handles that.
    """
    xml = build_scene_xml(include_robot=include_robot, camera_fovy=camera_fovy,
                           offwidth=offwidth, offheight=offheight)
    return _compile_model(xml, include_robot)


def apply_environment_only(scene_option, env_only: bool) -> None:
    """Mask geom groups so only the environment + robot render (clutter
    hidden), or all groups do.

    Works on any mujoco.MjvOption -- the viewer's live `viewer.opt`, or one
    handed to mujoco.Renderer for offscreen rendering (what the depth-camera
    ROS node will use).
    """
    if env_only:
        for g in range(mujoco.mjNGROUP):
            scene_option.geomgroup[g] = 1 if g in (GROUP_ENV, GROUP_ROBOT_VISUAL) else 0
    else:
        for g in range(mujoco.mjNGROUP):
            scene_option.geomgroup[g] = 1


def describe(model, data):
    """Print each geom's WORLD z-range. Must read data.geom_xpos (post
    mj_forward), not model.geom_pos, which is relative to the parent body."""
    mujoco.mj_forward(model, data)

    print(f"  bodies : {model.nbody}    geoms : {model.ngeom}    DOF : {model.nv}")
    print()
    print("  geom                world z-range (metres)")
    print("  " + "-" * 48)
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"<unnamed:{i}>"
        gtype = mujoco.mjtGeom(model.geom_type[i]).name.replace("mjGEOM_", "")
        cz = data.geom_xpos[i][2]
        size = model.geom_size[i]
        if gtype == "PLANE":
            print(f"  {name:<19} z = {cz:+.3f} (infinite plane)")
            continue
        elif gtype == "BOX":
            hz = size[2]
        elif gtype == "SPHERE":
            hz = size[0]
        elif gtype in ("CYLINDER", "ELLIPSOID"):
            hz = size[1] if gtype == "CYLINDER" else size[2]
        elif gtype == "CAPSULE":
            hz = size[1] + size[0]
        else:
            hz = float(size.max())
        print(f"  {name:<19} {cz - hz:+.3f} .. {cz + hz:+.3f}")


CLUTTER_BOTTOM_GEOMS = ["glass_geom", "bottle_body_geom",
                        "block_small_geom", "block_medium_geom", "block_flat_geom"]


def _clutter_z_ranges(model, data):
    """Return {geom_name: (z_min, z_max)} for the objects that must rest on
    the table (bottle_neck excluded -- it does not touch the table itself)."""
    out = {}
    mujoco.mj_forward(model, data)
    for name in CLUTTER_BOTTOM_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        cz = data.geom_xpos[gid][2]
        gtype = mujoco.mjtGeom(model.geom_type[gid]).name.replace("mjGEOM_", "")
        size = model.geom_size[gid]
        hz = size[2] if gtype == "BOX" else size[1]
        out[name] = (cz - hz, cz + hz)
    return out


def _check_robot(model, data):
    print("\n  -- robot checks --")
    expected_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    found = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
             for j in range(model.njnt)]
    missing = [j for j in expected_joints if j not in found]
    print(f"  joints found ({len(found)}): {found}")
    assert not missing, f"missing expected joints: {missing}"
    assert model.nu == 6, f"expected 6 actuators, found {model.nu}"
    print(f"  actuators: {model.nu}  (OK)")

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key_id >= 0, "no 'home' keyframe found"
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    ee_pos = data.site_xpos[site_id].copy()
    print(f"  attachment_site world pos @ home keyframe: "
          f"({ee_pos[0]:+.4f}, {ee_pos[1]:+.4f}, {ee_pos[2]:+.4f})")
    assert ee_pos[2] > 0, (
        "end-effector z <= 0 at home -- the arm is buried in the table, "
        "the mounting convention came out wrong"
    )
    print("  end-effector z > 0  (OK -- arm is above the table, not buried in it)")

    print("\n  -- stability check: stepping 300 steps --")
    before = _clutter_z_ranges(model, data)
    for _ in range(300):
        mujoco.mj_step(model, data)
    after = _clutter_z_ranges(model, data)
    for name in CLUTTER_BOTTOM_GEOMS:
        z0_min, _ = before[name]
        z1_min, z1_max = after[name]
        drift = z1_min - z0_min
        print(f"  {name:<18} z_min before={z0_min:+.4f}  after={z1_min:+.4f}  "
              f"(drift {drift:+.4f})")
        assert z1_min > -0.02, f"{name} sank through the table (z_min={z1_min:.4f})"
        assert z1_max < 2.0, f"{name} appears to have exploded (z_max={z1_max:.4f})"


def main():
    include_robot = "--no-robot" not in sys.argv

    xml = build_scene_xml(include_robot=include_robot)
    model = _compile_model(xml, include_robot)
    data = mujoco.MjData(model)

    print(f"Scene compiled OK (include_robot={include_robot}).\n")
    describe(model, data)

    print()
    ranges = _clutter_z_ranges(model, data)
    print("  clutter resting check (must start at +0.000):")
    all_zero = True
    for name, (zmin, zmax) in ranges.items():
        ok = abs(zmin) < 1e-6
        all_zero &= ok
        print(f"    {name:<18} {zmin:+.4f} .. {zmax:+.4f}  {'OK' if ok else 'NOT ON TABLE'}")
    assert all_zero, "at least one clutter object is not resting exactly at z=0"

    if include_robot:
        _check_robot(model, data)

    print("\n  -- build_scene_xml(include_robot=False) also compiles --")
    xml_no_robot = build_scene_xml(include_robot=False)
    model_no_robot = _compile_model(xml_no_robot, include_robot=False)
    print(f"  OK: {model_no_robot.nbody} bodies, {model_no_robot.ngeom} geoms, "
          f"{model_no_robot.nv} DOF (no robot)")

    print(f"\n  Table top surface : z = 0.000")
    print(f"  Floor             : z = {-TABLE_HEIGHT:.3f}")
    print(f"\n  Viewer key '{ENV_ONLY_TOGGLE_KEY}' toggles environment-only rendering.")

    if "--check" in sys.argv:
        return

    print("\nOpening viewer -- close the window to exit.")
    print(f"Press '{ENV_ONLY_TOGGLE_KEY}' to toggle environment-only rendering.")

    state = {"env_only": False}
    viewer_ref = {}

    def key_callback(keycode):
        if keycode == ord(ENV_ONLY_TOGGLE_KEY):
            state["env_only"] = not state["env_only"]
            v = viewer_ref.get("viewer")
            if v is not None:
                apply_environment_only(v.opt, state["env_only"])
                print(f"[toggle] environment_only = {state['env_only']}")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer_ref["viewer"] = viewer
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            leftover = model.opt.timestep - (time.time() - step_start)
            if leftover > 0:
                time.sleep(leftover)


if __name__ == "__main__":
    main()
