import mujoco
import mujoco.viewer

# A tiny "workbench": ground plane + a sphere obstacle + a cylinder obstacle
XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom name="head_sphere" type="sphere" size="0.15" rgba="0.9 0.3 0.3 0.6"/>
    </body>
    <body pos="0.5 0 0.4">
      <freejoint/>
      <geom name="arm_cylinder" type="cylinder" size="0.08 0.25" rgba="0.3 0.5 0.9 0.6"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)  # opens an interactive window; close it to exit