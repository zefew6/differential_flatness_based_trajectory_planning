"""
multi-particle scene generator, randomly generated multi-particle scene.xml
"""

import numpy as np
import mujoco
import mujoco.viewer


def pusher_template_generator(robot_id, pos):
    pusher_template = f"""

            
            <body name="{robot_id}" pos="{pos}">

            <freejoint/>
            <geom name="base_{robot_id}"
                    type="cylinder"
                    pos="0 0 0.03"
                    size="0.15 0.03"    
                    rgba="1.0 0.6 0.2 1"/>

            <geom type="box" pos="0.16 0 0.03" size="0.01 0.14 0.05" rgba="1 1 1 1" />

            <geom name="front wheel_{robot_id}" pos="0.12 0 -.015" type="sphere" size=".015" condim="1" priority="1"/>

            <body name="left wheel_{robot_id}" pos="-.02 .16 0" zaxis="0 1 0">
            <joint name="left_{robot_id}"/>
            <geom class="wheel"/>
            <site class="decor" size=".006 .025 .012"/>
            <site class="decor" size=".025 .006 .012"/>
            </body>

            <body name="right wheel_{robot_id}" pos="-.02 -.16 0" zaxis="0 1 0">
            <joint name="right_{robot_id}"/>
            <geom class="wheel"/>
            <site class="decor" size=".006 .025 .012"/>
            <site class="decor" size=".025 .006 .012"/>
            </body>

        </body>
    """
    return pusher_template


    

def empty_world_template(particles_str, robots_str, joint_str):
    return f"""
<mujoco>
  <compiler meshdir="asset" texturedir="asset" angle="degree" coordinate="local"/>
  <option timestep="0.05" integrator="implicitfast"/>
  <statistic meansize=".05"/>

  <visual>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="4096"/>
    <map stiffness="700" shadowscale="0.5" fogstart="1" fogend="15" zfar="40" haze="1"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="texplane" type="2d" builtin="checker" rgb1="0 0 0" rgb2="0 0 0"
      width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>

    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="10 10" texuniform="true"/>
  </asset>


  <default>
        <joint damping=".03" actuatorfrcrange="-0.5 0.5"/>
        <default class="wheel">
            <geom type="cylinder" size=".03 .01" rgba=".1 1 .1 1"/>
        </default>
        <default class="decor">
            <site type="box" rgba=".5 1 .5 1"/>
        </default>
    </default>

  <worldbody>

    <geom name="ground" type="plane" size="0 0 1" pos="0 0 0" quat="1 0 0 0" material="matplane" condim="1"/>
    <camera name="topdown" pos="0 0 6" quat="1 0 0 0"/>

    {particles_str}

    {robots_str}
  
  </worldbody>


    {joint_str}

</mujoco>
"""


def particle_body_template(name, type, pos, friction, size, rgba, density):
    return f"""
<body name="{name}" pos="{pos}">
      <freejoint/>
      <geom type="{type}" size="{size}" rgba="{rgba}" friction="{friction}" density="{density}"/>
</body>
"""


def joint_template(robot_id):
    return f"""
<tendon>
  <fixed name="forward_{robot_id}">
    <joint joint="left_{robot_id}" coef=".5"/>
    <joint joint="right_{robot_id}" coef=".5"/>
  </fixed>
  <fixed name="turn_{robot_id}">
    <joint joint="left_{robot_id}" coef="-.5"/>
    <joint joint="right_{robot_id}" coef=".5"/>
  </fixed>
</tendon>

<actuator>
  <motor name="forward_{robot_id}" tendon="forward_{robot_id}" ctrlrange="-1 1"/>
  <motor name="turn_{robot_id}" tendon="turn_{robot_id}" ctrlrange="-1 1"/>
</actuator>


<sensor>
    <jointactuatorfrc name="right_{robot_id}" joint="right_{robot_id}"/>
    <jointactuatorfrc name="left_{robot_id}" joint="left_{robot_id}"/>
</sensor>
"""
    


def multi_particle_generator(num, pos_range):
    """
    range: 3x2, 3: xyz, 2:[min, max]
    """

    pi_xml = "\n"

    for i in range(num):
        particle_name = f"p{i}"
        rand_pos = np.array([
            np.random.uniform(low=pos_range[0][0], high=pos_range[0][1]),
            np.random.uniform(low=pos_range[1][0], high=pos_range[1][1]),
            np.random.uniform(low=pos_range[2][0], high=pos_range[2][1]),
        ])
        rand_pos_str = f"{rand_pos[0]} {rand_pos[1]} {rand_pos[2]}"
        type = "box"
        size = "0.03 0.03 0.03"
        rgba = "0.8 0.5 0.6 1"
        friction = "1.0 0.1 0.001"
        density = "5"

        pi_xml += particle_body_template(particle_name, type, rand_pos_str, friction, size, rgba, density)

        pi_xml += "\n"

    return pi_xml



def multi_robot_generator(num, pos):

    robot_xml = "\n"
    joint_xml = "\n"

    for i in range(num):
        robot_name = f"{i}"

        pos_str = f"{pos[i][0]} {pos[i][1]} {pos[i][2]}"
        robot_xml += pusher_template_generator(robot_name, pos_str)

        robot_xml += "\n"

        joint_xml += joint_template(robot_name)
        joint_xml += "\n"

    return robot_xml, joint_xml



def scene_generator(num_particles, particle_pos_range, num_robot, robot_pos_range):
    robot_xml, joint_xml = multi_robot_generator(num_robot, robot_pos_range)
    particle_xml = multi_particle_generator(num_particles, particle_pos_range)

    return empty_world_template(particle_xml, robot_xml, joint_xml)



def save_scene_xml(file_path, scene_xml):
    with open(file_path, "w") as f:
        f.write(scene_xml)
    print(f"scene xml saved to {file_path}")



######## main function
if __name__ == "__main__":
    particle_pos_range = np.array([
        [-1.4, 1.4],
        [-1.4, 1.4],
        [0.02, 0.025]
    ])
    robot_pos = np.array([
        [2.0, 1, 0.5],
        [-2.0, -1, 0.5],
        [-3.0, -2, 0.5],
        [2.0, -1, 0.5],
        [1.0, 3, 0.5]
    ])
    num_bot = robot_pos.shape[0]
    num_par = 170
    scene_xml = scene_generator(num_par, particle_pos_range, num_bot, robot_pos)

    #print(scene_xml)

    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

    save_scene_xml(f"assets/scene_p{num_par}_r{num_bot}.xml", scene_xml)
    print(f"Scene XML generated successfully. Save to assets/scene_p{num_par}_r{num_bot}.xml")

