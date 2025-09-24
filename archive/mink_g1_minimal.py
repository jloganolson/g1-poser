from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink


_HERE = Path(__file__).parent
# Load a scene that includes g1 plus a mocap target for the right hand
_XML = _HERE / "g1_description" / "scene_g1_targets.xml"


if __name__ == "__main__":
    # Load the scene with mocap target(s).
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    # Build configuration and define a minimal set of tasks that do not rely on mocap targets.
    configuration = mink.Configuration(model)

    tasks = [
        pelvis_orientation_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        # Anchor pelvis position so moving a hand target doesn't translate the whole body.
        pelvis_position_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=10.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        ),
        torso_orientation_task := mink.FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        # Keep joints near their current configuration.
        posture_task := mink.PostureTask(model, cost=1e-1),
    ]

    # Add a right hand task to follow the mocap target
    right_hand_task = mink.FrameTask(
        frame_name="right_palm",
        frame_type="site",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=1.0,
    )
    tasks.append(right_hand_task)

    # Keep the lower arm (forearm) level by constraining elbow link orientation
    right_forearm_orientation_task = mink.FrameTask(
        frame_name="right_elbow_link",
        frame_type="body",
        position_cost=0.0,
        orientation_cost=2.0,
        lm_damping=1.0,
    )
    tasks.append(right_forearm_orientation_task)

    # Use only basic configuration limits for the smallest working example.
    limits = [mink.ConfigurationLimit(model)]

    model = configuration.model
    data = configuration.data

    # Keep the default solver from mink or use a common one if available.
    # If DAQP is installed, you can set solver = "daqp" for better performance.
    solver = "daqp"

    # Launch the passive viewer.
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=True, show_right_ui=True
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize task targets from the current configuration.
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)
        pelvis_position_task.set_target_from_configuration(configuration)
        torso_orientation_task.set_target_from_configuration(configuration)
        right_forearm_orientation_task.set_target_from_configuration(configuration)

        # Move the mocap right_palm_target to the current right_palm site pose
        right_palm_mid = model.body("right_palm_target").mocapid[0]
        mink.move_mocap_to_frame(model, data, "right_palm_target", "right_palm", "site")

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update right hand task target from mocap body pose
            right_hand_task.set_target(mink.SE3.from_mocap_id(data, right_palm_mid))

            # Solve a small IK step that tries to hold posture/orientation and reach the hand target
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)

            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()


