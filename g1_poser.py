
import math
import time
from pathlib import Path
from loop_rate_limiters import RateLimiter

import mujoco
import mujoco.viewer

_HERE = Path(__file__).parent
_XML = _HERE / "g1_description" / "g1.xml"


def main() -> None:
    """A minimal script to pose the G1 humanoid without physics."""
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Disable gravity.
    # model.opt.gravity = (0.0, 0.0, 0.0)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Get joint IDs.
        left_shoulder_pitch_id = model.joint("left_shoulder_pitch_joint").id
        right_shoulder_pitch_id = model.joint("right_shoulder_pitch_joint").id

        rate = RateLimiter(frequency=60.0, warn=False)
        start_time = time.time()
        while viewer.is_running():
            # Animate the arms.
            amplitude = 0.5
            frequency = 0.5
            elapsed_time = time.time() - start_time
            angle = amplitude * math.sin(2 * math.pi * frequency * elapsed_time)
            data.qpos[left_shoulder_pitch_id] = angle
            data.qpos[right_shoulder_pitch_id] = -angle

            # Update the kinematics of the model.
            mujoco.mj_forward(model, data)

            viewer.sync()

            rate.sleep()


if __name__ == "__main__":
    main()
