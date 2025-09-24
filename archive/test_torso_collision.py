import mujoco
import mujoco.viewer as viewer


def load_callback(model=None, data=None):
    # Load the test scene that includes the g1 model and a floor
    model = mujoco.MjModel.from_xml_path('./g1_description/scene_torso_collision_test.xml')
    data = mujoco.MjData(model)

    # Set to keyframe where robot is on its back
    try:
        on_back_id = model.key('on_back').id
        mujoco.mj_resetDataKeyframe(model, data, on_back_id)
    except Exception:
        # Fallback: just reset data
        mujoco.mj_resetData(model, data)

    model.opt.timestep = 0.005
    return model, data


if __name__ == '__main__':
    viewer.launch(loader=load_callback)


