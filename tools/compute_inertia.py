import numpy as np

def add_inertial(mass: float, com: np.ndarray, inertia: np.ndarray):
    # Ensuring epsilon masses and inertias
    mass = max(1e-9, mass)
    inertia[0, 0] = max(1e-9, inertia[0, 0])
    inertia[1, 1] = max(1e-9, inertia[1, 1])
    inertia[2, 2] = max(1e-9, inertia[2, 2])

    # Populating body inertial properties
    # https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-inertial
    inertial_str: str = "<inertial " # Renamed to avoid conflict with inertia array
    inertial_str += 'pos="%g %g %g" ' % tuple(com)
    inertial_str += 'mass="%g" ' % mass
    inertial_str += 'fullinertia="%g %g %g %g %g %g" ' % (
        inertia[0, 0],
        inertia[1, 1],
        inertia[2, 2],
        inertia[0, 1],
        inertia[0, 2],
        inertia[1, 2],
    )
    inertial_str += " />"
    print(inertial_str)

# --- Variables from Screenshot (converted to MuJoCo standard units) ---

# Mass: 106 g -> 0.106 kg
mass = 0.106

# Center of Mass (COM): (111.202, -0.072, 0.102) mm -> meters
com = np.array([0.111202, -0.000072, 0.000102])

# Mass Moments of Inertia: g mm^2 -> kg m^2 (multiply by 1e-9)
# Onshape: [Lxx, Lxy, Lxz], [Lyx, Lyy, Lyz], [Lzx, Lzy, Lzz]
inertia = np.array([
    [50258.52, -791.587, 1137.984],
    [-791.587, 444833.684, -120.104],
    [1137.984, -120.104, 444930.92]
]) * 1e-9

# --- Function Call ---
add_inertial(mass=mass, com=com, inertia=inertia)