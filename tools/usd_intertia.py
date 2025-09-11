import numpy as np
from scipy.spatial.transform import Rotation

# 1. Define the full inertia tensor from MuJoCo
full_inertia = np.array([
    5.02585e-05,   # Ixx
    0.000444834,   # Iyy
    0.000444931,   # Izz
    -7.91587e-07,  # Ixy
    1.13798e-06,   # Ixz
    -1.20104e-07   # Iyz
])

# 2. Construct the 3x3 inertia matrix
I = np.array([
    [full_inertia[0], full_inertia[3], full_inertia[4]],
    [full_inertia[3], full_inertia[1], full_inertia[5]],
    [full_inertia[4], full_inertia[5], full_inertia[2]]
])

# 3. Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(I)

# The eigenvalues are the diagonal inertia
diagonal_inertia = eigenvalues

# The eigenvectors form a rotation matrix for the principal axes
# We can convert this matrix to a quaternion or Euler angles for USD
rotation = Rotation.from_matrix(eigenvectors)
principal_axes_quat = rotation.as_quat() # (x, y, z, w) format
principal_axes_euler = rotation.as_euler('xyz', degrees=True)

# 4. Print the results for USD
print(f"Mass: 0.106")
print(f"Center of Mass: (0.111202, -0.000072, 0.000102)")
print(f"Diagonal Inertia (Eigenvalues): {diagonal_inertia}")
print(f"Principal Axes (Quaternion xyzw): {principal_axes_quat}")
print(f"Principal Axes (Euler angles XYZ in degrees): {principal_axes_euler}")