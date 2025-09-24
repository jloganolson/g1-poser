from utils import (
    init_cmd_hg,
    create_damping_cmd,
    create_zero_cmd,
    MotorMode,
    NonBlockingInput,
    joint2motor_idx,
    Kp,
    Kd,
    G1_NUM_MOTOR,
    default_pos,
    default_angles_config,
    get_gravity_orientation,
    action_scale,
    RESTRICTED_JOINT_RANGE,
    G1MjxJointIndex,
    G1PyTorchJointIndex,
    pytorch2mujoco_idx,
    mujoco2pytorch_idx,
    init_joint_mappings,
    remap_pytorch_to_mujoco,
    remap_mujoco_to_pytorch,
)
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
import time
import numpy as np
import json
import tkinter as tk
from tkinter import ttk
 

NETWORK_CARD_NAME = 'enxc8a362b43bfd'
# Path to a poses JSON exported by pose-tool.py (edit this to your file)
POSES_JSON_PATH = '/home/logan/Projects/g1-poser/output/poses_20250908_075031.json'

# Default lerp duration (seconds) when activating a pose
DEFAULT_LERP_DURATION = 2.0



class Controller:
    def __init__(self, smooth_start: bool = True, smooth_start_duration: float = 1.0) -> None:

        self.qj = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self.dqj = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self.action = np.zeros(G1_NUM_MOTOR, dtype=np.float32)  # In MuJoCo order
        self.last_action_pytorch = np.zeros(G1_NUM_MOTOR, dtype=np.float32)  # In PyTorch order
        self.counter = 0

        # Convert joint range tuples to numpy arrays for efficient clamping
        joint_limits = np.array(RESTRICTED_JOINT_RANGE, dtype=np.float32)
        self._joint_lower_bounds = joint_limits[:, 0]
        self._joint_upper_bounds = joint_limits[:, 1]

        self.control_dt = 0.02

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        self.default_pos_array = np.array(default_pos)
        self.default_angles_array = np.array(default_angles_config)
        self.start_time = time.time()

        # Smooth start (optional) configuration
        self._smooth_start_enabled = bool(smooth_start)
        self._smooth_start_duration = float(smooth_start_duration)
        self._smooth_start_t0 = None
        self._smooth_start_initial_targets = None

        # Pose/lerp state
        self.poses: list[np.ndarray] = []  # list of absolute target joint arrays (len 23)
        self._lerp_active = False
        self._lerp_t0 = 0.0
        self._lerp_duration = float(DEFAULT_LERP_DURATION)
        self._lerp_start: np.ndarray | None = None
        self._lerp_target: np.ndarray | None = None
        # Last commanded (hold) target when not lerping
        self._hold_target = self.default_angles_array.copy()

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def load_poses_from_json(self, path: str) -> None:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load poses JSON: {e}")
            self.poses = []
            return

        poses: list[np.ndarray] = []
        for item in data.get('poses', []):
            joints_dict = item.get('joints', {}) or {}
            target = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
            try:
                # keys may be strings in JSON
                for k, v in joints_dict.items():
                    try:
                        jid = int(k)
                    except Exception:
                        continue
                    # Pose tool uses MuJoCo joint ids; free joint is id 0.
                    # Our 23 actuated joints are ids 1..23 -> indices 0..22.
                    jid0 = jid - 1
                    if 0 <= jid0 < G1_NUM_MOTOR:
                        target[jid0] = float(v)
                # Clamp to joint limits
                target = np.clip(target, self._joint_lower_bounds, self._joint_upper_bounds)
                poses.append(target)
            except Exception as e:
                print(f"Skipping pose due to error: {e}")
        self.poses = poses
        print(f"Loaded {len(self.poses)} poses from {path}")

    def start_lerp_to_target(self, target: np.ndarray, duration: float | None = None) -> None:
        # Build current absolute joint vector from sensors
        current_q = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        for i in range(G1_NUM_MOTOR):
            current_q[i] = self.low_state.motor_state[joint2motor_idx[i]].q
        tgt = np.clip(target.astype(np.float32), self._joint_lower_bounds, self._joint_upper_bounds)
        self._lerp_start = current_q
        self._lerp_target = tgt
        self._lerp_duration = float(duration) if duration is not None else self._lerp_duration
        self._lerp_t0 = time.time()
        self._lerp_active = True
        self._hold_target = tgt.copy()
        # Disable any smooth-start if a manual lerp begins
        self._smooth_start_enabled = False
        self._smooth_start_initial_targets = None
        print("Starting lerp to target pose")

    def start_lerp_to_pose_index(self, idx: int, duration: float | None = None) -> None:
        if not (0 <= idx < len(self.poses)):
            print(f"Invalid pose index: {idx}")
            return
        self.start_lerp_to_target(self.poses[idx], duration)

    def reset_oscillator_phase(self) -> None:
        """Reset the internal oscillator phase to avoid jump at start."""
        self.start_time = time.time()

    def send_cmd(self, cmd: LowCmd_):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Press Enter to continue...")
        with NonBlockingInput() as nbi:
            while not nbi.check_key('\n'):
                create_zero_cmd(self.low_cmd)
                self.send_cmd(self.low_cmd)
        print("Zero torque state confirmed. Proceeding...")

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.control_dt)

        init_dof_pos = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        for i in range(G1_NUM_MOTOR):
            init_dof_pos[i] = self.low_state.motor_state[joint2motor_idx[i]].q

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(G1_NUM_MOTOR):
                motor_idx = joint2motor_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * \
                    (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = Kp[j]
                self.low_cmd.motor_cmd[motor_idx].kd = Kd[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Press Enter to start the controller...")
        with NonBlockingInput() as nbi:
            while not nbi.check_key('\n'):  # Check for Enter key
                # Keep sending default position commands while waiting
                for i in range(len(joint2motor_idx)):
                    motor_idx = joint2motor_idx[i]
                    self.low_cmd.motor_cmd[motor_idx].q = default_pos[i]
                    self.low_cmd.motor_cmd[motor_idx].qd = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = Kp[i]
                    self.low_cmd.motor_cmd[motor_idx].kd = Kd[i]
                    self.low_cmd.motor_cmd[motor_idx].tau = 0

                self.send_cmd(self.low_cmd)
                time.sleep(self.control_dt)
        print("Default position state confirmed. Starting controller...")

    def get_obs(self) -> np.ndarray:
        """Construct observation in the format expected by the PyTorch model."""
        # Get projected gravity (3 dimensions)
        quat = self.low_state.imu_state.quaternion
        gravity = get_gravity_orientation(quat)
        
        # Get velocity commands (3 dimensions)
        velocity_commands = self._controller.get_command()
        
        # Get joint positions and velocities in MuJoCo order, then convert to PyTorch order
        joint_pos_mujoco = self.qj.copy()  # Already relative to default_angles_config
        joint_vel_mujoco = self.dqj.copy()
        
        # Convert to PyTorch model joint order for the observation
        joint_pos_pytorch = remap_mujoco_to_pytorch(joint_pos_mujoco)
        joint_vel_pytorch = remap_mujoco_to_pytorch(joint_vel_mujoco)
        
        # Concatenate all observations: 3 + 3 + 23 + 23 + 23 = 75
        obs = np.hstack([
            gravity,
            velocity_commands, 
            joint_pos_pytorch,
            joint_vel_pytorch,
            self.last_action_pytorch,
        ])
        
        return obs.astype(np.float32)

    def run(self):
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(G1_NUM_MOTOR):
            self.qj[i] = self.low_state.motor_state[joint2motor_idx[i]].q - self.default_angles_array[i]
            self.dqj[i] = self.low_state.motor_state[joint2motor_idx[i]].dq

        # Initialize smooth-start reference on the first run tick
        if self._smooth_start_enabled and self._smooth_start_t0 is None:
            self._smooth_start_t0 = time.time()
            current_q = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
            for i in range(G1_NUM_MOTOR):
                current_q[i] = self.low_state.motor_state[joint2motor_idx[i]].q
            self._smooth_start_initial_targets = current_q

        # Determine target command for this tick
        motor_targets_unclamped = self._hold_target.copy()

        # If a manual lerp is active, interpolate from current to target
        if self._lerp_active and self._lerp_start is not None and self._lerp_target is not None:
            elapsed = time.time() - self._lerp_t0
            dur = max(1e-3, self._lerp_duration)
            t = min(1.0, max(0.0, elapsed / dur))
            motor_targets_unclamped = (1.0 - t) * self._lerp_start + t * self._lerp_target
            if t >= 1.0:
                self._lerp_active = False
                self._hold_target = self._lerp_target.copy()
                print("Lerp complete")

        # Clamp motor targets to joint limits and check for clamping
        motor_targets = np.clip(
            motor_targets_unclamped, self._joint_lower_bounds, self._joint_upper_bounds
        )
        # Optional: print clamping warnings (commented to avoid spam)
        # clamped_indices = np.where(motor_targets != motor_targets_unclamped)[0]
        # if clamped_indices.size > 0:
        #     print("WARNING: Clamping motor targets for joints:")
        #     for idx in clamped_indices:
        #         print(f"  Joint {idx}: {motor_targets_unclamped[idx]:.3f} -> {motor_targets[idx]:.3f}")

        # Apply optional smooth-start interpolation (lerp) from current pose to target
        if self._smooth_start_enabled and self._smooth_start_t0 is not None:
            elapsed = time.time() - self._smooth_start_t0
            if elapsed < self._smooth_start_duration:
                alpha = elapsed / self._smooth_start_duration
                motor_targets = (1.0 - alpha) * self._smooth_start_initial_targets + alpha * motor_targets
                # Ensure final command stays within limits
                motor_targets = np.clip(
                    motor_targets, self._joint_lower_bounds, self._joint_upper_bounds
                )
            else:
                # Transition complete
                self._smooth_start_enabled = False
                self._smooth_start_initial_targets = None

        # Build low cmd
        for i in range(G1_NUM_MOTOR):
            motor_idx = joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = motor_targets[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = Kp[i]
            self.low_cmd.motor_cmd[motor_idx].kd = Kd[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)
        time.sleep(self.control_dt)


if __name__ == "__main__":
    print("Running pose activator.")
    print("WARNING: Ensure a clear area around the robot.")
    # Initial prompt doesn't need non-blocking


    ChannelFactoryInitialize(0, NETWORK_CARD_NAME)

    controller = Controller(smooth_start=True, smooth_start_duration=1.0)
    # Load poses JSON
    controller.load_poses_from_json(POSES_JSON_PATH)

    # Enter the zero torque state, press Enter key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # # Enter the default position state, press Enter key to continue executing
    # controller.default_pos_state()

    input("Press Enter to begin running the controller.")

    # Small Tk GUI with one button per pose
    root = tk.Tk()
    root.title("G1 Poses")
    frame = ttk.Frame(root, padding=8)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text=f"Loaded poses: {len(controller.poses)}", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")

    btns = ttk.Frame(frame)
    btns.pack(fill="x", pady=(6, 4))

    for i in range(len(controller.poses)):
        ttk.Button(btns, text=f"Pose {i+1}", command=lambda idx=i: controller.start_lerp_to_pose_index(idx, DEFAULT_LERP_DURATION)).grid(row=i // 4, column=i % 4, padx=4, pady=4, sticky="ew")

    # Default pose button
    ttk.Button(frame, text="Default Pose", command=lambda: controller.start_lerp_to_target(controller.default_angles_array, DEFAULT_LERP_DURATION)).pack(anchor="w", pady=(6, 0))

    # Resize columns
    for c in range(4):
        btns.grid_columnconfigure(c, weight=1)

    # Reset oscillator phase not needed; we drive by poses

    print("Controller running. Press 'q' to quit.")
    with NonBlockingInput() as nbi:  # Use context manager for the main loop
        while True:
            # Update GUI
            try:
                root.update_idletasks()
                root.update()
            except Exception:
                pass

            controller.run()
            # Check for 'q' key press to exit
            if nbi.check_key('q'):
                print("\n'q' pressed. Exiting loop...")
                break
            # Add a small sleep to prevent busy-waiting if controller.run() is very fast
            time.sleep(0.001)

    print("Returning to zero torque state...")
    create_zero_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)

    print("Exit") 