"""
==============================================================================
constants.py - Task constants of the vertical-pipe task
==============================================================================
Copied 1:1 from the MuJoCo version:
    D:\\mujoco\\Continuum_MuJoCo\\task_vertical_pipe\\vertical_pipe_env.py
    D:\\mujoco\\Continuum_MuJoCo\\continuum_algorithm.py   (motor-angle conversion)
plus the physics settings that vertical_pipe_env.py sets in code (_prepare_robot,
_add_pipe, _configure_model). Nothing here depends on Isaac Sim or MuJoCo.
==============================================================================
"""

import os

import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(TASK_DIR, "assets")
SOURCE_MJCF = os.path.join(ASSETS_DIR, "mjcf", "ContinuumRobot_Native.xml")
GENERATED_DIR = os.path.join(ASSETS_DIR, "generated")

# ---------------------------------------------------------------------------
# Rig geometry (world frame, metres)
# ---------------------------------------------------------------------------
ELEV_RANGE = (-0.211, 0.112)          # Elevator_Joint range used by the task, + is up
TIP_OFFSET = 0.0275                   # Seg15 body origin -> physical tip face
PLATE_TOP_Z = 0.4572                  # upper face of the frame's top plate
PLATE_THICKNESS = 0.030
PLATE_HOLE_XY = np.array([0.0369, 0.2316])
PLATE_HOLE_RADIUS = 0.0345

# Collision cylinder of each segment: (radius, z_start, z_end) in the Seg body frame,
# whose +z runs down the robot.
SEG_COLLIDERS = {}
for _i in range(1, 16):
    if _i <= 4:
        SEG_COLLIDERS[_i] = (0.044, -0.002, 0.040)
    elif _i == 5:
        SEG_COLLIDERS[_i] = (0.042, -0.002, 0.040)
    elif _i <= 10:
        SEG_COLLIDERS[_i] = (0.034, -0.002, 0.038)
    elif _i == 11:
        SEG_COLLIDERS[_i] = (0.0244, -0.002, 0.031)
    elif _i <= 14:
        SEG_COLLIDERS[_i] = (0.0244, -0.002, 0.036)
    else:
        SEG_COLLIDERS[_i] = (0.0244, -0.002, TIP_OFFSET)
TIP_DISC_RADIUS = 0.0244

# Eye-in-hand camera on Seg15 (vision_env.py): on the tip axis, looking out of the tip face
CAM_WIDTH = CAM_HEIGHT = 160
CAM_FOV_DEG = 120.0                   # wide-angle endoscope camera
CAM_OFFSET = TIP_OFFSET + 0.0005      # camera centre 0.5 mm outside the tip face

# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------
SIM_DT = 0.002           # MuJoCo opt.timestep
N_SUBSTEPS = 10          # 10 x 2 ms = 20 ms per env step (50 Hz control) -> Isaac Lab decimation
BEND_RATE = np.array([0.008, 0.012, 0.016])
ELEV_RATE = 0.0015       # m per step (7.5 cm/s)
THETA_MAX = 1.2          # max total bend per section (0.24 rad per joint, limit 0.436)
ELEV_KP, ELEV_KV = 3.0e4, 1.2e3
BEND_KP, BEND_KV = 100.0, 2.0          # the 30 bending servos (gainprm / biasprm in _prepare_robot)
JOINT_ARMATURE = 0.01                  # m.dof_armature[:] = 0.01
JOINT_DAMPING = 0.5                    # m.dof_damping[:] = 0.5
BEND_FORCE_LIMIT = 20.0                # actuatorfrcrange of the Seg joints in the MJCF
ELEV_FORCE_LIMIT = 1.0e7               # actuatorfrcrange of Elevator_Joint in the MJCF
N_STAVES = 24            # the pipe wall is a ring of box staves (collision)
EXIT_MARGIN = 0.02       # the tip must pass this far below the bottom of the bore
START_GAP = (0.02, 0.07)  # initial tip height above the pipe top
PIPE_WALL = 0.004
PIPE_HEIGHT = 0.07
CONTACT_FRICTION = 0.3   # sliding friction of the colliders (condim 3: torsional/rolling unused)

# ---------------------------------------------------------------------------
# Reward: potential-based progress + event terms
#   Phi = -(W_LAT*lat_tip/10cm + W_LAT_S3*lat_s3/10cm + W_TILT*tilt/0.5rad + W_HEIGHT*h_left/10cm)
# ---------------------------------------------------------------------------
W_LAT = 4.0
W_LAT_S3 = 2.0
W_TILT = 1.0
W_HEIGHT = 4.0
R_SUCCESS = 100.0
R_FAIL = -30.0
R_CONTACT = -0.1
R_TIME = -0.01
W_SMOOTH = 0.002

STAGE_NAMES = ("ALIGN", "INSERT", "EXIT")
PIPE_MODES = ("rig", "random")
# failure codes used on the GPU side (strings only exist in the info dicts)
FAILURE_NAMES = ("", "unstable", "rim_hit", "missed_pipe")

# ---------------------------------------------------------------------------
# Lead-screw motor conversion (continuum_algorithm.py) - only used for info / HUD
# ---------------------------------------------------------------------------
SECTION_LENGTHS = np.array([0.185, 0.159, 0.175])
CABLE_RADIUS = np.array([0.040, 0.030, 0.020])
LEAD_SCREW_PITCH = 0.002
MOTOR_ALPHAS = np.deg2rad([0.0, 90.0, 30.0, 300.0, 150.0, 240.0])


def K_to_Length_3Seg(k1, phi1, k2, phi2, k3, phi3):
    """Bend of the 3 sections -> 6 lead-screw motor angles (deg), as continuum_algorithm.py."""
    L, r, alpha = SECTION_LENGTHS, CABLE_RADIUS, MOTOR_ALPHAS
    K, Phi = [k1, k2, k3], [phi1, phi2, phi3]
    deltaL = np.zeros(6)
    for i in range(2):
        deltaL[i] = -K[0] * L[0] * r[0] * np.cos(Phi[0] - alpha[i])
    for i in range(4, 6):
        deltaL[i] = (-K[0] * L[0] * r[1] * np.cos(Phi[0] - alpha[i])
                     - K[1] * L[1] * r[1] * np.cos(Phi[1] - alpha[i]))
    for i in range(2, 4):
        deltaL[i] = (-K[0] * L[0] * r[2] * np.cos(Phi[0] - alpha[i])
                     - K[1] * L[1] * r[2] * np.cos(Phi[1] - alpha[i])
                     - K[2] * L[2] * r[2] * np.cos(Phi[2] - alpha[i]))
    return (deltaL / LEAD_SCREW_PITCH) * 360.0


def inner_contact_radius(pipe_radius=PLATE_HOLE_RADIUS):
    """Contacts farther out than the polygon corners are on the rim / outside."""
    return pipe_radius / np.cos(np.pi / N_STAVES) + 0.0015


def bore_length(pipe_mode, pipe_height=PIPE_HEIGHT):
    """In rig mode the bore continues through the plate hole below the pipe."""
    return pipe_height + (PLATE_THICKNESS if pipe_mode == "rig" else 0.0)
