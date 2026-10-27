# Neural Network
ARCHITECTURE = "4x128"
ACTIVATION = "Tanh"

# Training
EPOCHS = 10000
LR = 1e-3
OPTIMIZER = "Adam"
REPEATS = 3

# Training Points
N_F = 4096       # PDE collocation points
N_IC = 256       # Initial-condition points
N_BC = 128       # Boundary-condition pairs (256 locations)

# Evaluation
NX_EVAL = 200
NT_EVAL = 200    # Total evaluation points = 40,000

# Reference Solver
NX_SPEC = 512
NT_REF = 500
SOLVER = "DOP853"
RTOL = 1e-9
ATOL = 1e-11

# PDE Parameters
ALPHA = 1.0
BETA = 0.0025
A_SOL = 1.0
X0_SOL = 0.0

# Domain
X_LEFT = -1.0
X_RIGHT = 1.0
T_FINAL = 1.0



# Random Seeds
COLLOCATION_SEEDS = [1000, 1001, 1002]
NETWORK_SEEDS = [0, 1, 2]
