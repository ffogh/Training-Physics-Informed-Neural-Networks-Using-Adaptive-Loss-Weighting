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

# PDE Parameters
ALPHA = 1.0
BETA = 0.0025
A_SOL = 1.0
X0_SOL = 0.0

# Domain
X_LEFT = -1.0
X_RIGHT = 1.0
T_FINAL = 1.0

# NTK Settings
NTK_FREQ = 1
N_NTK = 64
