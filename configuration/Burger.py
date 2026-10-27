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
N_BC = 256       # Boundary-condition points (128 per side)

# Evaluation
NX_EVAL = 200
NT_EVAL = 200    # Total evaluation points = 40,000

# Reference Solver
NX_REF = 1024
NT_REF = 500
SOLVER = "BDF"
RTOL = 1e-10
ATOL = 1e-12

# PDE Parameters
NU_VALUES = [0.05, 0.01, 0.005]

# Domain
X_LEFT = -1.0
X_RIGHT = 1.0
T_FINAL = 1.0

# NTK Settings
NTK_FREQ = 1
N_NTK = 64

# APINN Settings
APINN_FREQ = 100
APINN_WIN = 100
