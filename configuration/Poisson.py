# Neural Network
ARCHITECTURES = ["4x128", "5x128", "6x128"]
ACTIVATION = "Tanh"

# Training
EPOCHS = 10000
LR = 1e-3
OPTIMIZER = "Adam"
REPEATS = 3

# Training Points
N_F = 4096       # PDE collocation points
N_BC = 256       # Boundary points (64 per edge)
N_IC = 0         # No initial condition (steady PDE)

# Evaluation
NX = 100
NY = 100        # Total evaluation points = 10,000

# Reference Solution
REFERENCE = "Analytical"
EXACT_SOLUTION = "sin(m*pi*x) * sin(m*pi*y)"

# PDE Parameters
M_VALUES = [1, 2, 3]

# Domain
X_DOMAIN = (0.0, 1.0)
Y_DOMAIN = (0.0, 1.0)




# Random Seeds
COLLOCATION_SEEDS = {
    1: [1100, 1101, 1102],
    2: [1200, 1201, 1202],
    3: [1300, 1301, 1302]
}
NETWORK_SEEDS = [0, 1, 2]
