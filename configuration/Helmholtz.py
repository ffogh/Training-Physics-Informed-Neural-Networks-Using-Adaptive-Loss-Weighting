# Neural Network
ARCHITECTURES = ["4x128", "5x128", "6x128"]
ACTIVATION = "Tanh"

# Training
EPOCHS = 2000
LR = 1e-3
OPTIMIZER = "Adam"
REPEATS = 3

# Training Points
N_F = 8192       # PDE collocation points
N_BC = 256       # Boundary points (64 per edge)
N_IC = 0         # No initial condition (steady PDE)

# Evaluation
NX = 200
NY = 200        # Total evaluation points = 40,000

# Reference Solution
REFERENCE = "Analytical"
EXACT_SOLUTION = "sin(k*pi*x) * sin(k*pi*y)"

# PDE Parameters
K_VALUES = [8, 16, 32]

# PDE
EQUATION = "-(u_xx + u_yy) - k**2*u = f(x,y)"

# Domain
X_DOMAIN = (0.0, 1.0)
Y_DOMAIN = (0.0, 1.0)

# Boundary Conditions
BC = "Homogeneous Dirichlet (u=0 on all edges)"

# Fourier Features
USE_FOURIER = True
N_FOURIER = 24
K_MAX_FF = 32.0
FOURIER_INPUT_DIM = 98


# Random Seeds
NETWORK_SEEDS = [0, 1, 2]

COLLOCATION_SEEDS = {
    8:  [1800, 1801, 1802],
    16: [2600, 2601, 2602],
    32: [4200, 4201, 4202]
}

For each seed, the neural network initialization, interior collocation points, and boundary points change to assess robustness and 
reduce dependence on random initialization and sampling. 
The PDE, wave number \(k\), exact solution, architecture, Fourier features, training configuration, and evaluation grid remain fixed.


