# Neural Network
ARCHITECTURES = ["4x128"]
ACTIVATION = "Tanh"

# Training
EPOCHS = 10000
LR = 1e-3
OPTIMIZER = "Adam"
LR_SCHEDULER = "CosineAnnealingLR"
LR_MIN = 1e-5
REPEATS = 3

# Training Points
N_F = 4096       # Total PDE collocation points
BATCH_F = 1024   # PDE points sampled per epoch
N_IC = 256       # Initial-condition points
N_BC = 256       # Boundary points (128 per edge)

# Evaluation
NX = 200
NT = 200        # Total evaluation points = 40,000

# Reference Solution
REFERENCE = "Analytical"
EXACT_SOLUTION = "sin(k*pi*x) * cos(omega*t)"
OMEGA = "sqrt((k*pi)**2 + MASS**2)"

# PDE
EQUATION = "u_tt - u_xx + MASS**2*u = 0"

# PDE Parameters
MASS = 0.5
K_VALUES = [1, 2, 3]

# Domain
X_DOMAIN = (0.0, 1.0)
T_DOMAIN = (0.0, 1.0)

# Initial Conditions
IC_DISPLACEMENT = "u(x,0) = sin(k*pi*x)"
IC_VELOCITY = "u_t(x,0) = 0"

# Boundary Conditions
BC_LEFT = "u(0,t) = 0"
BC_RIGHT = "u(1,t) = 0"



# Random Seeds
NETWORK_SEEDS = [0, 1, 2]

COLLOCATION_SEEDS = {
    1: [1100, 1101, 1102],
    2: [1200, 1201, 1202],
    3: [1300, 1301, 1302]
}



For each seed, the neural network initialization, interior collocation points, initial-condition points, and boundary points change 
to assess robustness and reduce dependence on random initialization and sampling. The PDE mass parameter, wave number (k), architecture, 
Fourier features, training configuration, analytical solution, and evaluation grid remain fixed. All nine methods use the same initial collocation-point pool and 
network initialization within each repetition, although interior mini-batches are independently sampled during training.



