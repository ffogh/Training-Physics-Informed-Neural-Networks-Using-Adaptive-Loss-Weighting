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
EXACT_SOLUTION = "sinh(pi*y)/sinh(pi) * sin(pi*x)"

# PDE
EQUATION = "u_xx + u_yy = 0"

# Domain
X_DOMAIN = (0.0, 1.0)
Y_DOMAIN = (0.0, 1.0)

# Boundary Conditions
BC_LEFT = 0
BC_RIGHT = 0
BC_BOTTOM = 0
BC_TOP = "sin(pi*x)"


# Random Seeds
COLLOCATION_SEEDS = [1000, 1001, 1002]
NETWORK_SEEDS = [0, 1, 2]


For each seed, the neural network initialization, interior collocation points, and boundary points change **to assess robustness and reduce dependence on random 
initialization and sampling**, while the PDE, exact solution, architecture, training configuration, and evaluation grid remain fixed.


  
