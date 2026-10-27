# Neural Network
ARCHITECTURES = ["4x128"]
ACTIVATION = "Tanh"

# Training
EPOCHS = 10000
LR = 1e-3
OPTIMIZER = "Adam"
REPEATS = 3

# Training Points
N_F = 4096       # Interior collocation points
N_IC = 256       # Initial-condition points
N_BC = 256       # Boundary points (128 per edge)

# Evaluation
NX = 256
NT = 100        # Total evaluation points = 25,600

# Reference Solution
REFERENCE = "Analytical"
EXACT_SOLUTION = "sin(pi*x) * cos(C*pi*t)"

# PDE
EQUATION = "u_tt = C**2 * u_xx"

# PDE Parameters
C = 2.0         # Wave propagation speed

# Domain
X_DOMAIN = (0.0, 1.0)
T_DOMAIN = (0.0, 1.0)

# Initial Conditions
IC_DISPLACEMENT = "u(x,0) = sin(pi*x)"
IC_VELOCITY = "u_t(x,0) = 0"

# Boundary Conditions
BC_LEFT = "u(0,t) = 0"
BC_RIGHT = "u(1,t) = 0"


# Random Seeds
NETWORK_SEEDS = [0, 1, 2]
COLLOCATION_SEEDS = [1000, 1001, 1002]

For each seed, the neural network initialization, interior collocation points, initial-condition points, and boundary points change 
to assess robustness and reduce dependence on random initialization and sampling. The PDE parameters, wave speed, initial and boundary 
conditions, architecture, training configuration, analytical solution, and evaluation grid remain fixed. 
All nine methods use the same training points and network initialization within each repetition, ensuring a controlled comparison.
