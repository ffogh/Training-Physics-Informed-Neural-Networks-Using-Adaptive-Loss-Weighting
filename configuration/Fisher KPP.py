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
N_IC = 256       # Initial-condition points
N_BC = 128       # Boundary points per edge (256 total)

# Evaluation
NX_EVAL = 200
NT_EVAL = 200    # Total evaluation points = 40,000

# Reference Solution
REFERENCE = "Numerical"
SOLVER = "Radau"
NX_REF = 2000
NT_REF = 500
RTOL = 1e-8
ATOL = 1e-10

# PDE
EQUATION = "u_t = D*u_xx + rho*u*(1-u)"

# PDE Parameters
D_COEFF = 1e-3
RHO = 1.0

# Domain
X_DOMAIN = (0.0, 1.0)
T_DOMAIN = (0.0, 5.0)

# Initial Condition
X0_FRONT = 0.25
LAMBDA_F = sqrt(RHO / D_COEFF)
INITIAL_CONDITION = "1 / (1 + exp(lambda*(x-x0)))"

# Boundary Conditions
BC_LEFT = "u(0,t) = 1"
BC_RIGHT = "u_x(1,t) = 0"


# Random Seeds
COLLOCATION_SEEDS = [1000, 1001, 1002]
NETWORK_SEEDS = [0, 1, 2]

For each seed, the neural network initialization, interior collocation points, initial-condition points, and
boundary points change to assess robustness and reduce dependence on random initialization and sampling. 
The PDE parameters, initial and boundary conditions, architecture, training configuration, numerical reference solution, and evaluation grid remain fixed.

