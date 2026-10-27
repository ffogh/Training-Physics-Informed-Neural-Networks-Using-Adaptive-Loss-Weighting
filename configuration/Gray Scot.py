# Neural Network
ARCHITECTURES = ["4x128"]
ACTIVATION = "Tanh"
INPUT_DIM = 2
OUTPUT_DIM = 2  # u and v

# Training
EPOCHS = 10000
LR = 1e-3
OPTIMIZER = "Adam"
REPEATS = 3

# Training Points
N_F = 4096       # Interior collocation points
N_IC = 256       # Initial-condition points
N_BC = 128       # Periodic boundary pairs

# Evaluation
NX_EVAL = 200
NT_EVAL = 200    # Total evaluation points = 40,000

# Reference Solution
REFERENCE = "Numerical"
SOLVER = "BDF"
SPATIAL_METHOD = "Pseudo-spectral"
NX_SPEC = 128
NT_REF = 500
RTOL = 1e-8
ATOL = 1e-10

# PDE
EQUATION_U = "u_t = Du*u_xx - u*v**2 + F*(1-u)"
EQUATION_V = "v_t = Dv*v_xx + u*v**2 - (F+k)*v"

# PDE Parameters
DU = 1e-2
DV = 5e-3
F_RATE = 0.04
K_RATE = 0.06

# Domain
X_DOMAIN = (0.0, 1.0)
T_DOMAIN = (0.0, 5.0)

# Initial Conditions
IC_U = "1 - 0.5*exp(-200*(x-0.5)**2)"
IC_V = "0.25*exp(-200*(x-0.5)**2)"

# Boundary Conditions
BC_U = "u(0,t) = u(1,t)"
BC_V = "v(0,t) = v(1,t)"


# Random Seeds
NETWORK_SEEDS = [0, 1, 2]
COLLOCATION_SEEDS = [1000, 1001, 1002]


For each seed, the neural network initialization, interior collocation points, initial-condition points, and periodic boundary points 
change to assess robustness and reduce dependence on random initialization and sampling. The PDE coefficients,
initial and boundary conditions, architecture, training configuration, numerical reference solution, and evaluation grid remain fixed. 
All nine methods use the same training points and network initialization within each repetition, ensuring a controlled comparison.
