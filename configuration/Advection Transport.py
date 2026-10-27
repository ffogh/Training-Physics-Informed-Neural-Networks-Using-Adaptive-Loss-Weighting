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
N_IC = 128       # Initial-condition points
N_BC = 128       # Periodic boundary pairs (256 boundary coordinates)

# Evaluation
NX = 100
NT = 100        # Total evaluation points = 10,000

# Reference Solution
REFERENCE = "Analytical"
EXACT_SOLUTION = "sin(m*pi*(x-C*t))"

# PDE
EQUATION = "u_t + C*u_x = 0"

# PDE Parameters
C = 1.0
M_VALUES = [1, 2, 3]

# Domain
X_DOMAIN = (0.0, 1.0)
T_DOMAIN = (0.0, 1.0)

# Initial Condition
IC = "u(x,0) = sin(m*pi*x)"

# Boundary Condition
BC = "u(0,t) = u(1,t)"


# Random Seeds
NETWORK_SEEDS = [0, 1, 2]

COLLOCATION_SEEDS = {
    1: [1100, 1101, 1102],
    2: [1200, 1201, 1202],
    3: [1300, 1301, 1302]
}


For each seed, the neural network initialization, interior collocation points, initial-condition points, and periodic boundary points 
change to assess robustness and reduce dependence on random initialization and sampling. The PDE coefficient (\(c=1\)), initial and boundary 
conditions, architecture, training configuration, analytical solution, and evaluation grid remain fixed. 
All nine methods use the same training points and network initialization within each repetition, ensuring a controlled comparison.







    
