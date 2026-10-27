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
NX = 200
NT = 200        # Total evaluation points = 40,000

# Reference Solution
REFERENCE = "Analytical"
EXACT_SOLUTION = "exp((R-D*(m*pi)**2)*t)*sin(m*pi*x)"

# PDE
EQUATION = "u_t = D*u_xx + R*u"

# PDE Parameters
D_COEFF = 0.01
R_COEFF = 0.05
M_VALUES = [1, 2, 3]

# Domain
X_DOMAIN = (0.0, 1.0)
T_DOMAIN = (0.0, 1.0)

# Initial Condition
IC = "u(x,0) = sin(m*pi*x)"

# Boundary Conditions
BC_LEFT = "u(0,t) = 0"
BC_RIGHT = "u(1,t) = 0"

# NTK Settings
NTK_FREQ = 1
N_NTK = 64

# APINN Settings
APINN_FREQ = 100
APINN_WIN = 100

# Random Seeds
NETWORK_SEEDS = [0, 1, 2]

COLLOCATION_SEEDS = {
    1: [1100, 1101, 1102],
    2: [1200, 1201, 1202],
    3: [1300, 1301, 1302]
}
