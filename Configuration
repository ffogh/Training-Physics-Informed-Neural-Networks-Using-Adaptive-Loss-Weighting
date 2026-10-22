DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PDE parameters (four sine modes)
NU = 0.0077                              # diffusion coefficient
KX = [4.0, 4.0, 1.0, 4.0]               # spatial wave numbers in x
KY = [3.0, 1.0, 2.0, 2.0]               # spatial wave numbers in y
JMAX = 4                                # number of sine modes

# Network and training
N_LAYERS, WIDTH = 5, 128
EPOCHS = 10_000
LR = 1e-3
SEEDS = [0, 1, 2]                       # network-initialization seeds

# Training and evaluation points
N_F, N_IC, N_BC = 4096, 1024, 1024
POINT_SEED = 1777
NX_EVAL, NY_EVAL, NT_EVAL = 50, 50, 25

# Method-specific hyperparameters
APINN_FREQ, APINN_ALPHA = 100, 0.5
NTK_FREQ, N_NTK = 100, 64
GW_EPS2 = 1.0

# Convert wave numbers to tensors
kx = torch.tensor(KX, dtype=torch.float32, device=DEVICE)
ky = torch.tensor(KY, dtype=torch.float32, device=DEVICE)





**PDE Instance Variation Across Seeds**

Each random seed generates a different instance of the 2D Heat equation by independently sampling the diffusion coefficient $\nu$, spatial wave numbers $k_x$ and $k_y$, and amplitude coefficients $A$.

For example, one repeat produces:

python
nu = 0.0077
kx = [4.0, 4.0, 1.0, 4.0]
ky = [3.0, 1.0, 2.0, 2.0]


The next seed generates a new set of parameters. Consequently, each repeat has a different initial condition and exact solution, which can lead to different L2RE and MSE values.

Within each repeat, however, all compared methods use the same PDE parameters, training points, and network initialization, ensuring a fair comparison.

Thus, the reported mean and standard deviation reflect variability across different PDE instances and network initializations, rather than network initialization alone.
