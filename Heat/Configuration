
# PDE parameters (sum of four sine modes)
NU = 0.007741                                   # diffusion coefficient
KX = [4.0, 4.0, 1.0, 4.0]                       # x wave numbers
KY = [3.0, 1.0, 2.0, 2.0]                       # y wave numbers
AMP = [0.1943, -0.2857, -0.0386, -0.1567]       # mode amplitudes

# Network and training
N_LAYERS, WIDTH = 5, 128
EPOCHS = 10_000
LR = 1e-3
SEEDS = [0, 1, 2]                # network-initialisation seeds

# Training and evaluation points
N_F, N_IC, N_BC = 4096, 1024, 1024
POINT_SEED = 1777
NX_EVAL, NY_EVAL, NT_EVAL = 50, 50, 25

# Method-specific hyperparameters
APINN_FREQ, APINN_ALPHA = 100, 0.5       # APINNs: update interval, scale
NTK_FREQ, N_NTK = 100, 64                # NTK: update interval, sampled points
GW_EPS2 = 1.0                            # GW-PINN: gradient sensitivity



L2RE and MSE give identical method rankings when evaluated
# on the same PDE instance and evaluation grid.
#
# However, their rankings may differ after averaging results
# across multiple seeds due to differences in run-to-run
# variability.
#
# When PDE parameters change across seeds, the reference
# solution may also change, introducing an additional source
# of ranking differences.
#
# Consequently, different aggregated L2RE and MSE rankings
# do not necessarily indicate numerical inconsistency.
#
# Mean L2RE is used as the primary comparison metric,
# while MSE is reported as a complementary accuracy metric.
