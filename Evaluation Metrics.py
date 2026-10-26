@torch.no_grad()
def evaluate(net):
    x = np.linspace(*X_DOMAIN, NX); y = np.linspace(*Y_DOMAIN, NY)      # 100 x 100 test grid
    gx, gy = np.meshgrid(x, y, indexing="ij")
    xy = torch.tensor(np.stack([gx.ravel(), gy.ravel()], 1), ...)
    pred = net(xy).squeeze().cpu().numpy()                              # network prediction
    exact = exact_solution(gx, gy).ravel()                              # exact solution
    err = pred - exact                                                  # error at each grid point
    return float(np.sqrt(np.sum(err**2) / np.sum(exact**2))),  float(np.mean(err**2))
    #            └──────────── L2RE ──────────────────────┘    └──── MSE ────┘
