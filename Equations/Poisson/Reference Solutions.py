def exact_solution(x, y, m):
    return np.sin(m * np.pi * x) * np.sin(m * np.pi * y)


def get_reference(m):
    """Return (x_grid, y_grid, U_exact) on an NX x NY grid for frequency m."""
    x = np.linspace(X_DOMAIN[0], X_DOMAIN[1], NX)
    y = np.linspace(Y_DOMAIN[0], Y_DOMAIN[1], NY)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    U = exact_solution(gx, gy, m)
    return gx, gy, U
