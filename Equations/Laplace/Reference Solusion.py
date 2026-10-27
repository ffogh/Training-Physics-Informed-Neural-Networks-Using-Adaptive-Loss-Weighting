def exact_solution(x, y):
    return np.sinh(np.pi * y) / np.sinh(np.pi) * np.sin(np.pi * x)


def get_reference():
    """Return (x_grid, y_grid, U_exact) on an NX x NY grid."""
    x = np.linspace(X_DOMAIN[0], X_DOMAIN[1], NX)
    y = np.linspace(Y_DOMAIN[0], Y_DOMAIN[1], NY)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    U = exact_solution(gx, gy)
    return gx, gy, U
