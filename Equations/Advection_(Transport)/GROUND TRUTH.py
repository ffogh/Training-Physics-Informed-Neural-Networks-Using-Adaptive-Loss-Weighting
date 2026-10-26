
def exact_solution(x, t, m):
    return np.sin(m * np.pi * (x - C * t))


def get_reference(m):
    x = np.linspace(X_DOMAIN[0], X_DOMAIN[1], NX)
    t = np.linspace(T_DOMAIN[0], T_DOMAIN[1], NT)
    gx, gt = np.meshgrid(x, t, indexing="ij")
    U = exact_solution(gx, gt, m)
    return gx, gt, U
