def exact_np(x, t):
    """Exact u(x,t) = sin(pi x) cos(c pi t)  (numpy)."""
    return np.sin(np.pi * x) * np.cos(C * np.pi * t)
