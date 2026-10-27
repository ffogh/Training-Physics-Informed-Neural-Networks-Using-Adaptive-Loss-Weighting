def exact_solution(xyt):
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
    decay = torch.exp(-NU * (KX ** 2 + KY ** 2) * np.pi ** 2 * t)
    modes = A * decay * torch.sin(KX * np.pi * x) * torch.sin(KY * np.pi * y)
    return modes.sum(dim=1, keepdim=True)
