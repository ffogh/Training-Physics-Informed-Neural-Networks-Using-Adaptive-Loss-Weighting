
# Exact Solution
def exact_solution(xyt):
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
    decay = torch.exp(
        -NU * (KX**2 + KY**2) * np.pi**2 * t
    )
    return (
        AMP * decay
        * torch.sin(KX * np.pi * x)
        * torch.sin(KY * np.pi * y)
    )


Why do you need this function in your PINN?
Your PINN predicts \(u_\theta(x,y,t)\), while exact_solution() gives the known mathematical solution \(u(x,y,t)\).

You compare them to calculate L2RE and MSE:
\[
\mathrm{L2RE}=\frac{\|u_\theta-u\|_2}{\|u\|_2}.
\]
