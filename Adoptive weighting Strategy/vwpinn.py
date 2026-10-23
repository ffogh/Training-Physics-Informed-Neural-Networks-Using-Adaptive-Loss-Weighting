
def train_vwpinn(net, x_f, x_ic, u_ic, x_bc):
    
    pts = x_f.detach().cpu().numpy().T
    rho = torch.tensor(gaussian_kde(pts)(pts), dtype=torch.float32, device=DEVICE).view(-1, 1)
    v2 = (1.0 / (rho + 1e-8)) ** 2
    v2 = v2 / v2.sum()                               # normalised weights

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for _ in range(EPOCHS):
        opt.zero_grad()
        r, _ = residual(net, x_f)
        loss = torch.sum(v2 * r ** 2) + ic_loss(net, x_ic, u_ic) + bc_loss(net, x_bc)
        loss.backward()
        opt.step()
    return net


Our VW-PINN implementation follows the original KDE-based volume-weighting formulation. However, we use Adam instead of the original 
Adam–L-BFGS training procedure to ensure a consistent comparison across methods. Additionally, KDE bandwidth is selected automatically using Scott's rule. 
