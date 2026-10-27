def residual_pointwise(net, xy, m):
    xy = xy.clone().requires_grad_(True)
    u = net(xy)
    ones = torch.ones_like(u)
    g = torch.autograd.grad(u, xy, grad_outputs=ones, create_graph=True)[0]
    u_x, u_y = g[:, 0:1], g[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xy, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xy, grad_outputs=torch.ones_like(u_y),
                               create_graph=True)[0][:, 1:2]
    x, y = xy[:, 0:1], xy[:, 1:2]
    f_val = 2.0 * (m * np.pi) ** 2 * torch.sin(m * np.pi * x) * torch.sin(m * np.pi * y)
    r = u_xx + u_yy + f_val                        # -Delta u = f  ->  u_xx+u_yy+f = 0
    div_surrogate = u_x + u_y                      # 2-D divergence surrogate
    return r, div_surrogate


def bc_loss(net, x_bc):
    return torch.mean(net(x_bc) ** 2)


def all_losses(net, x_f, x_bc, m):
    r, _ = residual_pointwise(net, x_f, m)
    L_f  = torch.mean(r ** 2)
    L_bc = bc_loss(net, x_bc)
    return L_f, L_bc






