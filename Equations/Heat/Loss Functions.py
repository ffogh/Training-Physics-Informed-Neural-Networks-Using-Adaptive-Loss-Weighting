def residual(net, xyt):
    
    xyt = xyt.clone().requires_grad_(True)
    u = net(xyt)
    du = torch.autograd.grad(u, xyt, torch.ones_like(u), create_graph=True)[0]
    u_x, u_y, u_t = du[:, 0:1], du[:, 1:2], du[:, 2:3]
    u_xx = torch.autograd.grad(u_x, xyt, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xyt, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    return u_t - NU * (u_xx + u_yy), u_x + u_y


def ic_loss(net, x_ic, u_ic):
    return torch.mean((net(x_ic) - u_ic) ** 2)


def bc_loss(net, x_bc):
    return torch.mean(net(x_bc) ** 2)


def all_losses(net, x_f, x_ic, u_ic, x_bc):
    r, _ = residual(net, x_f)
    return torch.mean(r ** 2), ic_loss(net, x_ic, u_ic), bc_loss(net, x_bc)
