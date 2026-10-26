def residual_pointwise(net, xt, return_ux=False):
    """Pointwise Sine-Gordon residual  r = u_tt - u_xx + sin(u).

    Autograd path:
       1 call  ->  (u_x, u_t)          [first derivatives, shared]
       1 call  ->  u_xx from u_x       [second spatial]
       1 call  ->  u_tt from u_t       [second temporal]
    Total: 3 autograd calls  (KdV needs 4 for u_xxx)."""
    xt = xt.clone().requires_grad_(True)
    u  = net(xt)
    ones = torch.ones_like(u)

    # first derivatives  (single call, both components)
    g1  = torch.autograd.grad(u, xt, grad_outputs=ones, create_graph=True)[0]
    u_x = g1[:, 0:1]
    u_t = g1[:, 1:2]

    # second spatial derivative  u_xx
    u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:, 0:1]

    # second temporal derivative  u_tt
    u_tt = torch.autograd.grad(u_t, xt, grad_outputs=torch.ones_like(u_t),
                               create_graph=True)[0][:, 1:2]

    r = u_tt - u_xx + torch.sin(u)

    if return_ux:
        return r, u_x
    return r


def ic_loss(net, x_ic, u_ic_target):
    """MSE combining  u(x,0) = u0(x)  AND  u_t(x,0) = 0."""
    xt = x_ic.clone().requires_grad_(True)
    u  = net(xt)

    # position IC
    L_u = torch.mean((u - u_ic_target) ** 2)

    # velocity IC:  u_t(x,0) = 0
    g   = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_t = g[:, 1:2]
    L_v = torch.mean(u_t ** 2)

    return L_u + L_v


def bc_loss(net, x_bc_left, x_bc_right):
    """Periodic:  u(0,t) = u(L,t)  AND  u_x(0,t) = u_x(L,t)."""
    xl = x_bc_left.clone().requires_grad_(True)
    xr = x_bc_right.clone().requires_grad_(True)
    u_l = net(xl)
    u_r = net(xr)

    L_u  = torch.mean((u_l - u_r) ** 2)

    gl = torch.autograd.grad(u_l, xl, grad_outputs=torch.ones_like(u_l),
                             create_graph=True)[0]
    gr = torch.autograd.grad(u_r, xr, grad_outputs=torch.ones_like(u_r),
                             create_graph=True)[0]
    L_ux = torch.mean((gl[:, 0:1] - gr[:, 0:1]) ** 2)

    return L_u + L_ux


def all_losses(net, x_f, x_ic, u_ic_tgt, x_bc_l, x_bc_r):
    """Return three scalars:  L_f, L_ic, L_bc."""
    r    = residual_pointwise(net, x_f)
    L_f  = torch.mean(r ** 2)
    L_ic = ic_loss(net, x_ic, u_ic_tgt)
    L_bc = bc_loss(net, x_bc_l, x_bc_r)
    return L_f, L_ic, L_bc
