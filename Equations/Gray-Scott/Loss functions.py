def residual_pointwise(net, xt, return_ux=False):
    """Pointwise Gray-Scott residuals for both species.
    Returns (r_u, r_v) each of shape (N,1).
    4 autograd calls: first + second derivatives for u and v separately."""
    xt  = xt.clone().requires_grad_(True)
    out = net(xt)                                   # (N, 2)
    u, v = out[:, 0:1], out[:, 1:2]
    ones = torch.ones_like(u)

    # -- u derivatives (2 calls) --
    gu   = torch.autograd.grad(u, xt, grad_outputs=ones, create_graph=True)[0]
    u_x, u_t = gu[:, 0:1], gu[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt, grad_outputs=ones,
                               create_graph=True)[0][:, 0:1]

    # -- v derivatives (2 calls) --
    gv   = torch.autograd.grad(v, xt, grad_outputs=ones, create_graph=True)[0]
    v_x, v_t = gv[:, 0:1], gv[:, 1:2]
    v_xx = torch.autograd.grad(v_x, xt, grad_outputs=ones,
                               create_graph=True)[0][:, 0:1]

    uv2 = u * v ** 2
    r_u = u_t - DU * u_xx + uv2 - F_RATE * (1.0 - u)
    r_v = v_t - DV * v_xx - uv2 + (F_RATE + K_RATE) * v

    if return_ux:
        return r_u, r_v, u_x
    return r_u, r_v


def ic_loss(net, x_ic, u_ic_tgt, v_ic_tgt):
    """Combined IC loss for both species (pure forward pass)."""
    out = net(x_ic)
    return (torch.mean((out[:, 0:1] - u_ic_tgt) ** 2)
          + torch.mean((out[:, 1:2] - v_ic_tgt) ** 2))




def bc_loss(net, x_bc_l, x_bc_r):
    """Periodic BC: match u, v and their x-derivatives at both ends."""
    xl = x_bc_l.clone().requires_grad_(True)
    xr = x_bc_r.clone().requires_grad_(True)
    ol, orr = net(xl), net(xr)
    L_val = torch.mean((ol - orr) ** 2)
    L_der = 0.0
    for k in range(2):  # k=0: u, k=1: v
        gl = torch.autograd.grad(ol[:, k:k+1], xl, torch.ones_like(ol[:, k:k+1]),
                                 create_graph=True)[0][:, 0:1]
        gr = torch.autograd.grad(orr[:, k:k+1], xr, torch.ones_like(orr[:, k:k+1]),
                                 create_graph=True)[0][:, 0:1]
        L_der = L_der + torch.mean((gl - gr) ** 2)
    return L_val + L_der



def all_losses(net, x_f, x_ic, u_ic, v_ic, x_bc_l, x_bc_r):
    """Return three scalars: L_f (combined residual), L_ic, L_bc."""
    r_u, r_v = residual_pointwise(net, x_f)
    L_f  = torch.mean(r_u ** 2) + torch.mean(r_v ** 2)
    L_ic = ic_loss(net, x_ic, u_ic, v_ic)
    L_bc = bc_loss(net, x_bc_l, x_bc_r)
    return L_f, L_ic, L_bc
