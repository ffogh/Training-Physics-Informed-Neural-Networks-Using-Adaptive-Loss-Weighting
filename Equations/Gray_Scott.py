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
    """Periodic BC: match both u and v at boundaries (forward pass only)."""
    return torch.mean((net(x_bc_l) - net(x_bc_r)) ** 2)


def all_losses(net, x_f, x_ic, u_ic, v_ic, x_bc_l, x_bc_r):
    """Return three scalars: L_f (combined residual), L_ic, L_bc."""
    r_u, r_v = residual_pointwise(net, x_f)
    L_f  = torch.mean(r_u ** 2) + torch.mean(r_v ** 2)
    L_ic = ic_loss(net, x_ic, u_ic, v_ic)
    L_bc = bc_loss(net, x_bc_l, x_bc_r)
    return L_f, L_ic, L_bc
