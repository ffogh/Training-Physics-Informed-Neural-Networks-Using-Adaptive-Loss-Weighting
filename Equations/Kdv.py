def derivatives_u(net,xt,need_third=False):
    xt=xt.clone().requires_grad_(True);u=net(xt);ones=torch.ones_like(u)
    g1=torch.autograd.grad(u,xt,grad_outputs=ones,create_graph=True)[0]
    u_x,u_t=g1[:,0:1],g1[:,1:2]
    g2=torch.autograd.grad(u_x,xt,grad_outputs=torch.ones_like(u_x),create_graph=True)[0]
    u_xx=g2[:,0:1]
    if not need_third: return u,u_x,u_t,u_xx
    g3=torch.autograd.grad(u_xx,xt,grad_outputs=torch.ones_like(u_xx),create_graph=True)[0]
    return u,u_x,u_t,u_xx,g3[:,0:1]

def residual_pointwise(net,xt):
    u,u_x,u_t,_,u_xxx=derivatives_u(net,xt,need_third=True)
    return u_t+ALPHA*u*u_x+BETA*u_xxx

def ic_loss(net,x_ic,u_ic): return torch.mean((net(x_ic)-u_ic)**2)

def bc_loss(net,x_bc_l,x_bc_r):
    u_l,ux_l,_,uxx_l=derivatives_u(net,x_bc_l)
    u_r,ux_r,_,uxx_r=derivatives_u(net,x_bc_r)
    return torch.mean((u_l-u_r)**2)+torch.mean((ux_l-ux_r)**2)+torch.mean((uxx_l-uxx_r)**2)

def all_losses(net,xf,xic,uic,xbl,xbr):
    r=residual_pointwise(net,xf)
    return torch.mean(r**2),ic_loss(net,xic,uic),bc_loss(net,xbl,xbr)

