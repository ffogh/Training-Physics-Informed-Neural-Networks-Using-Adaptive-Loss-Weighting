def residual_pointwise(net,xt,nu):
    """r = u_t + u*u_x - nu*u_xx"""
    xt=xt.clone().requires_grad_(True); u=net(xt); ones=torch.ones_like(u)
    g1=torch.autograd.grad(u,xt,grad_outputs=ones,create_graph=True)[0]
    u_x,u_t=g1[:,0:1],g1[:,1:2]
    u_xx=torch.autograd.grad(u_x,xt,grad_outputs=torch.ones_like(u_x),create_graph=True)[0][:,0:1]
    r=u_t + u*u_x - nu*u_xx
    return r,u_x

def ic_loss(net,x_ic,u_ic): return torch.mean((net(x_ic)-u_ic)**2)
def bc_loss(net,x_bc): return torch.mean(net(x_bc)**2)

def all_losses(net,x_f,x_ic,u_ic,x_bc,nu):
    r,_=residual_pointwise(net,x_f,nu)
    return torch.mean(r**2),ic_loss(net,x_ic,u_ic),bc_loss(net,x_bc)
