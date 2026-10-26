
def residual_pointwise(net,xt,mass):
    xt=xt.clone().requires_grad_(True);u=net(xt);ones=torch.ones_like(u)
    g=torch.autograd.grad(u,xt,grad_outputs=ones,create_graph=True)[0]
    u_x,u_t=g[:,0:1],g[:,1:2]
    u_xx=torch.autograd.grad(u_x,xt,grad_outputs=torch.ones_like(u_x),create_graph=True)[0][:,0:1]
    u_tt=torch.autograd.grad(u_t,xt,grad_outputs=torch.ones_like(u_t),create_graph=True)[0][:,1:2]
    return u_tt-u_xx+(mass**2)*u, u_x

def ic_terms(net,x_ic,u_ic0):
    x_ic=x_ic.clone().requires_grad_(True);u=net(x_ic)
    u_t=torch.autograd.grad(u,x_ic,grad_outputs=torch.ones_like(u),create_graph=True)[0][:,1:2]
    return torch.mean((u-u_ic0)**2),torch.mean(u_t**2)

def bc_loss(net,x_bc): return torch.mean(net(x_bc)**2)

def all_losses(net,x_f,x_ic,u_ic0,x_bc,mass):
    r,_=residual_pointwise(net,x_f,mass);Lr=torch.mean(r**2)
    Lic0,Lic1=ic_terms(net,x_ic,u_ic0);Lbc=bc_loss(net,x_bc)
    return Lr,Lic0,Lic1,Lbc
