
def _flat_grad(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                      for g, p in zip(grads, params)])







def train_dbpinn(net, x_f, x_ic, u_ic, x_bc):
    
    params = list(net.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    w = torch.ones(2, device=DEVICE)
    mu = None

    for ep in range(EPOCHS):
        L_f, L_ic, L_bc = all_losses(net, x_f, x_ic, u_ic, x_bc)
        L_cond = torch.stack([L_ic, L_bc])

        # Inter-balancing: residual gradient scale vs. condition gradient scale
        g_max = _flat_grad(L_f, params).abs().max()
        G = sum(g_max / (_flat_grad(w[i] * L_i, params).abs().mean() + 1e-12)
                for i, L_i in enumerate((L_ic, L_bc)))

        # Intra-balancing: current loss relative to its running mean
        t = ep + 1
        mu = L_cond.detach() if mu is None else (1 - 1 / t) * mu + (1 / t) * L_cond.detach()
        I = L_cond.detach() / (mu + 1e-12)

        w = ((1 - 1 / t) * w + (1 / t) * (I / (I.sum() + 1e-12)) * G).detach()

        opt.zero_grad()
        (L_f + w[0] * L_ic + w[1] * L_bc).backward()
        opt.step()
    return net

#We found the followong on the paper 

#we formulate the calculation procedure of the aggregated
#weight by adopting the mean magnitude of gradients

#Page 3, right column, Section 3.1 “Inter-Balancing Between PDE Residual Loss and Condition-Fitting Losses”. At the end of the paragraph, just before Eq. (4):

#“Specifically, we formulate the calculation procedure of the aggregated weight by adopting the mean magnitude of gradients (i.e., mean [Wang et al., 2021]) as a statistic:”

#followed by Eq. (4):

#G = Σᵢ max{|∇θLᵣ|} / |∇θλᵢLᵢ| (with a bar over the denominator)


#For a fair and computationally consistent comparison, we adapted the methods to a unified experimental framework while accounting for their computational costs.  
#All methods were trained using the Adam optimizer under identical network architectures, training budgets, and experimental conditions.  
#We adopted the default hyperparameters reported in the original papers for all methods, without performing any additional hyperparameter tuning.
