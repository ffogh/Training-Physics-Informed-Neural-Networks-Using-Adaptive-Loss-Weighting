def train_apinn(net, x_f, x_ic, u_ic, x_bc):
    
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    w = torch.ones(3, device=DEVICE)

    for ep in range(EPOCHS):
        opt.zero_grad()
        terms = torch.stack(all_losses(net, x_f, x_ic, u_ic, x_bc))

        if ep % APINN_FREQ == 0:
            L = terms.detach()
            if L.max() / (L.min() + 1e-12) > 10.0:
                w = 1.0 + APINN_ALPHA * (L - L.min()) / (L.max() - L.min() + 1e-12)

        (w * terms).sum().backward()
        opt.step()
    return net


#"For a fair and computationally consistent comparison, we adapted the methods to a unified experimental framework while accounting for their computational costs.  
#All methods were trained using the Adam optimizer under identical network architectures, training budgets, and experimental conditions.  
#We adopted the default hyperparameters reported in the original papers for all methods, without performing any additional hyperparameter tuning."



#"Algorithm 1 Adaptive loss Weighting Algorithm
#1: Initialize λi = 1 ▷ (i = 1, 2, 3, 4)
#2: Pick α > 0
#3: while iter%f = 0 do
#4: calculate MSEu0 , MSElb, MSEub, MSEf ,
#5: calculate ratio = max(MSEu0 ,MSElb,MSEub,MSEf )
#min(MSEu0 ,MSElb,MSEub,MSEf )
#6: while ratio > 10 do
#7: calculate normi = MSEi−min(MSEu0 ,MSElb,MSEub,MSEf )
#max(MSEu0 ,MSElb,MSEub,MSEf )−min(MSEu0 ,MSElb,MSEub,MSEf )
#8: calculate λi = 1 + αnormi
#9: end while
#10: end while
#11: return λi"
