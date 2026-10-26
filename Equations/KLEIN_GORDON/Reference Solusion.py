def exact_solution(x,t,k): return np.sin(k*np.pi*x)*np.cos(omega_of(k)*t)
def get_reference(k):
    x=np.linspace(*X_DOMAIN,NX);t=np.linspace(*T_DOMAIN,NT)
    gx,gt=np.meshgrid(x,t,indexing="ij");return gx,gt,exact_solution(gx,gt,k)
