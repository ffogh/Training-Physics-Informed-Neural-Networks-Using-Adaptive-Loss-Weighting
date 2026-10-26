def omega_of(k): return float(np.sqrt((k*np.pi)**2+MASS**2))
def exact_solution(x,t,k): return np.sin(k*np.pi*x)*np.cos(omega_of(k)*t)
def get_reference(k):
    x=np.linspace(*X_DOMAIN,NX);t=np.linspace(*T_DOMAIN,NT)
    gx,gt=np.meshgrid(x,t,indexing="ij");return gx,gt,exact_solution(gx,gt,k)


For the Klein–Gordon equation, we applied a Fourier feature embedding to the network inputs, which feeds the network 
ready-made sine and cosine waves so it can learn oscillating solutions that a plain 
network learns slowly because of spectral bias; the same embedding was used for all methods to keep the comparison fair.
