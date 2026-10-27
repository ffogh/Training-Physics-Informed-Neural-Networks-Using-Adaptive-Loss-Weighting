_REF_CACHE = {}

def _get_reference(nu):
    if nu in _REF_CACHE:
        return _REF_CACHE[nu]
    from scipy.integrate import solve_ivp
    from scipy.interpolate import RegularGridInterpolator
    print(f"  Building Burgers reference (nu={nu}) ...", end="", flush=True)
    t0=time.time()
    N=NX_REF; dx=L_DOMAIN/(N+1)
    x_int=np.linspace(X_LEFT+dx, X_RIGHT-dx, N)
    u0=-np.sin(np.pi*x_int)
    def rhs(t,u):
        ue=np.concatenate([[0.],u,[0.]])
        ux=(ue[2:]-ue[:-2])/(2*dx)
        uxx=(ue[2:]-2*ue[1:-1]+ue[:-2])/dx**2
        return -u*ux + nu*uxx
    t_eval=np.linspace(0.,T_FINAL,NT_REF)
    sol=solve_ivp(rhs,(0.,T_FINAL),u0,t_eval=t_eval,method="BDF",rtol=1e-10,atol=1e-12)
    assert sol.success, f"Solver failed: {sol.message}"
    # Build interpolator including boundary points
    x_full=np.concatenate([[X_LEFT],x_int,[X_RIGHT]])
    U_full=np.concatenate([np.zeros((1,NT_REF)),sol.y,np.zeros((1,NT_REF))],axis=0)
    interp=RegularGridInterpolator((x_full,t_eval),U_full,method="linear",
                                    bounds_error=False,fill_value=None)
    # Eval grid
    xe=np.linspace(X_LEFT,X_RIGHT,NX_EVAL); te=np.linspace(0.,T_FINAL,NT_EVAL)
    gx,gt=np.meshgrid(xe,te,indexing="ij")
    pts=np.stack([gx.ravel(),gt.ravel()],1)
    U_eval=interp(pts).reshape(NX_EVAL,NT_EVAL)
    _REF_CACHE[nu]=(gx,gt,U_eval)
    print(f" done ({time.time()-t0:.1f}s)", flush=True)
    return gx,gt,U_eval
