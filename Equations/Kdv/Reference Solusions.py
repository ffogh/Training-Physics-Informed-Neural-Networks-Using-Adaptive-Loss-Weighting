def build_reference():
    from scipy.integrate import solve_ivp
    from scipy.interpolate import RegularGridInterpolator
    N=NX_SPEC; dx=L_DOMAIN/N
    x=np.linspace(X_LEFT,X_RIGHT,N,endpoint=False)
    k=2.*np.pi*np.fft.fftfreq(N,d=dx); u0=A_SOL/np.cosh(K_SOL*(x-X0_SOL))**2
    def rhs(t,u):
        uh=np.fft.fft(u); ux=np.fft.ifft(1j*k*uh).real
        uxxx=np.fft.ifft((1j*k)**3*uh).real; return -ALPHA*u*ux-BETA*uxxx
    t_eval=np.linspace(0.,T_FINAL,NT_REF)
    sol=solve_ivp(rhs,(0.,T_FINAL),u0,t_eval=t_eval,method="DOP853",rtol=1e-9,atol=1e-11)
    assert sol.success
    x_ext=np.concatenate([x,[X_RIGHT]]); U_ext=np.concatenate([sol.y,sol.y[0:1,:]],axis=0)
    interp=RegularGridInterpolator((x_ext,t_eval),U_ext,method="linear",
                                    bounds_error=False,fill_value=None)
    return interp

print("Building KdV reference ...", flush=True)
t0=time.time(); REF_INTERP=build_reference()
print(f"  Done in {time.time()-t0:.1f}s.", flush=True)

xe=np.linspace(X_LEFT,X_RIGHT,NX_EVAL); te=np.linspace(0.,T_FINAL,NT_EVAL)
EVAL_GX,EVAL_GT=np.meshgrid(xe,te,indexing="ij")
EVAL_U=REF_INTERP(np.stack([EVAL_GX.ravel(),EVAL_GT.ravel()],1)).reshape(NX_EVAL,NT_EVAL)
