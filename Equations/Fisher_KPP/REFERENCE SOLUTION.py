
# ================================================================== #
# 1.  NUMERICAL REFERENCE SOLUTION  (method of lines, Radau)         #
# ================================================================== #
def build_reference():
    """Solve Fisher-KPP on a fine grid and return an interpolator."""
    from scipy.integrate import solve_ivp
    from scipy.interpolate import RegularGridInterpolator

    nx = NX_REF
    dx = L_DOMAIN / (nx - 1)
    x  = np.linspace(0.0, L_DOMAIN, nx)

    # traveling-front initial condition
    u0 = 1.0 / (1.0 + np.exp(LAMBDA_F * (x - X0_FRONT)))
    u0[0] = 1.0                                    # enforce left Dirichlet

    def rhs(t, u):
        dudt = np.zeros(nx)
        # left BC: Dirichlet u(0,t)=1 is frozen
        dudt[0] = 0.0
        # interior: vectorised central differences
        uxx  = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2
        reac = RHO * u[1:-1] * (1.0 - u[1:-1])
        dudt[1:-1] = D_COEFF * uxx + reac
        # right BC: Neumann u_x(L,t)=0 -> ghost u[N]=u[N-2]
        uxx_R = 2.0 * (u[-2] - u[-1]) / dx**2
        dudt[-1] = D_COEFF * uxx_R + RHO * u[-1] * (1.0 - u[-1])
        return dudt

    t_eval = np.linspace(0.0, T_FINAL, NT_REF)
    sol = solve_ivp(rhs, [0.0, T_FINAL], u0, t_eval=t_eval,
                    method="Radau", rtol=1e-8, atol=1e-10)
    assert sol.success, f"Reference solver failed: {sol.message}"

    U_ref  = sol.y                                  # (nx, nt)
    interp = RegularGridInterpolator(
        (x, t_eval), U_ref, method="linear",
        bounds_error=False, fill_value=None,
    )
    return interp, x, t_eval, U_ref


print("Building numerical reference solution (Radau, method of lines) ...")
t0_ref = time.time()
REF_INTERP, REF_X, REF_T, REF_U = build_reference()
print(f"  Done in {time.time() - t0_ref:.1f}s.  Grid {REF_X.size}x{REF_T.size}, "
      f"u in [{REF_U.min():.6f}, {REF_U.max():.6f}]")
