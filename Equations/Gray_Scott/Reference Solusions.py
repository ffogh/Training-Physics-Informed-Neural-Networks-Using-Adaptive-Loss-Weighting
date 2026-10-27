def build_reference():
    """Solve 1-D Gray-Scott with periodic BC via pseudo-spectral + BDF."""
    from scipy.integrate import solve_ivp
    from scipy.interpolate import RegularGridInterpolator

    N  = NX_SPEC
    dx = L_DOMAIN / N
    x  = np.linspace(0.0, L_DOMAIN, N, endpoint=False)
    k  = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    k2 = k ** 2

    # IC
    gauss = np.exp(-200.0 * (x - 0.5) ** 2)
    u0 = 1.0 - 0.5 * gauss
    v0 = 0.25 * gauss
    y0 = np.concatenate([u0, v0])

    def rhs(t, y):
        u, v = y[:N], y[N:]
        u_xx = np.fft.ifft(-k2 * np.fft.fft(u)).real
        v_xx = np.fft.ifft(-k2 * np.fft.fft(v)).real
        uv2  = u * v ** 2
        du   = DU * u_xx - uv2 + F_RATE * (1.0 - u)
        dv   = DV * v_xx + uv2 - (F_RATE + K_RATE) * v
        return np.concatenate([du, dv])

    t_eval = np.linspace(0.0, T_FINAL, NT_REF)
    sol = solve_ivp(rhs, [0.0, T_FINAL], y0, t_eval=t_eval,
                    method="BDF", rtol=1e-8, atol=1e-10)
    assert sol.success, f"Reference solver failed: {sol.message}"

    U_ref = sol.y[:N, :]
    V_ref = sol.y[N:, :]

    # extend for periodic interpolation at x = L
    x_ext = np.concatenate([x, [L_DOMAIN]])
    U_ext = np.concatenate([U_ref, U_ref[0:1, :]], axis=0)
    V_ext = np.concatenate([V_ref, V_ref[0:1, :]], axis=0)

    interp_u = RegularGridInterpolator(
        (x_ext, t_eval), U_ext, method="linear",
        bounds_error=False, fill_value=None)
    interp_v = RegularGridInterpolator(
        (x_ext, t_eval), V_ext, method="linear",
        bounds_error=False, fill_value=None)
    return interp_u, interp_v, x, t_eval, U_ref, V_ref


print("Building Gray-Scott reference (128 modes, BDF) ...")
t0_ref = time.time()
REF_U_INTERP, REF_V_INTERP, REF_X, REF_T, REF_U, REF_V = build_reference()
print(f"  Done in {time.time() - t0_ref:.1f}s.  "
      f"u in [{REF_U.min():.4f}, {REF_U.max():.4f}],  "
      f"v in [{REF_V.min():.4f}, {REF_V.max():.4f}]")



