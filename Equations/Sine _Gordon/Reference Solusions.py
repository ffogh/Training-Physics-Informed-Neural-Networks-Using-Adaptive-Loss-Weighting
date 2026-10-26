x_ext = np.concatenate([x, [L_DOMAIN]])
    U_ext = np.concatenate([U_ref, U_ref[0:1, :]], axis=0)

    interp = RegularGridInterpolator(
        (x_ext, t_eval), U_ext, method="linear",
        bounds_error=False, fill_value=None,
    )
    return interp, x, t_eval, U_ref


print("Building spectral reference (256 modes, DOP853) ...")
t0_ref = time.time()
REF_INTERP, REF_X, REF_T, REF_U = build_reference()
print(f"  Done in {time.time() - t0_ref:.1f}s.  Grid {REF_X.size}x{REF_T.size}, "
      f"u in [{REF_U.min():.4f}, {REF_U.max():.4f}]")
