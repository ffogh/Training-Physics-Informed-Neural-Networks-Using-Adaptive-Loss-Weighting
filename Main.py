METHODS = {
    "Regular PINN": train_regular, "SA-PINNs": train_sapinn, "APINNs": train_apinn,
    "I-PINN": train_ipinn, "GW-PINN": train_gwpinn, "VW-PINN": train_vwpinn,
    "PIPNNs-NTK": train_pipnns_ntk, "DB-PINN": train_dbpinn, "wbPINN": train_wbpinn,
}


def run_benchmark():
    results = {"L2RE": {}, "MSE": {}}
    for m in METHODS:
        results["L2RE"][m] = {a[0]: [] for a in ARCHS}
        results["MSE"][m] = {a[0]: [] for a in ARCHS}
    histories = {}
    for arch_name, n_layers, width in ARCHS:
        print(f"\n=== Architecture {arch_name}  ({n_layers} hidden x {width}) ===")
        for rep in range(REPEATS):
            nu, kx, ky, A = sample_instance(seed=1000 + rep)
            x_f, x_ic, x_bc = make_points(seed=1000 + rep)
            u_ic = exact_solution(x_ic, kx, ky, A, nu)
            print(f"  repeat {rep+1}/{REPEATS}: nu={nu:.4f}, kx={kx.tolist()}, ky={ky.tolist()}")
            for name, train_fn in METHODS.items():
                net = build_net(n_layers, width, seed=rep)
                rec = {"epoch": [], "mse": [], "ctx": (kx, ky, A, nu)} if rep == 0 else None
                t0 = time.time(); net = train_fn(net, x_f, x_ic, u_ic, x_bc, nu, rec=rec)
                l2re, mse = evaluate(net, kx, ky, A, nu)
                results["L2RE"][name][arch_name].append(l2re)
                results["MSE"][name][arch_name].append(mse)
                if rec is not None: histories[name] = rec
                print(f"    {name:<13s}  L2RE={l2re:.3e}  MSE={mse:.3e}  ({time.time()-t0:.1f}s)")
    return results, histories


def _fmt(vals):
    a = np.array(vals); return f"{a.mean():.2e}+/-{a.std():.1e}"

def print_table(results, metric):
    arch_names = [a[0] for a in ARCHS]
    print(f"\n########  {metric}  (mean +/- std over {REPEATS} repeats)  ########")
    header = f"{'Method':<14s}" + "".join(f"{a:>22s}" for a in arch_names)
    print(header); print("-" * len(header))
    for m in METHODS:
        print(f"{m:<14s}" + "".join(f"{_fmt(results[metric][m][a]):>22s}" for a in arch_names))
