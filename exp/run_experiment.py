import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.model import PARAMS, GAMMAS
from src.simulation import run_all_gammas, compute_stats

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Run simulations for all gamma values                             #
    # ------------------------------------------------------------------ #
    all_results = run_all_gammas(GAMMAS, PARAMS, seed=42)

    # ------------------------------------------------------------------ #
    # 2. Compute per-gamma statistics                                     #
    # ------------------------------------------------------------------ #
    stats_by_gamma = {g: compute_stats(all_results[g]) for g in GAMMAS}

    # ------------------------------------------------------------------ #
    # 3. Print summary table                                              #
    # ------------------------------------------------------------------ #
    header = "Gamma | Inv Mean | Inv Std  | Sym Mean | Sym Std  | Inv Q Std | Sym Q Std"
    print(header)
    print('-' * 70)
    for g in GAMMAS:
        s = stats_by_gamma[g]
        print(
            f"{g:5.3f} | {s['inv_mean']:8.4f} | {s['inv_std']:8.4f} | "
            f"{s['sym_mean']:8.4f} | {s['sym_std']:8.4f} | "
            f"{s['inv_q_std']:9.4f} | {s['sym_q_std']:9.4f}"
        )

    # ------------------------------------------------------------------ #
    # 4. Save metrics CSV                                                 #
    # ------------------------------------------------------------------ #
    csv_path = os.path.join('results', 'metrics_table.csv')
    with open(csv_path, 'w') as f:
        f.write('gamma,inv_mean,inv_std,sym_mean,sym_std,inv_q_std,sym_q_std\n')
        for g in GAMMAS:
            s = stats_by_gamma[g]
            f.write(
                f"{g},{s['inv_mean']},{s['inv_std']},"
                f"{s['sym_mean']},{s['sym_std']},"
                f"{s['inv_q_std']},{s['sym_q_std']}\n"
            )

    # ------------------------------------------------------------------ #
    # 5a. Figure 1 – P&L distributions                                   #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, len(GAMMAS), figsize=(5 * len(GAMMAS), 5))
    for ax, g in zip(axes, GAMMAS):
        s = stats_by_gamma[g]
        res = all_results[g]
        inv_pnl = res['inventory_pnl']
        sym_pnl = res['symmetric_pnl']
        ax.hist(inv_pnl, bins=50, density=True, alpha=0.6, color='blue', label='Inventory')
        ax.hist(sym_pnl, bins=50, density=True, alpha=0.6, color='orange', label='Symmetric')
        ax.axvline(inv_pnl.mean(), color='blue', linestyle='--', linewidth=1)
        ax.axvline(sym_pnl.mean(), color='orange', linestyle='--', linewidth=1)
        ax.set_title(f"gamma={g}: Inv std={s['inv_std']:.2f}, Sym std={s['sym_std']:.2f}")
        ax.set_xlabel('Final P&L')
        ax.set_ylabel('Frequency')
        ax.legend()
    fig.suptitle('AS 2008: P&L Distribution by Strategy and Gamma')
    fig.tight_layout()
    fig.savefig(os.path.join('results', 'fig1_pnl_distributions.png'), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 5b. Figure 2 – P&L std comparison bar chart                        #
    # ------------------------------------------------------------------ #
    inv_stds = [stats_by_gamma[g]['inv_std'] for g in GAMMAS]
    sym_stds = [stats_by_gamma[g]['sym_std'] for g in GAMMAS]
    x = np.arange(len(GAMMAS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, inv_stds, width, label='Inventory')
    ax.bar(x + width / 2, sym_stds, width, label='Symmetric')
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in GAMMAS])
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Standard Deviation of P&L')
    ax.set_title('P&L Standard Deviation: Inventory vs Symmetric Strategy')
    ax.legend(['Inventory', 'Symmetric'])
    fig.tight_layout()
    fig.savefig(os.path.join('results', 'fig2_std_comparison.png'), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 5c. Figure 3 – Terminal inventory distribution                      #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, len(GAMMAS), figsize=(5 * len(GAMMAS), 5))
    for ax, g in zip(axes, GAMMAS):
        inv_q = all_results[g]['inventory_terminal_q']
        inv_q_mean = stats_by_gamma[g]['inv_q_mean']
        inv_q_std = stats_by_gamma[g]['inv_q_std']
        ax.hist(inv_q, bins=50, color='steelblue', alpha=0.7)
        ax.set_title(f'gamma={g}: mean q_T={inv_q_mean:.2f}, std={inv_q_std:.2f}')
        ax.set_xlabel('Terminal Inventory (units)')
        ax.set_ylabel('Count')
    fig.suptitle('AS 2008: Terminal Inventory Distribution (Inventory Strategy)')
    fig.tight_layout()
    fig.savefig(os.path.join('results', 'fig3_terminal_inventory.png'), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 5d. Figure 4 – Sample mid-price paths                              #
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(42)
    n_show = 10
    n_steps = round(PARAMS['T'] / PARAMS['DT'])
    paths = np.zeros((n_show, n_steps + 1))
    paths[:, 0] = PARAMS['S0']
    for i in range(n_steps):
        signs = rng.choice([-1, 1], size=n_show)
        paths[:, i + 1] = paths[:, i] + signs * PARAMS['SIGMA'] * np.sqrt(PARAMS['DT'])

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    for j in range(n_show):
        ax.plot(paths[j], color=cmap(j), alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mid-Price')
    ax.set_title('Sample Mid-Price Paths (Arithmetic Brownian Motion)')
    fig.tight_layout()
    fig.savefig(os.path.join('results', 'fig4_sample_paths.png'), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 6. Figure 5 – Single-path quotes (gamma=0.1, seed=999)             #
    # ------------------------------------------------------------------ #
    gamma_trace = 0.1
    from src.model import indifference_price, optimal_spread
    n_steps = round(PARAMS['T'] / PARAMS['DT'])
    DT, SIGMA, T, K, A, S0 = (
        PARAMS['DT'], PARAMS['SIGMA'], PARAMS['T'],
        PARAMS['K'], PARAMS['A'], PARAMS['S0'],
    )
    rng2 = np.random.default_rng(999)
    s_path = [S0]
    r_path = []; pa_inv_path = []; pb_inv_path = []
    pa_sym_path = []; pb_sym_path = []
    s = S0; q = 0
    for i in range(n_steps):
        t = i * DT
        spread = optimal_spread(gamma_trace, SIGMA, T, t, K)
        r = indifference_price(s, q, gamma_trace, SIGMA, T, t)
        pa_inv = r + spread / 2; pb_inv = r - spread / 2
        pa_sym = s + spread / 2; pb_sym = s - spread / 2
        r_path.append(r); pa_inv_path.append(pa_inv); pb_inv_path.append(pb_inv)
        pa_sym_path.append(pa_sym); pb_sym_path.append(pb_sym)
        # simulate fills for trace
        u = rng2.uniform(size=2)
        lam_a = A * np.exp(-K * (pa_inv - s))
        lam_b = A * np.exp(-K * (s - pb_inv))
        if u[0] < min(lam_a * DT, 1.0): q -= 1
        if u[1] < min(lam_b * DT, 1.0): q += 1
        sign = 1 if rng2.random() < 0.5 else -1
        s += sign * SIGMA * np.sqrt(DT)
        s_path.append(s)

    steps = np.arange(len(s_path))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, s_path, 'k-', label='Mid-Price', linewidth=1.5)
    ax.plot(steps[:-1], pa_inv_path, 'r-', label='Inv Ask', linewidth=1)
    ax.plot(steps[:-1], pb_inv_path, 'b-', label='Inv Bid', linewidth=1)
    ax.plot(steps[:-1], pa_sym_path, color='orange', linestyle='--', label='Sym Ask', linewidth=1)
    ax.plot(steps[:-1], pb_sym_path, color='green', linestyle='--', label='Sym Bid', linewidth=1)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price')
    ax.set_title('AS 2008: Single Path Quotes (gamma=0.1, Inventory vs Symmetric)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join('results', 'fig5_single_path_quotes.png'), dpi=150)
    plt.close(fig)

    print('Experiment complete. Results saved to results/')
