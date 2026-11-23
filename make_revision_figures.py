"""Figures for the revised Section VI from revision_results/*.csv (same numbering as the paper)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = 'revision_results'
OUT = os.path.join(R, 'figures'); os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, GOLD, PINK, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#8a8984'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'legend.frameon': False, 'lines.linewidth': 1.6})
SC = {'static': ('Static', GREY), 'epoch E=10': ('Epoch, E=10', ORANGE), 'epoch E=50': ('Epoch, E=50', PINK),
      'aecfr': (r'$\alpha$-ECFR, $\alpha$=0.2', GOLD), 'cfp': (r'CFP, $\rho$=0.2', BLUE)}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight', pad_inches=0.02); fig.savefig(f'{OUT}/{name}.png', dpi=220, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def f3():
    d = pd.read_csv(f'{R}/fig3_cumulative_refresh.csv')
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    for c, rho in zip([ORANGE, BLUE, GREEN], sorted(d.rho.unique())):
        g = d[d.rho == rho]
        ax.plot(g.L, g.eq15, color=c, label=fr'$\rho$={rho} (model)')
        ax.plot(g.L, g.measured, 'o', ms=2.8, mfc='white', mec=c, mew=1)
    ax.axhline(0.95, color=INK2, ls='--', lw=0.8)
    ax.text(0.5, 0.955, '95%', ha='left', va='bottom', fontsize=7, color=INK2)
    ax.set_xlabel('Completed replacements k'); ax.set_ylabel(r'$R_{cum}(k)$: originals that have left')
    ax.set_ylim(0, 1.02); ax.legend(fontsize=6.8, loc='lower right')
    save(fig, 'fig3_cumulative_refresh')


def f4():
    d = pd.read_csv(f'{R}/fig4_rho_sensitivity.csv')
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.45, 4.6))
    a.errorbar(d.rho, d.mean_dur, yerr=d.mean_dur_ci95, color=BLUE, marker='o', ms=3.5, capsize=2, label='Mean')
    a.plot(d.rho, d.p95_dur, color=BLUE, ls='--', marker='s', ms=3, label='95th percentile')
    a.set_xlabel(r'Rotation rate $\rho$'); a.set_ylabel('Capture duration (blocks)')
    a.legend(fontsize=7); a.set_title('(a) Capture duration, f = 0.25', fontsize=8.5)
    x = np.arange(len(d))
    b.bar(x - 0.18, d.handoff_MB_full, 0.36, color=ORANGE, label='Full checkpoint')
    b.bar(x + 0.18, d.handoff_MB_delta, 0.36, color=GOLD, label='Records since last sync')
    b.axhline(0.256, color=INK2, ls='--', lw=0.8, label='Block payload (0.26 MB)')
    for xi, L in zip(x, d.blocks_to_95pct_refresh):
        b.text(xi, d.handoff_MB_full.iloc[xi] + 0.2, f'{L}', ha='center', fontsize=6.5)
    b.set_xticks(x); b.set_xticklabels([f'{v:g}' for v in d.rho])
    b.set_xlabel(r'Rotation rate $\rho$'); b.set_ylabel('Hand-off data per shard-block (MB)')
    b.set_title('(b) Hand-off data (labels: blocks to 95% refresh)', fontsize=8)
    b.legend(fontsize=6.5, loc='upper left'); b.grid(axis='x', visible=False)
    save(fig, 'fig4_rho_sensitivity')


LAB = {('none', 0.0): 'No attack', ('none', 0.02): 'No attack,\n2% delay',
       ('none', 0.05): 'No attack,\n5% delay', ('coordinated', 0.0): 'Coordinated\naccept',
       ('invalid', 0.0): 'Invalid\nproposals', ('withhold', 0.0): 'Withholding', ('equivocate', 0.0): 'Equivocation'}


def f5_8():
    d = pd.read_csv(f'{R}/fig5_8_cfi_controller.csv')
    c = d[d['mode'] == 'controller']
    keys = list(LAB)
    rows = [c[(c.strategy == s) & np.isclose(c.p_net, p)].iloc[0] for s, p in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    ax.bar(x, [r.cfi_mean for r in rows], color=BLUE, width=0.6)
    for xi, r in zip(x, rows):
        ax.text(xi, r.cfi_mean + 0.01, f'{r.cfi_mean:.2f}', ha='center', va='bottom', fontsize=7)
    ax.axhline(0.5, color=INK2, ls='--', lw=0.8); ax.text(-0.4, 0.52, r'$\tau_{CFI}$ = 0.5', ha='left', fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([LAB[k].replace('\n', ' ') for k in keys], fontsize=6.5, rotation=40, ha='right')
    ax.set_ylabel('Mean CFI'); ax.set_ylim(0, 0.7); ax.grid(axis='x', visible=False)
    save(fig, 'fig5_cfi_values')
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.45, 4.4))
    for ax, col, title in [(a, 'increase_share', '(a) Share of windows with increase'),
                           (b, 'mean_r', '(b) Mean replacements per block')]:
        vals = [r[col] for r in rows]
        ax.barh(x, vals, color=ORANGE if col == 'increase_share' else BLUE, height=0.6)
        for xi, v in zip(x, vals):
            ax.text(v, xi, f' {v:.2f}' if col == 'increase_share' else f' {v:.1f}', va='center', fontsize=7)
        ax.set_title(title, fontsize=8.5); ax.grid(axis='y', visible=False)
    for ax in (a, b):
        ax.set_yticks(x); ax.set_yticklabels([LAB[k].replace('\n', ' ') for k in keys], fontsize=6.5); ax.invert_yaxis()
    a.set_xlim(0, 0.55); b.set_xlim(0, 32)
    save(fig, 'fig8_controller_response')


def f6():
    d = pd.read_csv(f'{R}/fig6_capture_rate.csv')
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    s, c = d[d.policy == 'static'], d[d.policy == 'cfp']
    ax.axvspan(1 / 3, 0.42, color=GRID, alpha=0.6, lw=0)
    ax.text(0.338, 0.05, 'outside\nf < 1/3', fontsize=6.5, color=INK2)
    ax.plot(c.f, c.eq34, color=INK2, lw=1, label='Model')
    ax.errorbar(s.f - 0.003, s.capture_frac, yerr=s.capture_frac_ci95, fmt='s', ms=3.5, color=GREY, capsize=2, label='Static (sim.)')
    ax.errorbar(c.f + 0.003, c.capture_frac, yerr=c.capture_frac_ci95, fmt='o', ms=3.5, color=BLUE, capsize=2, label='CFP (sim.)')
    ax.set_xlabel('Global adversary fraction f'); ax.set_ylabel('Fraction of shard-blocks captured')
    ax.set_xlim(0.13, 0.42); ax.legend(fontsize=7, loc='upper left')
    save(fig, 'fig6_capture_rate')


def f7():
    d = pd.read_csv(f'{R}/fig7_12_strategies.csv')
    order = ['epoch E=50', 'epoch E=10', 'cfp']
    fs = sorted(d.f.unique()); w = 0.26
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    for i, s in enumerate(order):
        g = d[d.strategy == s].set_index('f').reindex(fs)
        xs = np.arange(len(fs)) + (i - 1) * w
        ax.bar(xs, g.mean_dur, w, yerr=g.mean_dur_ci95, color=SC[s][1], label=SC[s][0],
               error_kw=dict(lw=0.8, capsize=1.5, ecolor=INK2))
        for xx, v in zip(xs, g.mean_dur):
            ax.text(xx, v * 1.1, f'{v:.1f}', ha='center', fontsize=6, rotation=90, va='bottom')
    ax.set_yscale('log'); ax.set_ylim(1, 600)
    ax.set_xticks(range(len(fs))); ax.set_xticklabels([f'f = {f:.2f}' for f in fs])
    ax.set_ylabel('Mean capture duration (blocks)'); ax.grid(axis='x', visible=False)
    ax.legend(fontsize=6.5, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.16))
    save(fig, 'fig7_capture_duration')


def f9():
    tp = pd.read_csv(f'{R}/tput.csv')
    g = tp[(tp.K == 16) & (tp.rate == 4000) & (tp.T_ho == 0) & ~tp.system.str.startswith('CFP hand-off')]
    order = ['Monoxide (relay)', 'OmniLedger (Atomix)', 'RapidChain', 'BrokerChain (CLPA + 40 brokers)',
             'CFP (Alg. 1 + Alg. 5)', 'Ablation: CFP commit, hash placement', 'Ablation: relay, CFP placement']
    g = g.set_index('system').reindex(order)
    col = [GREY, ORANGE, PINK, GOLD, BLUE, '#9ec5f0', GREEN]
    h = tp[(tp.K == 16) & tp.system.str.startswith('CFP hand-off')]
    fig, (a, c) = plt.subplots(2, 1, figsize=(3.45, 4.8), gridspec_kw=dict(height_ratios=[1.5, 1]))
    y = np.arange(len(order))
    a.barh(y, g.tps, color=col, height=0.6)
    for yi, v, l in zip(y, g.tps, g.lat_mean):
        a.text(v, yi, f' {v:.0f} tx/s, {l:.1f} s', va='center', fontsize=6)
    a.set_yticks(y); a.set_yticklabels([o.replace('Ablation: ', 'Abl.: ').replace(' (CLPA + 40 brokers)', ' (40 brokers)') for o in order], fontsize=6); a.invert_yaxis(); a.set_xlim(0, g.tps.max() * 1.85)
    a.set_title('(a) Throughput and mean latency, K = 16', fontsize=8.5); a.grid(axis='y', visible=False)
    c.plot(h.T_ho, h.tps, color=BLUE, marker='o', ms=3.5)
    c.set_xlabel('Hand-off time per block (s)'); c.set_ylabel('CFP throughput (tx/s)')
    c.set_title('(b) CFP vs hand-off time', fontsize=8.5); c.set_ylim(0, h.tps.max() * 1.2)
    save(fig, 'fig9_performance')


def f11():
    d = pd.read_csv(f'{R}/fig11_shards.csv')
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.45, 4.4))
    c, e = d[d.policy == 'cfp'], d[d.policy == 'epoch']
    lab = [f'{k}\n(n={n})' for k, n in zip(c.K, c.n)]
    a.plot(lab, c.eq34, color=INK2, lw=1, label='Model')
    a.plot(lab, c.capture_frac, 'o', color=BLUE, label='CFP (sim.)')
    a.set_ylabel('Fraction of shard-blocks captured'); a.set_xlabel('Shards K (N = 3200, f = 0.25)')
    a.legend(fontsize=7); a.set_title('(a) Capture rate', fontsize=8.5)
    b.plot(lab, e.mean_dur, color=PINK, marker='s', label='Epoch, E=50')
    b.plot(lab, c.mean_dur, color=BLUE, marker='o', label=r'CFP, $\rho$=0.2')
    b.set_yscale('log'); b.set_ylabel('Mean capture duration (blocks)'); b.set_xlabel('Shards K')
    b.legend(fontsize=7); b.set_title('(b) Capture duration', fontsize=8.5)
    save(fig, 'fig11_shards')


def f12():
    d = pd.read_csv(f'{R}/fig7_12_strategies.csv')
    d = d[np.isclose(d.f, 0.30)].set_index('strategy')
    order = ['static', 'epoch E=50', 'epoch E=10', 'aecfr', 'cfp']
    H = 2000
    metrics = [('max_dur', 'Max capture duration'), ('p95_dur', '95th-pct duration'), ('mean_dur', 'Mean duration')]
    fig, axes = plt.subplots(3, 1, figsize=(3.45, 4.4), sharex=True)
    for ax, (m, t) in zip(axes, metrics):
        vals = [H if s == 'static' else d.loc[s, m] for s in order]
        ax.barh(range(len(order)), vals, color=[SC[s][1] for s in order], height=0.6)
        for i, v in enumerate(vals):
            ax.text(v * 1.1, i, 'whole run' if order[i] == 'static' else f'{v:.1f}' if v < 100 else f'{v:.0f}',
                    va='center', fontsize=6.5)
        ax.set_xscale('log'); ax.set_xlim(1, 2e4); ax.set_title(t + ' (blocks)', fontsize=8); ax.grid(axis='y', visible=False)
    for ax in axes:
        ax.set_yticks(range(len(order))); ax.set_yticklabels([SC[s][0] for s in order], fontsize=6.5); ax.invert_yaxis()
    save(fig, 'fig12_comparison')


def f13():
    d = pd.read_csv(f'{R}/fig13_timeline_seed0.csv')
    d = d[np.isclose(d.f, 0.40)]
    order = ['static', 'epoch E=50', 'epoch E=10', 'aecfr', 'cfp']
    fig, axes = plt.subplots(len(order), 1, figsize=(3.45, 4.2), sharex=True)
    for ax, s in zip(axes, order):
        g = d[d.strategy == s]
        ax.step(g.block, g.shard0_adv, where='post', color=SC[s][1], lw=1)
        ax.axhline(9, color=INK2, ls='--', lw=0.7)
        ax.set_ylim(0, 25); ax.set_yticks([0, 9, 25])
        ax.set_ylabel(SC[s][0], rotation=0, ha='right', va='center', fontsize=6.5); ax.tick_params(labelsize=6.5)
    axes[-1].set_xlabel('Block')
    save(fig, 'fig13_stress_timeline')


def f14():
    d = pd.read_csv(f'{R}/fig7_12_strategies.csv')
    d = d[np.isclose(d.f, 0.30)].set_index('strategy')
    c58 = pd.read_csv(f'{R}/fig5_8_cfi_controller.csv')
    eq = c58[(c58.strategy == 'equivocate') & (c58['mode'] == 'controller')].iloc[0].increase_share
    order = ['static', 'epoch E=50', 'epoch E=10', 'aecfr', 'cfp']
    H = 2000
    labels = ['Max capture\nduration', '95th-pct\nduration', 'Mean\nduration', 'Availability\n(1 - capture rate)',
              'Response to\nequivocation proofs']
    vals = {}
    for s_ in order:
        dur = [H if s_ == 'static' else d.loc[s_, m] for m in ['max_dur', 'p95_dur', 'mean_dur']]
        vals[s_] = dur + [1 - d.loc[s_, 'capture_frac'], eq if s_ == 'cfp' else 0.0]
    mx = [max(vals[t][i] for t in order) for i in range(3)]
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist(); ang += ang[:1]
    fig = plt.figure(figsize=(3.45, 3.6)); ax = plt.subplot(111, polar=True)
    for s_ in order:
        v = [1 - np.log1p(vals[s_][i]) / np.log1p(mx[i]) for i in range(3)] + [vals[s_][3], vals[s_][4] / max(eq, 1e-9)]
        v += v[:1]
        ax.plot(ang, v, color=SC[s_][1], lw=1.4, label=SC[s_][0]); ax.fill(ang, v, color=SC[s_][1], alpha=0.07)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=6.5); ax.tick_params(axis='x', pad=9); ax.set_yticklabels([]); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.3, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3)
    save(fig, 'fig14_profile')


PL_ORDER = ['Hash (Monoxide/OmniLedger/RapidChain)', 'Jump hash (Tangram)', 'CLPA (BrokerChain)', 'Metis',
            'Louvain, no budgets', 'CFP Alg. 1']
PL_COL = [GREY, PINK, GOLD, ORANGE, GREEN, BLUE]


def f10():
    d = pd.read_csv(f'{R}/fig10_placement.csv')
    Ks = sorted(d.K.unique()); w = 0.13
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.45, 4.8))
    for i, (m, c) in enumerate(zip(PL_ORDER, PL_COL)):
        g = d[d.method == m].set_index('K').reindex(Ks)
        xs = np.arange(len(Ks)) + (i - 2.5) * w
        a.bar(xs, g.cross_future, w, color=c, label=m)
        b.bar(xs, g.max_load_over_avg, w, color=c)
    for ax, t in [(a, '(a) Cross-shard ratio, evaluation window'), (b, '(b) Max shard load / average')]:
        ax.set_xticks(range(len(Ks))); ax.set_xticklabels([f'K = {k}' for k in Ks]); ax.set_title(t, fontsize=8.5)
        ax.grid(axis='x', visible=False)
    b.legend(*a.get_legend_handles_labels(), fontsize=6, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.15))
    save(fig, 'fig10_placement')


def f15():
    d = pd.read_csv(f'{R}/tput.csv')
    g = d[(d.K == 16) & (d.rate == 4000) & (d.T_ho == 0) & ~d.system.str.startswith('CFP hand-off')]
    order = ['Monoxide (relay)', 'OmniLedger (Atomix)', 'RapidChain', 'BrokerChain (CLPA + 40 brokers)',
             'CFP (Alg. 1 + Alg. 5)', 'Ablation: CFP commit, hash placement', 'Ablation: relay, CFP placement']
    g = g.set_index('system').reindex(order)
    col = [GREY, ORANGE, PINK, GOLD, BLUE, '#9ec5f0', GREEN]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.1, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    y = np.arange(len(order))
    a.barh(y, g.tps, color=col, height=0.6); b.barh(y, g.lat_mean, color=col, height=0.6)
    for yi, v in zip(y, g.tps): a.text(v, yi, f' {v:.0f}', va='center', fontsize=6.5)
    for yi, v in zip(y, g.lat_mean): b.text(v, yi, f' {v:.1f}', va='center', fontsize=6.5)
    a.set_yticks(y); a.set_yticklabels(order, fontsize=6.5); a.invert_yaxis(); b.set_yticks([]); b.invert_yaxis()
    a.set_title('(a) Throughput (tx/s)', fontsize=8.5); b.set_title('(b) Mean latency (s)', fontsize=8.5)
    a.set_xlim(0, g.tps.max() * 1.3); b.set_xlim(0, g.lat_mean.max() * 1.3)
    for ax in (a, b): ax.grid(axis='y', visible=False)
    h = d[(d.K == 16) & d.system.str.startswith('CFP hand-off')]
    c.plot(h.T_ho, h.tps, color=BLUE, marker='o', ms=3.5)
    c.set_xlabel('Hand-off time per block (s)'); c.set_title('(c) CFP vs hand-off time', fontsize=8.5)
    c.set_ylim(0, h.tps.max() * 1.2)
    save(fig, 'fig15_throughput')


if __name__ == '__main__':
    for f in [f3, f4, f5_8, f6, f7, f9, f10, f11, f12, f13, f14]:
        try:
            f()
        except FileNotFoundError as e:
            print('skip', f.__name__, e)
