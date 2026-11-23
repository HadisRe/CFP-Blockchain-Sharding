"""
Numerical check of the Section V formulas against the project simulator.
Writes revision_results/verify_formulas.csv (one row per check) and verify_survival.csv.
"""
import os, sys, math, random
import numpy as np, pandas as pd
from cfp import analysis as an
from cfp.attack import RotationStrategyExperiment, capture_episodes
from cfp.fluidity import ValidatorFluidity
from cfp.cross_shard import AsyncCommitSimulator
from sharding import Network

QUICK = '--quick' in sys.argv
OUT = 'revision_results'; os.makedirs(OUT, exist_ok=True)
SEEDS = 6 if QUICK else 40
BL = 1500 if QUICK else 4000
rows, surv_rows = [], []


def add(check, eq, config, theory, sim, ci=None, note=''):
    rows.append(dict(check=check, equation=eq, config=config, theory=theory, simulation=sim, sim_ci95=ci, note=note))


def sim_counts(K, n, N, f_M, r, policy='cfp', blocks=BL, seeds=SEEDS, **kw):
    out = []
    for s in range(seeds):
        e = RotationStrategyExperiment(K, n, N, f_M / N, s)
        # exact adversary count M
        pool = e.net.validator_pool.validators
        for v in pool:
            v.is_adversarial = False
        for v in e.rng.sample(pool, f_M):
            v.is_adversarial = True
        kw2 = dict(kw)
        if policy == 'cfp':
            kw2['rho'] = r / n
        out.append(e.run(policy, blocks, **kw2))
    return out


# ---------- Eq. (15): uninterrupted retention ----------
def eq15():
    n, K, N = 50, 16, 1600
    for r in [5, 10, 15]:
        acc = np.zeros(31)
        for s in range(SEEDS):
            random.seed(s)
            net = Network(K, n, N)
            flu = ValidatorFluidity(net, rho=r / n, fast=True, rng=random.Random(s))
            orig = {sid: {v.validator_id for v in sh.validators} for sid, sh in net.shards.items()}
            left = {sid: set() for sid in net.shards}
            for k in range(1, 31):
                info = flu.rotate_all_shards(block_hash='0' * 16)
                for i in info:
                    left[i['shard_id']].update(set(i['removed']) & orig[i['shard_id']])
                acc[k] += np.mean([len(left[sid]) / n for sid in net.shards])
        acc /= SEEDS
        for k in [5, 10, 14, 20, 30]:
            add('Uninterrupted-retention loss R_cum(k)', '15', f'n={n}, r={r}, k={k}', 1 - (1 - r / n) ** k, acc[k])


# ---------- Eqs. (33)-(37): K = 1 ----------
def eq33_37_K1():
    n, N, M = 30, 150, 45
    ps = an.p_static(n, N, M)
    for r in [3, 6, 10]:
        P = an.kernel_K1(n, N - n, M, r); mu = an.mu_K1(n, N - n, M)
        assert np.allclose(mu @ P, mu), 'mu not stationary'
        th = an.episode_quantities(P, mu, np.arange(n + 1) >= an.b_cap(n), Lmax=40, H=20)
        runs = sim_counts(1, n, N, M, r)
        cap = np.concatenate([c >= an.b_cap(n) for c in runs])
        d = np.concatenate([capture_episodes(c, n)[0] for c in runs])
        means = [capture_episodes(c, n)[0].mean() for c in runs if len(capture_episodes(c, n)[0])]
        cfg = f'K=1, n={n}, N={N}, M={M}, r={r}'
        add('Stationary capture probability', '33-34', cfg, ps, float(cap.mean()))
        add('Mean episode duration nu(I-Q)^-1 1', '37', cfg, th['mean'], float(d.mean()),
            1.96 * np.std(means, ddof=1) / np.sqrt(len(means)))
        add('E[min(D,20)]', '37b', cfg, th['mean_min_H'], float(np.minimum(d, 20).mean()))
        # p_seq(L): fraction of observation windows of length L fully captured (stationary start)
        for L in [2, 5]:
            sim = np.mean([np.mean([c[k:k + L].all() for k in range(0, len(c) - L, 7)])
                           for c in [x[:, 0] >= an.b_cap(n) for x in runs]])
            add(f'p_seq(L={L}) at stationarity', '35', cfg, th['pseq'][L - 1], float(sim))
        for L in range(1, 31):
            surv_rows.append(dict(config=cfg, r=r, L=L, theory=th['survival'][L - 1], sim=float((d >= L).mean())))


# ---------- Eqs. (32)-(37): joint K = 2 shared pool ----------
def eq32_37_K2():
    n, s, M, r = 15, 30, 18, 3
    P, mu, first = an.kernel_K2(n, s, M, r)
    assert np.allclose(mu @ P, mu)
    th = an.episode_quantities(P, mu, first >= an.b_cap(n), Lmax=40)
    runs = sim_counts(2, n, 2 * n + s, M, r)
    c0 = [x[:, :1] for x in runs]
    d = np.concatenate([capture_episodes(c, n)[0] for c in c0])
    cfg = f'K=2 joint, n={n}, s={s}, M={M}, r={r}'
    add('Stationary capture probability (shard 1)', '33', cfg, th['p_capture'],
        float(np.concatenate([c >= an.b_cap(n) for c in c0]).mean()))
    add('Mean episode duration (shard 1)', '36-37', cfg, th['mean'], float(d.mean()))
    for L in range(1, 31):
        surv_rows.append(dict(config=cfg, r=r, L=L, theory=th['survival'][L - 1], sim=float((d >= L).mean())))


# ---------- Eq. (38): endpoint overlap ----------
def eq38():
    n, N = 30, 150
    for r in [3, 6]:
        acc = {k: [] for k in [1, 5, 10, 30, 100]}
        for s in range(SEEDS):
            random.seed(s)
            net = Network(1, n, N)
            flu = ValidatorFluidity(net, rho=r / n, fast=True, rng=random.Random(s))
            orig = {v.validator_id for v in net.shards[0].validators}
            for k in range(1, 101):
                flu.rotate_all_shards(block_hash='0' * 16)
                if k in acc:
                    acc[k].append(len(orig & {v.validator_id for v in net.shards[0].validators}))
        for k, v in acc.items():
            add('Endpoint overlap E|V(0) n V(k)|', '38', f'K=1, n={n}, N={N}, r={r}, k={k}',
                an.endpoint_overlap(n, N, r, k), float(np.mean(v)))


# ---------- Eq. (39): epoch benchmark ----------
def eq39():
    K, n, N, M, E = 16, 50, 1600, 400, 10
    ps = an.p_static(n, N, M)
    runs = sim_counts(K, n, N, M, None, policy='epoch', epoch_length=E, blocks=BL)
    # with a random phase per run, average p_epoch over the uniform start offset a
    for L in [1, 5, 10, 15, 25]:
        th = np.mean([an.p_epoch(ps, a, L, E) for a in range(E)])
        sim = np.mean([np.mean([(c[k:k + L, sh] >= an.b_cap(n)).all() for k in range(0, len(c) - L, 3)
                                for sh in range(K)]) for c in runs])
        add(f'p_epoch(L={L}) averaged over offset a', '39', f'K={K}, n={n}, N={N}, M={M}, E={E}', th, float(sim))
    d = np.concatenate([capture_episodes(c, n)[0] for c in runs])
    add('Epoch mean episode duration E/(1-p_static)', '39', f'K={K}, n={n}, N={N}, M={M}, E={E}', E / (1 - ps), float(d.mean()))


# ---------- Eqs. (41)/(43): quorum intersection arithmetic ----------
def eq41():
    for n in [4, 25, 30, 50, 100, 101, 102]:
        b = math.ceil(n / 3) - 1; q = n - b
        add('Quorum intersection 2q - n >= b + 1', '41/43', f'n={n}, b={b}, q={q}', b + 1, 2 * q - n,
            note='holds' if 2 * q - n >= b + 1 else 'VIOLATED')


# ---------- Eqs. (44), (45), (47) ----------
def eq44_47():
    n, K = 50, 4; N = 400; M = 100; delta = 0.05
    th = an.pin_bounds(n, N, M, delta)
    res = []
    for s in range(3 if QUICK else 10):
        random.seed(s)
        net = Network(K, n, N)
        rng = random.Random(s)
        for v in rng.sample(net.validator_pool.validators, M):
            v.is_adversarial = True
        flu = ValidatorFluidity(net, rho=0.2, fast=True, rng=rng)
        res.append(AsyncCommitSimulator(net, flu, s, gst=0, delta=delta, n_tx=600 if QUICK else 1500).run())
    df = pd.DataFrame(res)
    cfg = f'n_pin={n}, N={N}, M={M}, delta={delta}'
    add('Pinned set outside fault bound p_ctrl', '45', cfg, th['p_ctrl'],
        float(df.signer_sets_outside_bound.sum() / (2 * df.cross.sum())), note='per pinned population')
    add('Quorum deficit on first collection p_live / (1 - p_ctrl)', '47', cfg, th['p_live'] / (1 - th['p_ctrl']),
        float(df.first_deficits.sum() / df.first_attempts.sum()), note='conditional on the safe regime')
    add('First-attempt decisions within Eq. (44) budget 3Delta + 2T_b', '44', 'Delta=200 ms, T_b=1000 ms', 1.0,
        float(df.first_try_within_eq44.sum() / df.first_try_n.sum()), note=f'max {df.first_try_max_ms.max():.0f} ms')


if __name__ == '__main__':
    for f in [eq15, eq33_37_K1, eq32_37_K2, eq38, eq39, eq41, eq44_47]:
        f(); print(f.__name__, 'done', flush=True)
    df = pd.DataFrame(rows)
    df['abs_diff'] = (df.theory - df.simulation).abs()
    df.to_csv(f'{OUT}/verify_formulas.csv', index=False)
    pd.DataFrame(surv_rows).to_csv(f'{OUT}/verify_survival.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
    print(df.round(4).to_string())
