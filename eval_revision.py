"""
Revised evaluation (Section VI) produced with the project's own simulator after the
reviewer-requested fixes. Writes CSV files and figures to revision_results/.

    python eval_revision.py            # all experiments
    python eval_revision.py --quick    # short smoke run
    python eval_revision.py fig6 fig7  # selected experiments
"""
import os, sys, math, time, random
from math import comb
from multiprocessing import Pool
import numpy as np
import pandas as pd

from sharding import Network
from cfp.fluidity import ValidatorFluidity
from cfp.attack import (RotationStrategyExperiment, ConsensusMonitoringExperiment,
                        capture_episodes, capture_count)
from cfp.cross_shard import AsyncCommitSimulator
from cfp.partitioning import AccountPartitioner, repartition_balanced, shard_load_ratio
import cfp.placement as PL
import cfp.throughput as TP
from cfp.messages import consensus_msgs, onboarding

QUICK = '--quick' in sys.argv
OUT = 'revision_results'
os.makedirs(OUT, exist_ok=True)

# Default configuration (Table I)
K, n, N = 16, 50, 1600          # 16 shards x 50 validators, standby pool s = 800
RHO = 0.2                       # r = ceil(0.2 * 50) = 10
BLOCKS = 600 if QUICK else 2000
SEEDS = 4 if QUICK else 20
F_MAIN = [0.20, 0.25, 0.30]
F_STRESS = [0.35, 0.40]


def p_capture_eq34(n, N, M):
    """Eq. (34): Pr(B >= ceil(n/3)) for a committee of n drawn from N identities with M adversarial."""
    tot = comb(N, n)
    return sum(comb(M, j) * comb(N - M, n - j) for j in range(capture_count(n), min(n, M) + 1)) / tot


def summarize(counts, n_):
    d, rc, lc = capture_episodes(counts, n_)
    return dict(capture_frac=float((counts >= capture_count(n_)).mean()), episodes=len(d),
                mean_dur=float(d.mean()) if len(d) else np.nan,
                p95_dur=float(np.percentile(d, 95)) if len(d) else np.nan,
                max_dur=int(d.max()) if len(d) else 0,
                right_censored=len(rc), max_any=int(max([d.max() if len(d) else 0] + list(rc) + [0])))


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def _strategy_job(args):
    policy, f, seed, kw, cfg = args
    K_, n_, N_ = cfg
    e = RotationStrategyExperiment(K_, n_, N_, f, seed)
    c = e.run(policy, BLOCKS, **kw)
    return dict(policy=policy, f=f, seed=seed, K=K_, n=n_, N=N_, **{k: v for k, v in kw.items()}, **summarize(c, n_))


def run_jobs(jobs):
    with Pool(max(1, os.cpu_count())) as p:
        return p.map(_strategy_job, jobs)


def agg(df, keys):
    g = df.groupby(keys, dropna=False)
    out = g.agg(capture_frac=('capture_frac', 'mean'), mean_dur=('mean_dur', 'mean'),
                p95_dur=('p95_dur', 'mean'), max_dur=('max_dur', 'max'), episodes=('episodes', 'sum'),
                right_censored=('right_censored', 'sum'), runs=('seed', 'count')).reset_index()
    out['mean_dur_ci95'] = g['mean_dur'].apply(ci).values
    out['capture_frac_ci95'] = g['capture_frac'].apply(ci).values
    return out


# ---------------- Fig. 3: cumulative refresh (measured vs Eq. 15) ----------------
def fig3():
    rows = []
    for rho in [0.1, 0.2, 0.3]:
        for seed in range(SEEDS):
            random.seed(seed)
            net = Network(K, n, N)
            flu = ValidatorFluidity(net, rho=rho, fast=True, rng=random.Random(seed))
            init = {sid: {v.validator_id for v in sh.validators} for sid, sh in net.shards.items()}
            left = {sid: set() for sid in net.shards}
            for L in range(1, 31):
                for info in flu.rotate_all_shards(block_hash='0' * 16):
                    left[info['shard_id']].update(set(info['removed']) & init[info['shard_id']])
                gone = np.mean([len(left[s_]) / n for s_ in net.shards])   # never-left originals (Eq. 15)
                rows.append(dict(rho=rho, L=L, seed=seed, measured=gone, eq15=1 - (1 - math.ceil(rho * n) / n) ** L))
    df = pd.DataFrame(rows).groupby(['rho', 'L']).agg(measured=('measured', 'mean'), eq15=('eq15', 'first')).reset_index()
    df.to_csv(f'{OUT}/fig3_cumulative_refresh.csv', index=False)


# ---------------- Fig. 4: rotation-rate sensitivity ----------------
def fig4():
    rhos = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    jobs = [('cfp', 0.25, s, dict(rho=r), (K, n, N)) for r in rhos for s in range(SEEDS)]
    df = agg(pd.DataFrame(run_jobs(jobs)), ['rho'])
    df['r'] = [math.ceil(r * n) for r in df.rho]
    df['blocks_to_95pct_refresh'] = [math.ceil(math.log(0.05) / math.log(1 - math.ceil(r * n) / n)) for r in df.rho]
    # Alg. 3 draw cost (seed proof verification + UniformSample) and hand-off data per block
    from cfp.fluidity import alg3_draw
    from ecdsa import SigningKey, SECP256k1
    sk = SigningKey.generate(curve=SECP256k1); vk = sk.get_verifying_key()
    A = len(trace()['accts'])
    ms, full, delta = [], [], []
    for r_ in rhos:
        rr = math.ceil(r_ * n)
        t = time.perf_counter()
        for k in range(20):
            alg3_draw('cfp', 0, 'h', k, list(range(n)), list(range(n, n + N - K * n)), rr, sk, vk)
        ms.append((time.perf_counter() - t) / 20 * 1000)
        full.append(onboarding(n, rr, K, n, A / K, 2000, 'full')['bytes'] / 1e6)
        delta.append(onboarding(n, rr, K, n, A / K, 2000, 'delta')['bytes'] / 1e6)
    df['alg3_draw_ms_per_shard'] = ms
    df['handoff_MB_full'] = full
    df['handoff_MB_delta'] = delta
    df.to_csv(f'{OUT}/fig4_rho_sensitivity.csv', index=False)


# ---------------- Figs. 5 and 8: CFI and the bounded controller ----------------
def _cfi_job(args):
    strat, f, pnet, mode, seed = args[:5]
    tau = args[5] if len(args) > 5 else 0.5
    H = 1000 if QUICK else 3000
    e = ConsensusMonitoringExperiment(K, n, N, f, seed)
    kw = {} if mode == 'controller' else dict(r_fixed=math.ceil(0.1 * n))
    o = e.run(H, strategy=strat, p_net=pnet, tau=tau, **kw)
    rec = o['records']
    c = [r['cfi'] for r in rec if r['cfi'] is not None]
    s = summarize(o['counts'], n)
    return dict(strategy=strat, f=f, p_net=pnet, mode=mode, seed=seed, tau=tau,
                cfi_mean=float(np.mean(c)) if c else np.nan,
                cfi_p10=float(np.percentile(c, 10)) if c else np.nan,
                cfi_available=len(c) / max(1, len(rec)),
                increase_share=float(np.mean([r['A'] for r in rec])) if rec else np.nan,
                mean_r=float(o['r_hist'].mean()), **s)


def fig5_8():
    cases = [('none', 0.25, 0.0), ('none', 0.25, 0.02), ('none', 0.25, 0.05),
             ('coordinated', 0.30, 0.0), ('invalid', 0.30, 0.0), ('withhold', 0.30, 0.0),
             ('equivocate', 0.30, 0.0)]
    jobs = [(s, f, p, m, seed) for s, f, p in cases for m in ['controller', 'fixed']
            for seed in range(max(3, SEEDS // 2))]
    with Pool(max(1, os.cpu_count())) as p:
        df = pd.DataFrame(p.map(_cfi_job, jobs))
    g = df.groupby(['strategy', 'f', 'p_net', 'mode']).agg(
        cfi_mean=('cfi_mean', 'mean'), cfi_p10=('cfi_p10', 'mean'), cfi_available=('cfi_available', 'mean'),
        increase_share=('increase_share', 'mean'), mean_r=('mean_r', 'mean'), capture_frac=('capture_frac', 'mean'),
        mean_dur=('mean_dur', 'mean'), p95_dur=('p95_dur', 'mean'), max_dur=('max_dur', 'max')).reset_index()
    g.to_csv(f'{OUT}/fig5_8_cfi_controller.csv', index=False)


# ---------------- Sensitivity of the controller to tau_CFI (R3.2) ----------------
def fig8_tau():
    cases = [('none', 0.25, 0.0), ('none', 0.25, 0.05), ('invalid', 0.30, 0.0),
             ('withhold', 0.30, 0.0), ('equivocate', 0.30, 0.0)]
    jobs = [(s, f, p, 'controller', seed, tau) for s, f, p in cases for tau in [0.2, 0.35, 0.5, 0.65, 0.8]
            for seed in range(max(3, SEEDS // 4))]
    with Pool(max(1, os.cpu_count())) as p:
        df = pd.DataFrame(p.map(_cfi_job, jobs))
    df.groupby(['strategy', 'f', 'p_net', 'tau']).agg(
        cfi_mean=('cfi_mean', 'mean'), increase_share=('increase_share', 'mean'), mean_r=('mean_r', 'mean'),
        mean_dur=('mean_dur', 'mean')).reset_index().to_csv(f'{OUT}/fig8_tau_sensitivity.csv', index=False)


# ---------------- Fig. 6: capture rate vs adversary fraction ----------------
def fig6():
    fs = [0.15] + F_MAIN + F_STRESS
    jobs = [(pol, f, s, dict(rho=RHO) if pol == 'cfp' else {}, (K, n, N)) for pol in ['static', 'cfp']
            for f in fs for s in range(SEEDS)]
    df = agg(pd.DataFrame(run_jobs(jobs)), ['policy', 'f'])
    df['eq34'] = [p_capture_eq34(n, N, int(round(f * N))) for f in df.f]
    df.to_csv(f'{OUT}/fig6_capture_rate.csv', index=False)


# ---------------- Figs. 7, 12, 14: capture duration by strategy ----------------
STRATS = [('static', {}), ('epoch', dict(epoch_length=10)), ('epoch', dict(epoch_length=50)),
          ('aecfr', dict(alpha=RHO)), ('cfp', dict(rho=RHO))]


def fig7_12():
    jobs = [(pol, f, s, kw, (K, n, N)) for f in F_MAIN for pol, kw in STRATS for s in range(SEEDS)]
    df = pd.DataFrame(run_jobs(jobs))
    df['strategy'] = [p if p not in ('epoch',) else f'epoch E={int(e)}' for p, e in zip(df.policy, df.get('epoch_length', np.nan))]
    out = agg(df, ['strategy', 'f'])
    out.to_csv(f'{OUT}/fig7_12_strategies.csv', index=False)


# ---------------- Fig. 13: stress timeline (beyond f < 1/3) ----------------
def fig13():
    rows, tl = [], []
    cfg = (4, 25, 200)
    for f in [0.40, 0.45]:
        for pol, kw in STRATS:
            for s in range(SEEDS):
                e = RotationStrategyExperiment(*cfg, f, s)
                c = e.run(pol, 300, **kw)
                name = pol if pol != 'epoch' else f"epoch E={kw['epoch_length']}"
                rows.append(dict(strategy=name, f=f, seed=s, **summarize(c, 25)))
                if s == 0:
                    cap0 = (c[:, 0] >= capture_count(25)).astype(int)
                    tl += [dict(strategy=name, f=f, block=b, shard0_captured=int(v), shard0_adv=int(a))
                           for b, (v, a) in enumerate(zip(cap0, c[:, 0]))]
    agg(pd.DataFrame(rows), ['strategy', 'f']).to_csv(f'{OUT}/fig13_stress.csv', index=False)
    pd.DataFrame(tl).to_csv(f'{OUT}/fig13_timeline_seed0.csv', index=False)


# ---------------- Fig. 11: shard-count trade-off at fixed population ----------------
def fig11():
    Ntot = 3200
    jobs = []
    for K_ in [4, 8, 16, 32, 64]:
        n_ = (Ntot // 2) // K_
        for pol, kw in [('cfp', dict(rho=RHO)), ('epoch', dict(epoch_length=50))]:
            for s in range(max(4, SEEDS // 2)):
                jobs.append((pol, 0.25, s, kw, (K_, n_, Ntot)))
    df = pd.DataFrame(run_jobs(jobs))
    out = agg(df, ['policy', 'K', 'n'])
    out['eq34'] = [p_capture_eq34(n_, Ntot, int(0.25 * Ntot)) for n_ in out.n]
    out.to_csv(f'{OUT}/fig11_shards.csv', index=False)


# ---------------- Fig. 9 + new table: cross-shard decisions, latency, rotation cost ----------------
def _commit_job(args):
    proto, ling, rho, gst, seed = args
    random.seed(seed)
    net = Network(K, n, N)
    rng = random.Random(seed)
    for v in rng.sample(net.validator_pool.validators, int(0.25 * len(net.validator_pool.validators))):
        v.is_adversarial = True
    flu = ValidatorFluidity(net, rho=rho, fast=True, rng=rng)
    if rho == 0:
        flu.shard_count = {sid: 0 for sid in net.shards}
    o = AsyncCommitSimulator(net, flu, seed, protocol=proto, lingering=ling, gst=gst,
                             n_tx=500 if QUICK else 1500).run()
    return dict(protocol=proto, lingering=ling, rho=rho, gst_s=gst / 1000, seed=seed, **o)


def fig9_commit():
    jobs = []
    for gst in [0, 60_000, 120_000]:
        for s in range(3 if QUICK else 10):
            jobs += [('cfp', True, RHO, gst, s), ('cfp', True, 0.0, gst, s), ('cfp', False, RHO, gst, s),
                     ('timeout', True, RHO, gst, s)]
    with Pool(max(1, os.cpu_count())) as p:
        df = pd.DataFrame(p.map(_commit_job, jobs))
    df.to_csv(f'{OUT}/fig9_commit_runs.csv', index=False)
    g = df.groupby(['protocol', 'lingering', 'rho', 'gst_s']).agg(
        cross=('cross', 'sum'), commits=('commits', 'sum'), aborts=('aborts', 'sum'), pending=('pending', 'sum'),
        pending_in_bound=('pending_in_bound', 'sum'), signer_sets_outside_bound=('signer_sets_outside_bound', 'sum'),
        atomicity_violations=('atomicity_violations', 'sum'),
        conservation_error=('conservation_error', lambda v: int(np.abs(v).sum())),
        erased_unrelated=('erased_unrelated', 'sum'), lat_post_p50=('lat_post_p50', 'mean'),
        lat_post_p95=('lat_post_p95', 'mean')).reset_index()
    g.to_csv(f'{OUT}/fig9_commit.csv', index=False)
    # one run with real ECDSA quorum certificates (signatures produced and verified)
    random.seed(99)
    net = Network(4, n, 4 * n * 2)
    rng = random.Random(99)
    for v in rng.sample(net.validator_pool.validators, int(0.25 * len(net.validator_pool.validators))):
        v.is_adversarial = True
    flu = ValidatorFluidity(net, rho=RHO, fast=True, rng=rng)
    o = AsyncCommitSimulator(net, flu, 99, sign=True, n_tx=100 if QUICK else 300, gst=0).run()
    pd.DataFrame([o]).to_csv(f'{OUT}/fig9_signed_run.csv', index=False)
    # Alg. 3 draw cost per shard vs committee size
    from cfp.fluidity import alg3_draw
    from ecdsa import SigningKey, SECP256k1
    sk = SigningKey.generate(curve=SECP256k1); vk = sk.get_verifying_key()
    rows = []
    for K_, n_ in [(4, 25), (4, 50), (16, 50), (16, 100), (16, 200)]:
        rr = math.ceil(RHO * n_)
        t = time.perf_counter()
        for k in range(20):
            alg3_draw('cfp', 0, 'h', k, list(range(n_)), list(range(n_, n_ + K_ * n_)), rr, sk, vk)
        rows.append(dict(K=K_, n=n_, N=K_ * n_ * 2, r=rr, ms_per_shard=(time.perf_counter() - t) / 20 * 1000))
    pd.DataFrame(rows).to_csv(f'{OUT}/fig9_rotation_cost.csv', index=False)


# ---------------- Fig. 10: account partitioning ----------------
def fig10():
    rows = []
    for K_ in [4, 8, 16]:
        for seed in range(2 if QUICK else 5):
            random.seed(seed)
            net = Network(K_, 10, K_ * 20)
            A = K_ * 50
            for i in range(A):
                net.register_account(f"user_{i}")
            p = AccountPartitioner(net)
            for _ in range(A * 10):
                if random.random() < 0.7:
                    b = random.randrange(K_) * 50
                    s, r = f"user_{b + random.randrange(50)}", f"user_{b + random.randrange(50)}"
                else:
                    s, r = f"user_{random.randrange(A)}", f"user_{random.randrange(A)}"
                if s != r:
                    p.record_transaction(s, r)
            before = p.analyze_current_distribution()['cross_ratio']
            lb = shard_load_ratio(p)
            t = time.perf_counter()
            moved, _ = repartition_balanced(p)
            rows.append(dict(K=K_, seed=seed, accounts=A, before=before, after=p.analyze_current_distribution()['cross_ratio'],
                             load_over_budget_before=lb, load_over_budget_after=shard_load_ratio(p),
                             migrated_frac=moved / A, plan_s=time.perf_counter() - t))
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/fig10_partitioning_runs.csv', index=False)
    g = df.groupby('K').mean(numeric_only=True).drop(columns='seed').reset_index()
    g['reduction_pct'] = 100 * (g.before - g.after) / g.before
    g.to_csv(f'{OUT}/fig10_partitioning.csv', index=False)


# ---------------- Fig. 10 (replaced): placement baselines on the Ethereum trace ----------------
_TRACE = {}


def trace():
    if not _TRACE:
        snd, rcv, accts = PL.load_trace()
        H = len(snd) // 2
        G, load = PL.build_graph(snd[:H], rcv[:H], len(accts))
        _TRACE.update(snd=snd, rcv=rcv, accts=accts, H=H, G=G, load=load)
    return _TRACE


def placements(K):
    T = trace(); snd, rcv, accts, H, G, load = T['snd'], T['rcv'], T['accts'], T['H'], T['G'], T['load']
    sh, rh = snd[:H], rcv[:H]
    pi0 = np.array([PL.hash_shard(a, K) for a in accts])
    U = np.full(K, 1.1 * max(load.sum() / K, load.max()))      # budgets must admit the heaviest account
    out, times = {}, {}
    t = time.perf_counter(); out['Hash (Monoxide/OmniLedger/RapidChain)'] = pi0; times['Hash (Monoxide/OmniLedger/RapidChain)'] = 0
    t = time.perf_counter(); out['Jump hash (Tangram)'] = PL.tangram(accts, K); times['Jump hash (Tangram)'] = time.perf_counter() - t
    t = time.perf_counter(); out['CLPA (BrokerChain)'] = PL.clpa(sh, rh, len(accts), K, pi0); times['CLPA (BrokerChain)'] = time.perf_counter() - t
    t = time.perf_counter(); out['Metis'] = PL.metis(G, load, K); times['Metis'] = time.perf_counter() - t
    t = time.perf_counter(); out['Louvain, no budgets'] = PL.louvain_unconstrained(G, K); times['Louvain, no budgets'] = time.perf_counter() - t
    t = time.perf_counter(); pc, st, why, res = PL.plan_cfp(G, load, K, U, pi0, sh, rh)
    out['CFP Alg. 1'] = pc; times['CFP Alg. 1'] = time.perf_counter() - t
    return out, times, dict(status=st, reason=why, resolution=res, U_over_avg=U[0] / (load.sum() / K))


def fig10():
    T = trace(); snd, rcv, H, load = T['snd'], T['rcv'], T['H'], T['load']
    sf, rf = snd[H:], rcv[H:]
    fl = np.bincount(sf, minlength=len(load)) + np.bincount(rf, minlength=len(load))
    rows, meta = [], []
    for K in [8, 16, 32]:
        out, times, m = placements(K)
        meta.append(dict(K=K, **m))
        pi0 = out['Hash (Monoxide/OmniLedger/RapidChain)']
        avg = load.sum() / K
        for name, pi in out.items():
            rows.append(dict(K=K, method=name, cross_history=PL.cut(snd[:H], rcv[:H], pi) / H,
                             cross_future=PL.cut(sf, rf, pi) / len(sf),
                             max_load_over_avg=PL.loads(pi, load, K).max() / avg,
                             future_max_over_avg=PL.loads(pi, fl, K).max() / (fl.sum() / K),
                             moved_from_hash=float((pi != pi0).mean()), plan_s=times[name]))
    pd.DataFrame(rows).to_csv(f'{OUT}/fig10_placement.csv', index=False)
    pd.DataFrame(meta).to_csv(f'{OUT}/fig10_cfp_profile.csv', index=False)
    # shard addition K -> K+1 (the Tangram scenario)
    rows = []
    for K in [8, 16]:
        a, _, _ = placements(K); b, _, _ = placements(K + 1)
        for name in a:
            pa, pb = a[name], b[name]
            cnt = np.bincount(pb, minlength=K + 1)
            rows.append(dict(K=K, method=name, accounts_moved=float((pa != pb).mean()),
                             moved_to_new_shard_only=bool(((pa != pb) & (pb != K)).sum() == 0),
                             cross_future_after=PL.cut(sf, rf, pb) / len(sf),
                             accounts_per_shard_cv=float(cnt.std() / cnt.mean())))
    pd.DataFrame(rows).to_csv(f'{OUT}/fig10_shard_addition.csv', index=False)


# ---------------- New: end-to-end throughput/latency on the Ethereum trace ----------------
def tput():
    T = trace(); snd, rcv, H, load = T['snd'], T['rcv'], T['H'], T['load']
    sf, rf = snd[H:], rcv[H:]
    rows = []
    for K in [8, 16, 32]:
        out, _, _ = placements(K)
        brokers = np.argsort(-load)[:40]
        cases = [('Monoxide (relay)', 'relay', out['Hash (Monoxide/OmniLedger/RapidChain)'], {}),
                 ('OmniLedger (Atomix)', 'atomix', out['Hash (Monoxide/OmniLedger/RapidChain)'], {}),
                 ('RapidChain', 'rapidchain', out['Hash (Monoxide/OmniLedger/RapidChain)'], {}),
                 ('BrokerChain (CLPA + 40 brokers)', 'broker', out['CLPA (BrokerChain)'], dict(brokers=brokers)),
                 ('CFP (Alg. 1 + Alg. 5)', 'cfp', out['CFP Alg. 1'], {}),
                 ('Ablation: CFP commit, hash placement', 'cfp', out['Hash (Monoxide/OmniLedger/RapidChain)'], {}),
                 ('Ablation: relay, CFP placement', 'relay', out['CFP Alg. 1'], {})]
        for rate in [2000, 4000]:
            for name, proto, pi, kw in cases:
                r = TP.simulate(sf, rf, pi, proto, K, rate=rate, **kw)
                rows.append(dict(system=name, **r))
        for tho in [0.0, 0.25, 0.5, 1.0, 2.0]:
            r = TP.simulate(sf, rf, out['CFP Alg. 1'], 'cfp', K, rate=4000, T_ho=tho)
            rows.append(dict(system=f'CFP hand-off {tho} s', **r))
    pd.DataFrame(rows).to_csv(f'{OUT}/tput.csv', index=False)


# ---------------- New: message accounting (sector-based BFT comparison) ----------------
def msgs():
    T = trace(); A = len(T['accts'])
    rows = []
    for n in [25, 50, 100, 200]:
        r = math.ceil(0.2 * n)
        for kind in ['PBFT', 'HotStuff', 'SectorBFT']:
            m, fan = consensus_msgs(n, kind)
            rows.append(dict(n=n, item=f'{kind} consensus', msgs=m, leader_fanout=fan, bytes=np.nan))
        for sync in ['full', 'delta']:
            o = onboarding(n, r, 16, n, A / 16, 2000, sync)
            rows.append(dict(n=n, item=f'CFP hand-off ({sync} checkpoint)', msgs=o['msgs'],
                             leader_fanout=n, bytes=o['bytes'], per_incoming_bytes=o['per_incoming_bytes']))
        rows.append(dict(n=n, item='Block payload (2000 tx x 128 B)', msgs=np.nan, leader_fanout=np.nan, bytes=2000 * 128))
    pd.DataFrame(rows).to_csv(f'{OUT}/msgs.csv', index=False)


ALL = dict(fig3=fig3, fig4=fig4, fig5_8=fig5_8, fig8_tau=fig8_tau, fig6=fig6, fig7_12=fig7_12, fig9=fig9_commit,
           fig10=fig10, fig11=fig11, fig13=fig13, tput=tput, msgs=msgs)

if __name__ == '__main__':
    sel = [a for a in sys.argv[1:] if not a.startswith('--')] or list(ALL)
    for k in sel:
        t = time.time()
        ALL[k]()
        print(f'{k} done in {time.time() - t:.0f}s', flush=True)
