"""
Block-level end-to-end throughput/latency simulation (answers R2.4), run on the same Ethereum trace,
shard count, block interval, block capacity, injection rate and network delay for every protocol.

Each shard produces a block every T_b seconds (CFP: T_b + T_ho, the per-block committee hand-off time of
Alg. 3) holding at most B commands that are ready at block start (FIFO). A transaction is confirmed when
its last command is included. Cross-shard handling, following each protocol's specification:

  relay      Monoxide [23] / BlockEmulator relay: debit at source; relay transaction enqueued at the
             destination Delta after the source block (eventual atomicity).                2 commands
  broker     BrokerChain [16] (BlockEmulator broker module): payer->broker at source, broker->payee at
             destination Delta after confirmation; brokers hold balance segments in every shard, so a
             transfer with a broker endpoint is intra-shard.                              2 commands
  atomix     OmniLedger [20]: client-driven lock at the input shard; the client obtains the proof (Delta)
             and submits unlock-to-commit to the output shard (Delta).                    2 commands
  rapidchain RapidChain [22]: requests travel over the inter-committee routing overlay
             (ceil(log2 K) hops of Delta each) to the output committee, which creates the input
             sub-transaction at the source; the output transaction is added after the source commits.
                                                                                         2 commands
  cfp        CFP Alg. 5: PREPARE vote finalized at every participant, certificates assembled (Delta),
             outcome applied at every participant.                                       4 commands
Epoch reconfiguration pauses of the baselines are not charged (favours the baselines).
"""
import heapq, math
import numpy as np


def simulate(snd, rcv, pi, protocol, K, T_b=5.0, B=2000, rate=2000.0, Delta=0.1, T_ho=0.0,
             brokers=None, horizon_extra=600.0):
    n_tx = len(snd)
    arr = np.arange(n_tx) / rate
    queues = [[] for _ in range(K)]          # heaps of (ready_time, seq, tx, step)
    seq = 0
    src = pi[snd]; dst = pi[rcv]
    is_broker = np.zeros(int(max(snd.max(), rcv.max())) + 1, bool)
    if brokers is not None:
        is_broker[brokers] = True
    hops = max(1, math.ceil(math.log2(K))) if K > 1 else 1
    need = np.zeros(n_tx, np.int8)          # outstanding commands of the current step
    confirm = np.full(n_tx, np.nan)
    cross = np.zeros(n_tx, bool)

    def push(s, t, i, step):
        nonlocal seq
        seq += 1
        heapq.heappush(queues[s], (t, seq, i, step))

    for i in range(n_tx):
        a, b = src[i], dst[i]
        intra = a == b or (protocol == 'broker' and (is_broker[snd[i]] or is_broker[rcv[i]]))
        if intra:
            push(a if a == b else (a if is_broker[rcv[i]] else b), arr[i], i, 'local')
            need[i] = 1
            continue
        cross[i] = True
        if protocol in ('relay', 'broker'):
            push(a, arr[i], i, 'src'); need[i] = 1
        elif protocol == 'atomix':
            push(a, arr[i] + Delta, i, 'src'); need[i] = 1
        elif protocol == 'rapidchain':
            push(a, arr[i] + 2 * hops * Delta, i, 'src'); need[i] = 1     # to output committee and back
        elif protocol == 'cfp':
            push(a, arr[i] + Delta, i, 'prep'); push(b, arr[i] + Delta, i, 'prep'); need[i] = 2
        else:
            raise ValueError(protocol)

    period = T_b + (T_ho if protocol == 'cfp' else 0.0)
    t = period
    t_end = arr[-1] + horizon_extra
    cfp_done = np.zeros(n_tx, np.int8)
    while t <= t_end:
        for s in range(K):
            q = queues[s]; taken = 0
            while q and taken < B and q[0][0] <= t:
                _, _, i, step = heapq.heappop(q); taken += 1
                if step == 'local':
                    confirm[i] = t
                elif step == 'src':
                    d = dst[i]
                    if protocol in ('relay', 'broker'):
                        push(d, t + Delta, i, 'dst')
                    elif protocol == 'atomix':
                        push(d, t + 2 * Delta, i, 'dst')
                    else:                                 # rapidchain: output committee commits
                        push(d, t + hops * Delta, i, 'dst')
                elif step == 'dst':
                    confirm[i] = t
                elif step == 'prep':
                    need[i] -= 1
                    if need[i] == 0:                      # both PREPARE votes certified -> outcome
                        ready = t + Delta
                        push(src[i], ready, i, 'apply'); push(dst[i], ready, i, 'apply'); need[i] = 2
                elif step == 'apply':
                    cfp_done[i] += 1
                    if cfp_done[i] == 2:
                        confirm[i] = t
        t += period
        if not any(queues) and t > arr[-1]:
            break
    ok = ~np.isnan(confirm)
    lat = confirm[ok] - arr[ok]
    span = np.nanmax(confirm) - arr[0] if ok.any() else np.nan
    return dict(protocol=protocol, K=K, rate=rate, T_ho=T_ho, confirmed=int(ok.sum()), total=n_tx,
                tps=float(ok.sum() / span), lat_mean=float(lat.mean()), lat_p95=float(np.percentile(lat, 95)),
                cross_ratio=float(cross.mean()))
