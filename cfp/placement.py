"""
Account placement on a real Ethereum trace (BlockEmulator selectedTxs_300K.csv, MIT licence).

Policies compared under the same shard count K and the same history/future windows:
  hash        Monoxide / BlockEmulator default (last 8 hex digits of the address mod K),
              also the placement used by OmniLedger-/RapidChain-style random assignment
  clpa        BrokerChain's constrained label propagation, ported from BlockEmulator
              partition/partition_CLPA.go (beta = 0.5, tau = 100 iterations, start = hash)
  metis       balanced Metis partition (pymetis), the offline reference used by BrokerChain/OptChain
  louvain     weighted Louvain communities mapped to shards without budgets
  cfp         Algorithm 1: weighted Louvain + capacity-constrained greedy placement (Eq. 9)
              with the acceptance rule of lines 24-29
"""
import csv, time
from collections import defaultdict
import numpy as np
import networkx as nx


def load_trace(path='data/selectedTxs_300K.csv'):
    fr, to = [], []
    for row in csv.reader(open(path)):
        a, b = row[3], row[4]
        if len(a) == 42 and len(b) == 42 and a != b:
            fr.append(a); to.append(b)
    accts = sorted(set(fr) | set(to))
    idx = {a: i for i, a in enumerate(accts)}
    return np.array([idx[a] for a in fr]), np.array([idx[b] for b in to]), accts


def hash_shard(addr, K):
    return int(addr[-8:], 16) % K


def build_graph(snd, rcv, A):
    """BUILDGRAPH: w(A,B) = number of transactions between A and B in either direction."""
    a, b = np.minimum(snd, rcv), np.maximum(snd, rcv)
    key = a.astype(np.int64) * A + b
    u, c = np.unique(key, return_counts=True)
    G = nx.Graph(); G.add_nodes_from(range(A))
    G.add_weighted_edges_from(zip((u // A).tolist(), (u % A).tolist(), c.tolist()))
    load = np.bincount(snd, minlength=A) + np.bincount(rcv, minlength=A)   # l_A (Eq. 8)
    return G, load


def cut(snd, rcv, pi):
    return int((pi[snd] != pi[rcv]).sum())


def loads(pi, load, K):
    return np.bincount(pi, weights=load, minlength=K)


# ---------------- Algorithm 1 ----------------
RESOLUTIONS = (1, 3, 10, 30)


def plan_cfp(G, load, K, U, pi0, snd, rcv, seed=0, resolutions=RESOLUTIONS):
    """Algorithm 1 with a fixed planning profile: the Louvain resolution is chosen from a fixed grid by
    the smallest historical cut among authorized candidates (planning inputs only; the evaluation window
    is never used). Returns (pi, status, reason, resolution)."""
    best = None
    for res in resolutions:
        pi, st, why = _plan_once(G, load, K, U, pi0, snd, rcv, seed, res)
        if st == 'AUTHORIZED' and (best is None or cut(snd, rcv, pi) < best[4]):
            best = (pi, st, why, res, cut(snd, rcv, pi))
    if best is None:
        return pi0.copy(), 'NOPLAN', why, None
    return best[:4]


def _plan_once(G, load, K, U, pi0, snd, rcv, seed, resolution):
    if G.number_of_edges() == 0:
        return pi0.copy(), 'NOPLAN', 'no cut opportunity'
    comms = nx.community.louvain_communities(G, weight='weight', seed=seed, resolution=resolution)
    units = [sorted(c) for c in comms]
    order = lambda m: (-int(load[m].sum()), m[0])          # decreasing workload, then smallest id
    queue = sorted(units, key=order)
    A = len(load); pi = np.full(A, -1); L = np.zeros(K)
    adj = G.adj
    while queue:
        C = queue.pop(0)
        lC = load[C].sum()
        elig = [j for j in range(K) if L[j] + lC <= U[j]]
        if not elig:
            if len(C) == 1:
                return pi0.copy(), 'NOPLAN', 'placement failed'
            queue = sorted(queue + [[v] for v in C], key=order)
            continue
        cost = np.zeros(K)
        for v in C:
            for u, d in adj[v].items():
                if pi[u] >= 0:
                    cost += d['weight']; cost[pi[u]] -= d['weight']
        j = min(elig, key=lambda j: (cost[j], (L[j] + lC) / U[j], j))
        pi[C] = j; L[j] += lC
    feasible0 = bool((loads(pi0, load, K) <= U).all())
    if feasible0:
        if cut(snd, rcv, pi) >= cut(snd, rcv, pi0):
            return pi0.copy(), 'NOPLAN', 'no strict improvement'
        return pi, 'AUTHORIZED', 'cut improvement'
    return pi, 'AUTHORIZED', 'feasibility restoration'


# ---------------- BrokerChain CLPA (port of partition_CLPA.go) ----------------
def clpa(snd, rcv, A, K, pi_init, beta=0.5, tau=100, max_moves=50):
    """Faithful port: unweighted multigraph edges (one per transaction, both directions stored),
    Edges2Shard W_k, score = e(v,k)/deg(v) * (1 - beta * W_k / min_k W_k), vertices visited in a fixed
    order, a vertex moves at most max_moves times, a shard never empties."""
    nbrs = [[] for _ in range(A)]
    for a, b in zip(snd.tolist(), rcv.tolist()):
        nbrs[a].append(b); nbrs[b].append(a)
    pi = pi_init.copy()
    cnt = np.bincount(pi, minlength=K)

    def edges2shard():
        W = np.zeros(K, dtype=np.int64); inter = np.zeros(K, dtype=np.int64)
        for v in range(A):
            sv = pi[v]
            for u in nbrs[v]:
                if pi[u] != sv:
                    W[pi[u]] += 1
                else:
                    inter[pi[u]] += 1
        return W + inter // 2
    W = edges2shard()
    moves = np.zeros(A, dtype=np.int32)
    for _ in range(tau):
        changed = 0
        for v in range(A):
            if moves[v] >= max_moves or not nbrs[v]:
                continue
            Wmin = W.min() if W.min() > 0 else 1
            deg = len(nbrs[v])
            per = defaultdict(int)
            for u in nbrs[v]:
                per[pi[u]] += 1
            best, bs = pi[v], -9999.0
            for k, e in per.items():
                sc = e / deg * (1 - beta * W[k] / Wmin)
                if sc > bs:
                    bs, best = sc, k
            old = pi[v]
            if best != old and cnt[old] > 1:
                pi[v] = best; moves[v] += 1; changed += 1
                cnt[old] -= 1; cnt[best] += 1
                for u in nbrs[v]:                       # changeShardRecompute
                    s = pi[u]
                    if s != best and s != old:
                        W[best] += 1; W[old] -= 1
                    elif s == best:
                        W[old] -= 1
                    else:
                        W[best] += 1
        if changed == 0:
            break
    return pi


def metis(G, load, K, seed=0, ufactor=100):
    import pymetis
    A = G.number_of_nodes()
    xadj, adjncy, eweights = [0], [], []
    for v in range(A):
        for u, d in G.adj[v].items():
            adjncy.append(u); eweights.append(int(d['weight']))
        xadj.append(len(adjncy))
    opts = pymetis.Options(seed=seed, ufactor=ufactor)      # ufactor 100 = 10% imbalance, matches eps
    _, parts = pymetis.part_graph(K, xadj=xadj, adjncy=adjncy, eweights=eweights,
                                  vweights=[max(1, int(x)) for x in load], options=opts)
    return np.array(parts)


def louvain_unconstrained(G, K, seed=0):
    comms = sorted(nx.community.louvain_communities(G, weight='weight', seed=seed), key=lambda c: -len(c))
    pi = np.empty(G.number_of_nodes(), int)
    for k, c in enumerate(comms):
        pi[list(c)] = k % K
    return pi


def jump_hash(key, buckets):
    """Jump Consistent Hash (Lamping and Veach), the placement used by Tangram [27]."""
    key &= 0xFFFFFFFFFFFFFFFF
    b, j = -1, 0
    while j < buckets:
        b = j
        key = (key * 2862933555777941757 + 1) & 0xFFFFFFFFFFFFFFFF
        j = int((b + 1) * (float(1 << 31) / float((key >> 33) + 1)))
    return b


def tangram(accts, K):
    return np.array([jump_hash(int(a[-16:], 16), K) for a in accts])
