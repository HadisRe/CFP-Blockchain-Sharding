"""
Closed-form quantities of Section V, implemented with exact integer binomials.

g(a; u, v, w)      Eq. (31)
P_i, P             Eq. (32)
mu                 Eq. (33)
p_static           Eq. (34)
p_seq, nu, Pr(D>=L), E[min(D,H)], uncensored mean   Eqs. (35)-(37)
endpoint overlap   Eq. (38)
p_epoch            Eq. (39)
p_ctrl, p_live, p_joint  Eqs. (45)-(49)
"""
from math import comb, ceil
from fractions import Fraction
import numpy as np


def g(a, u, v, w):
    if a < 0 or a > v or w - a < 0 or w - a > u - v or w > u:
        return 0.0
    return comb(v, a) * comb(u - v, w - a) / comb(u, w)


def b_cap(n):
    return ceil(n / 3)


def p_static(n, N, M):
    return sum(g(j, N, M, n) for j in range(b_cap(n), n + 1))


def kernel_K1(n, s, M, r):
    """Eq. (32) for K = 1: state = adversaries in the committee; standby holds M - x."""
    P = np.zeros((n + 1, n + 1))
    for x in range(n + 1):
        z = M - x
        if z < 0 or z > s:
            continue
        for d in range(r + 1):
            pd = g(d, n, x, r)
            if pd == 0:
                continue
            for a in range(r + 1):
                pa = g(a, s, z, r)
                if pa:
                    P[x, x - d + a] += pd * pa
    return P


def mu_K1(n, s, M):
    """Eq. (33) for K = 1."""
    N = n + s
    return np.array([comb(n, x) * comb(s, M - x) / comb(N, M) if 0 <= M - x <= s else 0.0 for x in range(n + 1)])


def kernel_K2(n, s, M, r):
    """Joint kernel P = P_1 P_2 for two shards sharing a standby pool, and mu of Eq. (33)."""
    states = [(a, b) for a in range(n + 1) for b in range(n + 1) if 0 <= M - a - b <= s]
    idx = {st: i for i, st in enumerate(states)}

    def Pi(c):
        P = np.zeros((len(states), len(states)))
        for (a, b), i in idx.items():
            x = (a, b)[c]
            z = M - a - b
            for d in range(r + 1):
                pd = g(d, n, x, r)
                if pd == 0:
                    continue
                for ad in range(r + 1):
                    pa = g(ad, s, z, r)
                    if pa:
                        y = x - d + ad
                        P[i, idx[(y, b) if c == 0 else (a, y)]] += pd * pa
        return P
    N = 2 * n + s
    mu = np.array([comb(n, a) * comb(n, b) * comb(s, M - a - b) / comb(N, M) for a, b in states])
    return Pi(0) @ Pi(1), mu, np.array([a for a, _ in states])


def episode_quantities(P, mu, captured, Lmax=60, H=None):
    """Eqs. (35)-(37): stationary p_seq(L), entry distribution nu, Pr(D >= L), E[min(D,H)], uncensored mean."""
    C = captured
    Q = P[np.ix_(C, C)]
    one = np.ones(Q.shape[0])
    alpha = mu[C]
    flow = mu[~C] @ P[np.ix_(~C, C)]
    nu = flow / flow.sum()
    pseq, surv = [], []
    va, vn = alpha.copy(), nu.copy()
    for _ in range(Lmax):
        pseq.append(va @ one); surv.append(vn @ one)
        va, vn = va @ Q, vn @ Q
    out = dict(p_capture=float(alpha.sum()), pseq=np.array(pseq), survival=np.array(surv),
               rho_Q=float(max(abs(np.linalg.eigvals(Q)))),
               mean=float(nu @ np.linalg.solve(np.eye(len(one)) - Q, one)))
    if H is not None:
        out['mean_min_H'] = float(sum(out['survival'][:H])) if H <= Lmax else None
    return out


def endpoint_overlap(n, N, r, k):
    """Eq. (38), isolated committee."""
    return n * n / N + n * (1 - n / N) * (1 - r / n - r / (N - n)) ** k


def p_epoch(pstat, a, L, E):
    """Eq. (39)."""
    J = 1 + (a + L - 1) // E
    return pstat ** J


def pin_bounds(n, N, M, delta):
    """Eqs. (22), (45), (47): b_pin, q_pin, p_ctrl, p_live for one pinned population (uniform sampling)."""
    b = ceil(n / 3) - 1
    q = n - b
    p_ctrl = sum(g(j, N, M, n) for j in range(b + 1, n + 1))
    p_live = 0.0
    for j in range(0, b + 1):
        pj = g(j, N, M, n)
        for h in range(0, min(q - 1, n - j) + 1):
            p_live += pj * comb(n - j, h) * (1 - delta) ** h * delta ** (n - j - h)
    return dict(b_pin=b, q_pin=q, p_ctrl=p_ctrl, p_live=p_live)
