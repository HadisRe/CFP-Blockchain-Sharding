"""
Per-block, per-shard message and byte accounting from the protocol specifications (answers R2.1).

Steady-state consensus (one finalized block, n replicas):
  PBFT        pre-prepare n + prepare n^2 + commit n^2                 leader fan-out n
  HotStuff    7 communication steps, leader-to-all and all-to-leader   7n, leader fan-out n   [25, Table I]
  Sector BFT  leader -> x = sqrt(n) sector replicas, each relays to n/x others, n votes:
              x + n + n messages; leader fan-out sqrt(n)              [25, Sec. VII-B]
CFP membership change per block (Alg. 3), r incoming validators:
  predecessor certificate: q partial signatures to the collector
  successor hand-off: C_p and records to the n successors; the r incoming ones also need the checkpoint X_h
  readiness certificate: q signatures; ACTIVATE: 1 message to the CC, whose finalization is shared by
  the K shards activated in the same CC block (HotStuff-style CC of size n_cc: 7 n_cc / K per shard)
Bytes: BLS signature 96 B, certificate 96 B + n/8-byte bitmap, transaction/record 128 B (payload size of [25]).
Checkpoint X_h: 'full' = all account records of the shard; 'delta' = the records written by the last block,
which is sufficient only if standby validators keep a synchronized follower copy of the shard state.
"""
import math

SIG, REC = 96, 128


def consensus_msgs(n, kind):
    if kind == 'PBFT':
        return n + 2 * n * n, n
    if kind == 'HotStuff':
        return 7 * n, n
    if kind == 'SectorBFT':
        x = max(1, round(math.sqrt(n)))
        return x + n + n, x
    raise ValueError(kind)


def onboarding(n, r, K, n_cc, accounts_per_shard, block_records, sync='full'):
    q = n - (math.ceil(n / 3) - 1)
    msgs = q + n + q + 1 + 7 * n_cc / K
    cert = SIG + n / 8
    ckpt = accounts_per_shard * REC if sync == 'full' else block_records * REC
    bytes_ = q * SIG + n * cert + r * ckpt + q * SIG + 2 * cert
    return dict(msgs=msgs, bytes=bytes_, per_incoming_bytes=ckpt + cert)
