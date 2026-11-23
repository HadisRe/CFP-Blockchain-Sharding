"""
Cross-Shard Transaction Protocol for CFP simulation.
Implements snapshot-anchored two-phase commit from paper section 4.6.

Uses equations (16) to (21) from the paper:
- Eq 16: Pinned signer set Signers_i = V_i(h_i)
- Eq 17: Prepare message with partial signatures
- Eq 18: Threshold requirement |signatures| >= 2n/3 + 1
- Eq 19: Prepare certificate PC_i = (Tx, h_i, snap_i, Sigma_i)
- Eq 20: Commit certificate C_commit = {PC_a, PC_b, ..., PC_m}
- Eq 21: Abort and rollback mechanism

Uses real ECDSA signatures from validators for multi-signature scheme.
"""

import time
import hashlib
import random


class CrossShardTransaction:
    """
    A transaction that spans multiple shards.
    Needs coordination between shards to ensure atomicity.
    """

    def __init__(self, tx_id, sender, receiver, amount, sender_shard, receiver_shard):
        self.tx_id = tx_id
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.sender_shard = sender_shard
        self.receiver_shard = receiver_shard
        self.timestamp = time.time()

        # status tracking
        self.status = "pending"  # pending, prepared, committed, aborted
        self.prepare_votes = {}  # shard_id -> True/False
        self.snapshots = {}  # shard_id -> Snapshot object
        self.prepare_certificates = {}  # shard_id -> PrepareCertificate

    def get_message_to_sign(self):
        """
        Create the message that validators will sign.
        This ensures all validators sign the same thing.
        """
        message = f"CROSS_TX|{self.tx_id}|{self.sender}|{self.receiver}|{self.amount}|{self.sender_shard}|{self.receiver_shard}"
        return message

    def __str__(self):
        return f"CrossTX[{self.tx_id[:8]}]: {self.sender}(S{self.sender_shard}) -> {self.receiver}(S{self.receiver_shard}) : {self.amount}"


class Snapshot:
    """
    State snapshot for a shard at specific height.
    From paper equation (16): Signers_i = V_i(h_i)

    The snapshot pins the validator set at a specific block height.
    This prevents attacks where validators change during the protocol.
    """

    def __init__(self, shard, height=None):
        if height is None:
            height = shard.get_chain_height() - 1

        block = shard.blockchain.get_block_by_number(height)

        self.shard_id = shard.shard_id
        self.height = height
        self.block_hash = block.block_hash if block else "genesis"

        # Pin the validator set - this is equation (16)
        # We store the actual validator objects so we can verify signatures
        self.pinned_validators = list(shard.validators)
        self.validator_ids = [v.validator_id for v in self.pinned_validators]
        self.validator_public_keys = {v.validator_id: v.public_key for v in self.pinned_validators}

        self.timestamp = time.time()

    def get_signer_set(self):
        """
        Get pinned signer set for this snapshot.
        From paper equation (16): Signers_i = V_i(h_i)
        """
        return self.pinned_validators.copy()

    def get_validator_count(self):
        """Number of validators in pinned set."""
        return len(self.pinned_validators)

    def get_threshold(self):
        """
        Calculate threshold for this snapshot.
        From paper equation (18): threshold = 2n/3 + 1
        """
        n = len(self.pinned_validators)
        return (2 * n // 3) + 1

    def is_valid_signer(self, validator_id):
        """Check if validator was in the pinned set."""
        return validator_id in self.validator_ids

    def get_public_key(self, validator_id):
        """Get public key for a validator in the pinned set."""
        return self.validator_public_keys.get(validator_id)

    def get_hash(self):
        """Get hash of snapshot for signing."""
        data = f"{self.shard_id}|{self.height}|{self.block_hash}|{','.join(sorted(self.validator_ids))}"
        return hashlib.sha256(data.encode()).hexdigest()

    def to_dict(self):
        return {
            "shard_id": self.shard_id,
            "height": self.height,
            "block_hash": self.block_hash,
            "validator_count": len(self.pinned_validators),
            "threshold": self.get_threshold()
        }


class ValidatorSignature:
    """
    A single validator's signature on a message.
    Contains everything needed to verify the signature.
    """

    def __init__(self, validator_id, public_key, message, signature_bytes):
        self.validator_id = validator_id
        self.public_key = public_key
        self.message = message
        self.signature_bytes = signature_bytes
        self.timestamp = time.time()

    def verify(self):
        """
        Verify this signature is valid.
        Uses the validator's public key to check.
        """
        try:
            import hashlib
            self.public_key.verify(
                self.signature_bytes,
                self.message.encode() if isinstance(self.message, str) else self.message,
                hashfunc=hashlib.sha256
            )
            return True
        except:
            return False


class MultiSignature:
    """
    Collection of signatures from multiple validators.
    Used instead of threshold signature for simplicity.

    Tracks:
    - All signatures received
    - Which are valid
    - Whether threshold is met
    """

    def __init__(self, message, threshold):
        self.message = message
        self.threshold = threshold
        self.signatures = {}  # validator_id -> ValidatorSignature
        self.valid_signatures = set()  # validator_ids with valid signatures

    def add_signature(self, validator_id, public_key, signature_bytes):
        """
        Add a signature from a validator.
        Immediately verifies it.
        """
        sig = ValidatorSignature(validator_id, public_key, self.message, signature_bytes)
        self.signatures[validator_id] = sig

        if sig.verify():
            self.valid_signatures.add(validator_id)
            return True
        return False

    def get_valid_count(self):
        """How many valid signatures do we have."""
        return len(self.valid_signatures)

    def has_threshold(self):
        """Check if we have enough valid signatures."""
        return self.get_valid_count() >= self.threshold

    def get_signers(self):
        """Get list of validators who signed validly."""
        return list(self.valid_signatures)

    def verify_all(self):
        """Re-verify all signatures and return count of valid ones."""
        valid = 0
        for sig in self.signatures.values():
            if sig.verify():
                valid += 1
        return valid


class PrepareMessage:
    """
    Prepare message from a validator.
    From paper equation (17).

    Each validator sends their vote (yes/no) and signature.
    """

    def __init__(self, shard_id, tx_id, validator_id, vote, signature):
        self.shard_id = shard_id
        self.tx_id = tx_id
        self.validator_id = validator_id
        self.vote = vote  # True = prepared, False = rejected
        self.signature = signature  # ValidatorSignature object
        self.timestamp = time.time()


class PrepareCertificate:
    """
    Certificate proving shard prepared a transaction.
    From paper equation (19): PC_i = (Tx, h_i, snap_i(h_i), Sigma_i)

    Contains:
    - Transaction ID
    - Snapshot (pinned validator set)
    - Multi-signature from validators
    """

    def __init__(self, tx_id, shard_id, snapshot, multi_signature):
        self.tx_id = tx_id
        self.shard_id = shard_id
        self.snapshot = snapshot
        self.multi_signature = multi_signature
        self.timestamp = time.time()

    def is_valid(self):
        """
        Verify this certificate is valid.
        Checks that multi-signature has enough valid signatures.
        """
        if self.multi_signature is None:
            return False
        return self.multi_signature.has_threshold()

    def get_signer_count(self):
        """How many validators signed."""
        return self.multi_signature.get_valid_count() if self.multi_signature else 0

    def get_threshold(self):
        """Required threshold."""
        return self.snapshot.get_threshold()

    def to_dict(self):
        return {
            "tx_id": self.tx_id[:8],
            "shard_id": self.shard_id,
            "height": self.snapshot.height,
            "signers": self.get_signer_count(),
            "threshold": self.get_threshold(),
            "valid": self.is_valid()
        }


class CommitCertificate:
    """
    Final commit certificate for cross-shard transaction.
    From paper equation (20): C_commit = {PC_a, PC_b, ..., PC_m}

    Contains prepare certificates from ALL involved shards.
    Only created when all shards have prepared successfully.
    """

    def __init__(self, tx_id, prepare_certificates):
        self.tx_id = tx_id
        self.prepare_certificates = prepare_certificates  # dict: shard_id -> PrepareCertificate
        self.timestamp = time.time()

    def is_complete(self, expected_shards):
        """Check if we have valid certificates from all shards."""
        for shard_id in expected_shards:
            if shard_id not in self.prepare_certificates:
                return False
            if not self.prepare_certificates[shard_id].is_valid():
                return False
        return True

    def verify(self, expected_shards):
        """
        Full verification of commit certificate.
        Checks all prepare certificates are valid.
        """
        if not self.is_complete(expected_shards):
            return False

        for pc in self.prepare_certificates.values():
            if not pc.is_valid():
                return False

        return True

    def to_dict(self):
        return {
            "tx_id": self.tx_id[:8],
            "shards": list(self.prepare_certificates.keys()),
            "all_valid": all(pc.is_valid() for pc in self.prepare_certificates.values())
        }


class AbortCertificate:
    """
    Abort certificate when transaction fails.
    From paper equation (21): triggers rollback on all shards.

    Created when:
    - A shard rejects (not enough votes)
    - Timeout occurs
    - Invalid state detected
    """

    def __init__(self, tx_id, reason, failed_shard=None, details=None):
        self.tx_id = tx_id
        self.reason = reason
        self.failed_shard = failed_shard
        self.details = details or {}
        self.timestamp = time.time()

    def to_dict(self):
        return {
            "tx_id": self.tx_id[:8],
            "reason": self.reason,
            "failed_shard": self.failed_shard
        }


class CrossShardCoordinator:
    """
    Coordinates cross-shard transactions.
    Manages the two-phase commit protocol with real signatures.
    """

    def __init__(self, network, threshold_ratio=0.67, timeout=10):
        """
        Setup coordinator.

        network: the Network object
        threshold_ratio: fraction of validators needed (default 2/3)
        timeout: max time for responses (seconds)
        """
        self.network = network
        self.threshold_ratio = threshold_ratio
        self.timeout = timeout

        self.pending_txs = {}  # tx_id -> CrossShardTransaction
        self.completed_txs = {}  # tx_id -> (status, certificate)
        self.stats = {
            "total": 0,
            "committed": 0,
            "aborted": 0,
            "signature_failures": 0
        }

    def create_cross_shard_tx(self, sender, receiver, amount):
        """Create a new cross-shard transaction."""
        sender_shard = self.network.get_account_shard(sender)
        receiver_shard = self.network.get_account_shard(receiver)

        if sender_shard is None or receiver_shard is None:
            return None

        if sender_shard == receiver_shard:
            return None

        tx_id = hashlib.sha256(
            f"{sender}{receiver}{amount}{time.time()}{random.random()}".encode()
        ).hexdigest()

        tx = CrossShardTransaction(
            tx_id, sender, receiver, amount,
            sender_shard, receiver_shard
        )

        self.pending_txs[tx_id] = tx
        self.stats["total"] += 1

        return tx

    def take_snapshot(self, tx, shard_id):
        """
        Take snapshot from a shard.
        From paper equation (16): pins the validator set.
        """
        shard = self.network.shards[shard_id]
        snapshot = Snapshot(shard)
        tx.snapshots[shard_id] = snapshot
        return snapshot

    def collect_prepare_votes(self, tx, shard_id):
        """
        Collect prepare votes from validators in a shard.

        Each validator:
        1. Checks if they can prepare (validate transaction)
        2. Signs the transaction message with their private key
        3. Sends their vote and signature

        Returns a MultiSignature object.
        """
        snapshot = tx.snapshots.get(shard_id)
        if snapshot is None:
            return None

        message = tx.get_message_to_sign()
        threshold = snapshot.get_threshold()

        multi_sig = MultiSignature(message, threshold)

        # Collect votes from pinned validators
        for validator in snapshot.get_signer_set():
            # Decide vote
            if validator.is_adversarial and random.random() < 0.3:
                # Adversarial validator might reject
                vote = False
            else:
                vote = True

            if vote:
                # Sign the message with real ECDSA signature
                signature_bytes = validator.sign(message)

                # Add to multi-signature
                success = multi_sig.add_signature(
                    validator.validator_id,
                    validator.public_key,
                    signature_bytes
                )

                if not success:
                    self.stats["signature_failures"] += 1

        return multi_sig

    def create_prepare_certificate(self, tx, shard_id, multi_sig):
        """
        Create prepare certificate if threshold is met.
        From paper equation (19): PC_i = (Tx, h_i, snap_i, Sigma_i)
        """
        snapshot = tx.snapshots[shard_id]

        if not multi_sig.has_threshold():
            return None

        cert = PrepareCertificate(tx.tx_id, shard_id, snapshot, multi_sig)
        tx.prepare_certificates[shard_id] = cert

        return cert

    def execute_two_phase_commit(self, tx):
        """
        Execute full two-phase commit protocol.

        Phase 0: Take snapshots (pin validator sets)
        Phase 1: Prepare (collect signatures from each shard)
        Phase 2: Decide (commit if all prepared, abort otherwise)

        Returns CommitCertificate or AbortCertificate.
        """
        involved_shards = [tx.sender_shard, tx.receiver_shard]

        # ============ Phase 0: Snapshot ============
        # Pin the validator set for each shard
        for shard_id in involved_shards:
            self.take_snapshot(tx, shard_id)

        # ============ Phase 1: Prepare ============
        # Collect votes and signatures from each shard
        all_prepared = True
        failed_shard = None

        for shard_id in involved_shards:
            # Collect signatures
            multi_sig = self.collect_prepare_votes(tx, shard_id)

            if multi_sig is None:
                all_prepared = False
                failed_shard = shard_id
                tx.prepare_votes[shard_id] = False
                break

            # Check if threshold met
            if not multi_sig.has_threshold():
                all_prepared = False
                failed_shard = shard_id
                tx.prepare_votes[shard_id] = False
                break

            # Create prepare certificate
            cert = self.create_prepare_certificate(tx, shard_id, multi_sig)

            if cert is None or not cert.is_valid():
                all_prepared = False
                failed_shard = shard_id
                tx.prepare_votes[shard_id] = False
                break

            tx.prepare_votes[shard_id] = True

        # ============ Phase 2: Decision ============
        if all_prepared:
            # All shards prepared successfully - COMMIT
            commit_cert = CommitCertificate(tx.tx_id, tx.prepare_certificates)

            # Verify the commit certificate
            if commit_cert.verify(involved_shards):
                tx.status = "committed"
                self.stats["committed"] += 1
                self.completed_txs[tx.tx_id] = ("committed", commit_cert)
                return commit_cert
            else:
                # This shouldn't happen but handle it
                abort_cert = AbortCertificate(
                    tx.tx_id,
                    "Commit verification failed",
                    details={"shards": involved_shards}
                )
                tx.status = "aborted"
                self.stats["aborted"] += 1
                self.completed_txs[tx.tx_id] = ("aborted", abort_cert)
                return abort_cert
        else:
            # Someone failed - ABORT
            abort_cert = AbortCertificate(
                tx.tx_id,
                "Prepare failed - threshold not met",
                failed_shard=failed_shard,
                details={
                    "votes": tx.prepare_votes.copy()
                }
            )
            tx.status = "aborted"
            self.stats["aborted"] += 1
            self.completed_txs[tx.tx_id] = ("aborted", abort_cert)
            return abort_cert

    def process_transaction(self, sender, receiver, amount):
        """
        Full process: create tx and execute 2PC.
        """
        tx = self.create_cross_shard_tx(sender, receiver, amount)

        if tx is None:
            return None, "Not cross-shard or accounts not found"

        result = self.execute_two_phase_commit(tx)
        return tx, result

    def verify_commit_certificate(self, commit_cert, expected_shards):
        """
        Independently verify a commit certificate.
        Anyone can call this to check the certificate is valid.

        Verifies:
        1. All expected shards have prepare certificates
        2. Each prepare certificate has valid signatures
        3. Each prepare certificate meets threshold
        """
        return commit_cert.verify(expected_shards)

    def get_success_rate(self):
        """Get transaction success rate."""
        if self.stats["total"] == 0:
            return 0
        return self.stats["committed"] / self.stats["total"]

    def print_stats(self):
        """Print coordinator statistics."""
        print(f"\nCross-Shard Coordinator Stats:")
        print(f"  Total transactions: {self.stats['total']}")
        print(f"  Committed: {self.stats['committed']}")
        print(f"  Aborted: {self.stats['aborted']}")
        print(f"  Signature failures: {self.stats['signature_failures']}")
        print(f"  Success rate: {self.get_success_rate():.1%}")


# Test code
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')

    from sharding import Network

    print("Testing Cross-Shard Protocol with Real Signatures")
    print("=" * 55)

    # Setup
    net = Network(num_shards=4, validators_per_shard=25, total_validators=200)

    # Register accounts
    accounts = [f"user_{i}" for i in range(100)]
    for acc in accounts:
        net.register_account(acc)

    # Create coordinator
    coord = CrossShardCoordinator(net)

    # Test single transaction in detail
    print("\n[1] Testing single transaction with signature verification:")
    print("-" * 50)

    # Find a cross-shard pair
    sender, receiver = None, None
    for i in range(100):
        s = f"user_{i}"
        r = f"user_{(i+50) % 100}"
        if net.is_cross_shard(s, r):
            sender, receiver = s, r
            break

    if sender and receiver:
        tx = coord.create_cross_shard_tx(sender, receiver, 100)
        print(f"  Transaction: {tx}")

        # Take snapshots
        coord.take_snapshot(tx, tx.sender_shard)
        coord.take_snapshot(tx, tx.receiver_shard)

        snapshot = tx.snapshots[tx.sender_shard]
        print(f"  Snapshot shard {tx.sender_shard}: {snapshot.get_validator_count()} validators")
        print(f"  Threshold: {snapshot.get_threshold()} signatures needed")

        # Collect signatures
        multi_sig = coord.collect_prepare_votes(tx, tx.sender_shard)
        print(f"  Signatures collected: {multi_sig.get_valid_count()}")
        print(f"  Threshold met: {multi_sig.has_threshold()}")

        # Execute full 2PC
        result = coord.execute_two_phase_commit(tx)
        print(f"  Result: {tx.status}")

        if isinstance(result, CommitCertificate):
            print(f"  Commit certificate valid: {result.verify([tx.sender_shard, tx.receiver_shard])}")

    # Test batch transactions
    print("\n[2] Testing batch transactions:")
    print("-" * 50)

    # Fresh coordinator
    coord2 = CrossShardCoordinator(net)

    for i in range(20):
        s = random.choice(accounts)
        r = random.choice([a for a in accounts if a != s])

        if net.is_cross_shard(s, r):
            tx, result = coord2.process_transaction(s, r, random.randint(10, 100))
            if tx:
                print(f"  TX {i+1}: {tx.status}")

    coord2.print_stats()

    # Test with adversaries
    print("\n[3] Testing with 30% adversarial validators:")
    print("-" * 50)

    net.validator_pool.set_adversarial_fraction(0.3)
    coord3 = CrossShardCoordinator(net)

    committed = 0
    total = 0

    for i in range(30):
        s = random.choice(accounts)
        r = random.choice([a for a in accounts if a != s])

        if net.is_cross_shard(s, r):
            tx, result = coord3.process_transaction(s, r, 50)
            if tx:
                total += 1
                if tx.status == "committed":
                    committed += 1

    print(f"  With 30% adversaries: {committed}/{total} committed ({100*committed/total:.1f}%)")

    print("\nCross-shard protocol test complete!")

# =====================================================================
# Revised cross-shard protocol (Sec. IV-F, Alg. 5) and the original timeout 2PC,
# both run on the same Network under partial synchrony.
# =====================================================================

import heapq as _heapq
import math as _math


class AsyncCommitSimulator:
    """
    Discrete-event simulation (time in ms) of cross-shard transfers on a Network whose
    committees rotate every block (Eq. 14).

    protocol='cfp'      Alg. 5: each participant shard finalizes exactly one local vote per
                        transaction (PREPARE or ABORT) in its own ordered ledger; a quorum
                        certificate from the pinned signer set certifies that vote; COMMIT needs
                        PREPARE certificates from all participants, one ABORT certificate decides
                        ABORT; prepared accounts are locked with tentative writes, and abort removes
                        only that transaction's writes (Eq. 30). Recovery re-requests votes and
                        relays certificates with doubling backoff capped at backoff_cap.
    protocol='timeout'  Original submission: a coordinator collects prepares; when the lease expires
                        before it has seen all of them it broadcasts ABORT; abort restores the shard
                        state to the transaction snapshot (Eq. 21 of the first version).

    Network: before GST each message is lost with prob loss_pre and delayed Exp(mean_pre); after GST
    delays are U(0, Delta). Pinned signers are the committee at admission; adversarial pinned signers
    withhold; honest ones are unavailable with prob delta per attempt. With lingering=True rotated-out
    pinned signers keep signing until the lease ends; with lingering=False they stop.
    sign=True produces and verifies real ECDSA quorum certificates with the validators' keys.
    """

    def __init__(self, network, fluidity, seed, protocol='cfp', lingering=True, sign=False,
                 accounts_per_shard=200, Tb=1000, gst=60_000, mean_pre=3000, loss_pre=0.2, Delta=200,
                 delta=0.05, lease=8000, T_agg=1500, T_reprobe=2000, backoff_cap=32_000,
                 horizon=1_500_000, n_tx=1000, cross_frac=0.3, p_coord_fail=0.1, arrival_window=300_000):
        self.net, self.flu = network, fluidity
        self.rng = random.Random(seed)
        self.protocol, self.lingering, self.sign = protocol, lingering, sign
        self.p = dict(Tb=Tb, gst=gst, mean_pre=mean_pre, loss_pre=loss_pre, Delta=Delta, delta=delta,
                      lease=lease, T_agg=T_agg, T_reprobe=T_reprobe, backoff_cap=backoff_cap,
                      horizon=horizon, p_coord_fail=p_coord_fail)
        self.K = len(network.shards)
        self.n = network.validators_per_shard
        self.q = self.n - _math.ceil(self.n / 3) + 1
        self.ev, self.seq, self.now, self.last_block = [], 0, 0, 0
        A = accounts_per_shard
        self.owner = [k for k in range(self.K) for _ in range(A)]
        self.bal = [self.rng.randint(50, 200) for _ in range(self.K * A)]
        self.total0 = sum(self.bal)
        self.locks, self.tent, self.vote, self.cert, self.applied, self.decided = {}, {}, {}, {}, {}, {}
        self.txs = {}
        self.shard_log = [[] for _ in range(self.K)]
        self.st = dict(retries=0, erased_unrelated=0, latency=[], lat_post=[], certs_signed=0,
                       certs_verified=0, cert_failures=0)
        self.n_tx, self.cross_frac, self.arrival_window = n_tx, cross_frac, arrival_window

    # ---------- engine ----------
    def at(self, t, fn, *a):
        self.seq += 1
        _heapq.heappush(self.ev, (t, self.seq, fn, a))

    def send(self, fn, *a):
        p = self.p
        if self.now < p['gst']:
            if self.rng.random() < p['loss_pre']:
                return
            d = self.rng.expovariate(1 / p['mean_pre'])
        else:
            d = self.rng.uniform(0, p['Delta'])
        self.at(self.now + d, fn, *a)

    def order_local(self, shard, fn, *a):
        Tb = self.p['Tb']
        t = (_math.floor(self.now / Tb) + 1) * Tb
        if self.now < self.p['gst'] and self.rng.random() < 0.3:
            t += Tb * self.rng.randint(1, 3)
        self.at(t, fn, shard, *a)

    def _grow(self, t):
        c = self.p['backoff_cap']
        return 2 * t if c is None else min(2 * t, max(c, t))

    def _advance_blocks(self):
        Tb = self.p['Tb']
        while self.last_block + Tb <= self.now:
            self.last_block += Tb
            self.flu.rotate_all_shards(block_hash='0' * 16)

    # ---------- workload ----------
    def generate(self):
        A = len(self.bal)
        for k in range(self.n_tx):
            t = self.rng.uniform(0, self.arrival_window)
            src = self.rng.randrange(A)
            if self.rng.random() < self.cross_frac:
                dst = self.rng.choice([a for a in range(A) if self.owner[a] != self.owner[src]])
            else:
                dst = self.rng.choice([a for a in range(A) if self.owner[a] == self.owner[src] and a != src])
            self.at(t, self.submit, k, src, dst, self.rng.randint(1, 59))

    def submit(self, tid, src, dst, amt):
        parts = sorted({self.owner[src], self.owner[dst]})
        tx = dict(id=tid, src=src, dst=dst, amt=amt, parts=parts, t0=self.now, pinned={}, done=set())
        for s in parts:   # snapshot pins the committee at admission (Eqs. 21-22)
            tx['pinned'][s] = list(self.net.shards[s].validators)
            if sum(v.is_adversarial for v in tx['pinned'][s]) > _math.ceil(self.n / 3) - 1:
                tx['exceed'] = True
        self.txs[tid] = tx
        if len(parts) == 1:
            self.order_local(parts[0], self.local_tx, tid)
            return
        tx['coord_alive'] = self.rng.random() >= self.p['p_coord_fail']
        for s in parts:
            self.send(self.on_prepare_req, s, tid)
        self.at(self.now + self.p['lease'], self.on_lease, tid)
        self.at(self.now + self.p['T_reprobe'], self.recover, tid, self.p['T_reprobe'])

    def local_tx(self, shard, tid):
        tx = self.txs[tid]
        s, d, a = tx['src'], tx['dst'], tx['amt']
        if s in self.locks or d in self.locks or self.bal[s] < a:
            self.applied[(shard, tid)] = 'A'
            return
        self.bal[s] -= a
        self.bal[d] += a
        self.applied[(shard, tid)] = 'C'
        self.shard_log[shard].append((self.now, tid))

    # ---------- local votes ----------
    def on_prepare_req(self, shard, tid):
        self.order_local(shard, self.fin_vote, tid, 'PREPARE')

    def fin_vote(self, shard, tid, kind):
        key = (shard, tid)
        if key in self.vote or key in self.applied:
            return
        tx = self.txs[tid]
        if kind == 'ABORT':
            self.vote[key] = 'A'
        else:
            mine = [x for x in (tx['src'], tx['dst']) if self.owner[x] == shard]
            ok = all(x not in self.locks for x in mine)
            if ok and self.owner[tx['src']] == shard and self.bal[tx['src']] < tx['amt']:
                ok = False
            if ok:
                for x in mine:
                    self.locks[x] = tid
                tx.setdefault('lock_t', {})[shard] = self.now
                w = []
                if self.owner[tx['src']] == shard:
                    w.append((tx['src'], -tx['amt']))
                if self.owner[tx['dst']] == shard:
                    w.append((tx['dst'], tx['amt']))
                self.tent[key] = w
                self.vote[key] = 'P'
            else:
                self.vote[key] = 'A'
        self.collect(shard, tid, self.p['T_agg'])

    def _available_signers(self, shard, tid):
        tx = self.txs[tid]
        active = {v.validator_id for v in self.net.shards[shard].validators}
        out = []
        for v in tx['pinned'][shard]:
            if v.is_adversarial:
                continue
            if not self.lingering and v.validator_id not in active:
                continue
            if self.rng.random() < self.p['delta']:
                continue
            if self.now < self.p['gst'] and self.rng.random() < 0.3:
                continue
            out.append(v)
        return out

    def collect(self, shard, tid, timeout):
        key = (shard, tid)
        if key in self.cert:
            return
        signers = self._available_signers(shard, tid)
        tx = self.txs[tid]
        if self.now >= self.p['gst'] and not tx.get('exceed') and (shard, tid) not in tx.setdefault('first_attempt', set()):
            tx['first_attempt'].add((shard, tid))
            self.st.setdefault('first_attempts', 0)
            self.st['first_attempts'] += 1
            self.st.setdefault('first_deficits', 0)
            self.st['first_deficits'] += len(signers) < self.q
        if len(signers) >= self.q:
            if self.sign:
                msg = f"{tid}|{shard}|{self.vote[key]}".encode()
                ms = MultiSignature(msg, self.q)
                for v in signers[:self.q]:
                    ms.add_signature(v.validator_id, v.public_key, v.sign(msg))
                self.st['certs_signed'] += 1
                if not (ms.has_threshold() and ms.verify_all()):
                    self.st['cert_failures'] += 1
                    return
                self.st['certs_verified'] += 1
            self.cert[key] = self.vote[key]
            if self.protocol == 'cfp' or self.txs[tid]['coord_alive']:
                self.send(self.on_cert, tid, shard)
        else:
            self.st['retries'] += 1
            tx['retried'] = True
            self.at(self.now + timeout, self.collect, shard, tid, self._grow(timeout))

    # ---------- outcome ----------
    def on_cert(self, tid, shard):
        tx = self.txs[tid]
        certs = [self.cert.get((s, tid)) for s in tx['parts']]
        if self.protocol == 'cfp':
            out = 'A' if 'A' in certs else 'C' if all(c == 'P' for c in certs) else None
            if out is None:
                return
            prev = self.decided.setdefault(tid, out)
            assert prev == out, 'conflicting certified outcomes'
            for s in tx['parts']:
                self.send(self.on_outcome, s, tid, out)
        elif tx['coord_alive'] and tid not in self.decided:
            if 'A' in certs:
                self._decide_timeout(tid, 'A')
            elif all(c == 'P' for c in certs):
                self._decide_timeout(tid, 'C')

    def _decide_timeout(self, tid, out):
        self.decided[tid] = out
        for s in self.txs[tid]['parts']:
            self.send(self.on_outcome, s, tid, out)

    def on_lease(self, tid):
        tx = self.txs[tid]
        if self.protocol == 'cfp':
            for s in tx['parts']:
                if (s, tid) not in self.vote:
                    self.order_local(s, self.fin_vote, tid, 'ABORT')
        else:
            certs = [self.cert.get((s, tid)) for s in tx['parts']]
            if not (tx['coord_alive'] and all(c == 'P' for c in certs) and tid in self.decided):
                for s in tx['parts']:
                    self.send(self.on_outcome, s, tid, 'A')

    def recover(self, tid, interval):
        tx = self.txs[tid]
        if all((s, tid) in self.applied for s in tx['parts']):
            return
        if self.protocol == 'cfp' or tx['coord_alive']:
            for s in tx['parts']:
                if (s, tid) not in self.vote:
                    self.send(self.on_prepare_req, s, tid)
                elif (s, tid) in self.cert:
                    self.send(self.on_cert, tid, s)
            if tid in self.decided:
                for s in tx['parts']:
                    if (s, tid) not in self.applied:
                        self.send(self.on_outcome, s, tid, self.decided[tid])
        self.at(self.now + interval, self.recover, tid, self._grow(interval))

    def on_outcome(self, shard, tid, out):
        self.order_local(shard, self.apply, tid, out)

    def apply(self, shard, tid, out):
        key = (shard, tid)
        if key in self.applied:
            return
        tx = self.txs[tid]
        if out == 'C':
            if key not in self.tent:
                return
            for acct, dl in self.tent.pop(key):
                self.bal[acct] += dl
            self.shard_log[shard].append((self.now, tid))
        else:
            self.tent.pop(key, None)
            if self.protocol != 'cfp' and shard in tx.get('lock_t', {}):
                t_snap = tx['lock_t'][shard]
                self.st['erased_unrelated'] += sum(1 for (t, o) in self.shard_log[shard]
                                                   if t > t_snap and o != tid)
        for acct, owner in list(self.locks.items()):
            if owner == tid and self.owner[acct] == shard:
                del self.locks[acct]
        self.applied[key] = out
        if all((s, tid) in self.applied for s in tx['parts']):
            self.st['latency'].append(self.now - tx['t0'])
            if tx['t0'] >= self.p['gst'] and not tx.get('exceed'):
                self.st['lat_post'].append(self.now - tx['t0'])
                if not tx.get('retried') and len(tx['parts']) == 2:
                    self.st.setdefault('lat_first_try', []).append(self.now - tx['t0'])

    # ---------- run ----------
    def run(self):
        self.generate()
        while self.ev:
            t, _, fn, a = _heapq.heappop(self.ev)
            if t > self.p['horizon']:
                break
            self.now = t
            self._advance_blocks()
            fn(*a)
        cross = [t for t in self.txs.values() if len(t['parts']) == 2]
        res = dict(cross=len(cross), commits=0, aborts=0, pending=0, pending_in_bound=0,
                   atomicity_violations=0)
        for tx in cross:
            outs = [self.applied.get((s, tx['id'])) for s in tx['parts']]
            got = [o for o in outs if o]
            if 'C' in got and 'A' in got:
                res['atomicity_violations'] += 1
            if len(got) < len(outs):
                res['pending'] += 1
                res['pending_in_bound'] += not tx.get('exceed')
            elif got[0] == 'C':
                res['commits'] += 1
            else:
                res['aborts'] += 1
        res['signer_sets_outside_bound'] = sum(1 for tx in cross if tx.get('exceed'))
        res['conservation_error'] = sum(self.bal) - self.total0
        res['erased_unrelated'] = self.st['erased_unrelated']
        lp = sorted(self.st['lat_post'])
        res['lat_post_p50'] = lp[len(lp) // 2] if lp else float('nan')
        res['lat_post_p95'] = lp[int(0.95 * (len(lp) - 1))] if lp else float('nan')
        res['first_attempts'] = self.st.get('first_attempts', 0)
        res['first_deficits'] = self.st.get('first_deficits', 0)
        ft = self.st.get('lat_first_try', [])
        bound = 3 * self.p['Delta'] + 2 * self.p['Tb']
        res['eq44_bound_ms'] = bound
        res['first_try_n'] = len(ft)
        res['first_try_within_eq44'] = sum(1 for x in ft if x <= bound)
        res['first_try_max_ms'] = max(ft) if ft else float('nan')
        res['certs_verified'] = self.st['certs_verified']
        res['cert_failures'] = self.st['cert_failures']
        return res
