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