"""
Attack Simulation module for CFP.
Implements Equilibrium Capture attack from paper Section 5.

Equilibrium Capture Attack:
- Adversary controls a fixed fraction of validators
- In static system: adversary's validators stay in place
- In CFP system: rotation disrupts adversary's control

Key metrics:
- Capture Duration: how long adversary maintains control
- Capture Probability: chance of getting majority in a shard
- Recovery Time: how fast system recovers after capture
"""

import random
import math
import math
from collections import defaultdict
import copy


def capture_count(n):
    """Eq. (6): a committee of size n is captured when the adversary holds at least ceil(n/3) seats."""
    return math.ceil(n / 3)


def is_captured(control_fraction, n=None, threshold=None):
    """Capture test. threshold=None uses Eq. (6); a numeric threshold reproduces the legacy fraction rule."""
    if threshold is not None:
        return control_fraction >= threshold
    if n is None:
        n = 25
    return control_fraction * n >= capture_count(n) - 1e-9



class AdversaryStrategy:
    """
    Models adversary behavior.

    The adversary:
    1. Controls a fixed set of validators
    2. Tries to concentrate them in one shard
    3. Votes in coordinated way to manipulate CFI
    """

    def __init__(self, controlled_validators):
        """
        Setup adversary with list of validators they control.
        These validators are marked adversarial and stay that way.
        """
        self.controlled_validators = set(controlled_validators)
        self.target_shard = None

        # Mark all controlled validators
        for v in controlled_validators:
            v.is_adversarial = True

    def get_controlled_count(self):
        """How many validators does adversary control."""
        return len(self.controlled_validators)

    def get_validators_in_shard(self, shard):
        """Count how many controlled validators are in a shard."""
        count = 0
        for v in shard.validators:
            if v in self.controlled_validators:
                count += 1
        return count

    def get_control_fraction(self, shard):
        """What fraction of shard does adversary control."""
        if len(shard.validators) == 0:
            return 0
        return self.get_validators_in_shard(shard) / len(shard.validators)

    def has_majority(self, shard, threshold=None):
        """Does adversary have majority control of shard."""
        n = len(shard.validators)
        return is_captured(self.get_control_fraction(shard), n, threshold)

    def coordinate_votes(self, shard, block_hash):
        """
        Make all controlled validators in shard vote the same way.
        This is how adversary tries to manipulate CFI.
        """
        for v in shard.validators:
            if v in self.controlled_validators:
                # Always vote True (coordinated)
                v.cast_vote(block_hash, True)
            else:
                # Honest validators vote randomly
                v.cast_vote(block_hash, random.choice([True, False]))


class StaticCommitteeSimulator:
    """
    Simulates attack on static committee (no rotation).

    In static system:
    - Validators are assigned once and never move
    - If adversary gets majority, they keep it forever
    """

    def __init__(self, network):
        self.network = network
        self.adversary = None

    def setup_adversary(self, fraction):
        """
        Setup adversary controlling a fraction of validators.
        Validators are randomly selected and stay adversarial.
        """
        total = len(self.network.validator_pool.validators)
        count = int(total * fraction)

        # Randomly select which validators adversary controls
        controlled = random.sample(self.network.validator_pool.validators, count)
        self.adversary = AdversaryStrategy(controlled)

        return self.adversary

    def simulate(self, num_blocks, threshold=None):
        """
        Simulate static system for num_blocks.

        In static system, validator assignment is fixed.
        We run multiple trials with different random assignments
        to show the RISK of static committees.

        Returns history of adversary control in each shard.
        """
        history = {
            "blocks": [],
            "shard_control": defaultdict(list),  # shard_id -> [fractions]
            "any_captured": [],  # was any shard captured this block
            "captured_shards": []  # which shards were captured
        }

        # In static system, the assignment is fixed for all blocks
        # But the RISK is that a bad assignment locks in adversary control

        for block_num in range(num_blocks):
            history["blocks"].append(block_num)

            captured_this_block = []

            for shard_id, shard in self.network.shards.items():
                # Calculate adversary control (fixed in static)
                control = self.adversary.get_control_fraction(shard)
                history["shard_control"][shard_id].append(control)

                if is_captured(control, len(shard.validators), threshold):
                    captured_this_block.append(shard_id)

                # Adversary coordinates votes
                self.adversary.coordinate_votes(shard, f"block_{block_num}")

            history["any_captured"].append(len(captured_this_block) > 0)
            history["captured_shards"].append(captured_this_block)

        return history

    def analyze_results(self, history, threshold=None):
        """Analyze simulation results."""
        total_blocks = len(history["blocks"])

        # How many blocks had at least one captured shard
        blocks_with_capture = sum(history["any_captured"])

        # Per-shard analysis
        shard_stats = {}
        for shard_id, controls in history["shard_control"].items():
            captured_blocks = sum(1 for c in controls if is_captured(c, self.network.validators_per_shard, threshold))
            avg_control = sum(controls) / len(controls)
            max_control = max(controls)

            shard_stats[shard_id] = {
                "captured_blocks": captured_blocks,
                "capture_rate": captured_blocks / total_blocks,
                "avg_control": avg_control,
                "max_control": max_control
            }

        return {
            "total_blocks": total_blocks,
            "blocks_with_capture": blocks_with_capture,
            "overall_capture_rate": blocks_with_capture / total_blocks,
            "shard_stats": shard_stats
        }


class CFPCommitteeSimulator:
    """
    Simulates attack on CFP system (with rotation).

    In CFP system:
    - Validators rotate every block based on rho
    - Adversary's control gets diluted over time
    - System can detect coordinated voting via CFI
    """

    def __init__(self, network, fluidity_manager, cfi_monitor=None):
        self.network = network
        self.fluidity = fluidity_manager
        self.cfi_monitor = cfi_monitor
        self.adversary = None

    def setup_adversary(self, fraction):
        """Setup adversary controlling a fraction of validators."""
        total = len(self.network.validator_pool.validators)
        count = int(total * fraction)

        controlled = random.sample(self.network.validator_pool.validators, count)
        self.adversary = AdversaryStrategy(controlled)

        return self.adversary

    def simulate(self, num_blocks, threshold=None):
        """
        Simulate CFP system for num_blocks.

        Each block:
        1. Rotate validators (CFP mechanism)
        2. Adversary coordinates votes
        3. Check CFI and possibly increase rho
        4. Record state
        """
        history = {
            "blocks": [],
            "shard_control": defaultdict(list),
            "any_captured": [],
            "captured_shards": [],
            "rho_values": [],
            "cfi_values": defaultdict(list)
        }

        for block_num in range(num_blocks):
            history["blocks"].append(block_num)
            history["rho_values"].append(self.fluidity.rho)

            # Step 1: Rotate validators (this is the CFP mechanism)
            self.fluidity.rotate_all_shards()

            captured_this_block = []

            for shard_id, shard in self.network.shards.items():
                # Calculate adversary control after rotation
                control = self.adversary.get_control_fraction(shard)
                history["shard_control"][shard_id].append(control)

                if is_captured(control, len(shard.validators), threshold):
                    captured_this_block.append(shard_id)

                # Adversary coordinates votes
                self.adversary.coordinate_votes(shard, f"block_{block_num}")

                # Calculate CFI if monitor available
                if self.cfi_monitor:
                    cfi = self.cfi_monitor.calculate_cfi(shard)
                    history["cfi_values"][shard_id].append(cfi)

                    # CFI response: increase rho if CFI is low
                    if cfi < self.cfi_monitor.threshold:
                        self.fluidity.increase_rho(0.02)

            history["any_captured"].append(len(captured_this_block) > 0)
            history["captured_shards"].append(captured_this_block)

        return history

    def analyze_results(self, history, threshold=None):
        """Analyze simulation results."""
        total_blocks = len(history["blocks"])

        blocks_with_capture = sum(history["any_captured"])

        # Per-shard analysis
        shard_stats = {}
        for shard_id, controls in history["shard_control"].items():
            captured_blocks = sum(1 for c in controls if is_captured(c, self.network.validators_per_shard, threshold))
            avg_control = sum(controls) / len(controls)
            max_control = max(controls)

            shard_stats[shard_id] = {
                "captured_blocks": captured_blocks,
                "capture_rate": captured_blocks / total_blocks,
                "avg_control": avg_control,
                "max_control": max_control
            }

        # Rho analysis
        initial_rho = history["rho_values"][0]
        final_rho = history["rho_values"][-1]

        # CFI analysis
        cfi_stats = {}
        for shard_id, cfis in history["cfi_values"].items():
            if cfis:
                cfi_stats[shard_id] = {
                    "avg_cfi": sum(cfis) / len(cfis),
                    "min_cfi": min(cfis)
                }

        return {
            "total_blocks": total_blocks,
            "blocks_with_capture": blocks_with_capture,
            "overall_capture_rate": blocks_with_capture / total_blocks,
            "shard_stats": shard_stats,
            "initial_rho": initial_rho,
            "final_rho": final_rho,
            "cfi_stats": cfi_stats
        }


class EquilibriumCaptureExperiment:
    """
    Main experiment comparing Static vs CFP under attack.

    Runs both simulations with same adversary fraction
    and compares results.
    """

    def __init__(self):
        self.results = {}

    def run_experiment(self, num_shards, validators_per_shard, total_validators,
                       adversary_fraction, num_blocks, rho=0.2, threshold=0.5):
        """
        Run complete experiment.

        Creates two identical networks, one static, one CFP.
        Runs attack simulation on both.

        KEY INSIGHT:
        - Static: if adversary gets lucky with initial assignment, they keep control forever
        - CFP: rotation means adversary can't maintain control even if they get it
        """
        from sharding import Network
        from cfp.fluidity import ValidatorFluidity
        from cfp.cfi import ConsensusFrictionIndex

        # ========== Run multiple trials to show Static risk ==========
        static_capture_trials = 0
        num_trials = 20

        for trial in range(num_trials):
            static_net = Network(
                num_shards=num_shards,
                validators_per_shard=validators_per_shard,
                total_validators=total_validators
            )

            static_sim = StaticCommitteeSimulator(static_net)
            static_sim.setup_adversary(adversary_fraction)

            # Check if ANY shard is captured with this random assignment
            for shard in static_net.shards.values():
                if static_sim.adversary.has_majority(shard, threshold):
                    static_capture_trials += 1
                    break

        static_risk = static_capture_trials / num_trials

        # ========== Static Network (single trial for detailed analysis) ==========
        static_net = Network(
            num_shards=num_shards,
            validators_per_shard=validators_per_shard,
            total_validators=total_validators
        )

        static_sim = StaticCommitteeSimulator(static_net)
        static_sim.setup_adversary(adversary_fraction)

        static_history = static_sim.simulate(num_blocks, threshold)
        static_results = static_sim.analyze_results(static_history, threshold)
        static_results["risk_across_trials"] = static_risk

        # ========== CFP Network ==========
        cfp_net = Network(
            num_shards=num_shards,
            validators_per_shard=validators_per_shard,
            total_validators=total_validators
        )

        fluidity = ValidatorFluidity(cfp_net, rho=rho)
        cfi_monitor = ConsensusFrictionIndex(fluidity, threshold=0.5, window=10)

        cfp_sim = CFPCommitteeSimulator(cfp_net, fluidity, cfi_monitor)
        cfp_sim.setup_adversary(adversary_fraction)

        cfp_history = cfp_sim.simulate(num_blocks, threshold)
        cfp_results = cfp_sim.analyze_results(cfp_history, threshold)

        # ========== Comparison ==========
        self.results = {
            "parameters": {
                "num_shards": num_shards,
                "validators_per_shard": validators_per_shard,
                "total_validators": total_validators,
                "adversary_fraction": adversary_fraction,
                "num_blocks": num_blocks,
                "rho": rho,
                "threshold": threshold
            },
            "static": static_results,
            "cfp": cfp_results,
            "static_history": static_history,
            "cfp_history": cfp_history
        }

        return self.results

    def calculate_improvement(self):
        """Calculate how much CFP improves over static."""
        if not self.results:
            return None

        static_rate = self.results["static"]["overall_capture_rate"]
        cfp_rate = self.results["cfp"]["overall_capture_rate"]

        if static_rate == 0:
            improvement = 0 if cfp_rate == 0 else -100
        else:
            improvement = (static_rate - cfp_rate) / static_rate * 100

        return {
            "static_capture_rate": static_rate,
            "cfp_capture_rate": cfp_rate,
            "improvement_percent": improvement
        }

    def print_results(self):
        """Print formatted results."""
        if not self.results:
            print("No results yet. Run experiment first.")
            return

        params = self.results["parameters"]
        static = self.results["static"]
        cfp = self.results["cfp"]

        print("\n" + "=" * 60)
        print("EQUILIBRIUM CAPTURE ATTACK EXPERIMENT")
        print("=" * 60)

        print(f"\nParameters:")
        print(f"  Shards: {params['num_shards']}")
        print(f"  Validators per shard: {params['validators_per_shard']}")
        print(f"  Total validators: {params['total_validators']}")
        print(f"  Adversary fraction: {params['adversary_fraction']:.0%}")
        print(f"  Simulation blocks: {params['num_blocks']}")
        print(f"  Capture threshold: {params['threshold']:.0%}")

        print(f"\n" + "-" * 40)
        print("STATIC COMMITTEE RESULTS:")
        print(f"  Risk of capture (across 20 trials): {static.get('risk_across_trials', 0):.1%}")
        print(f"  This trial capture rate: {static['overall_capture_rate']:.1%}")
        print(f"  NOTE: If captured in static, adversary keeps control FOREVER")

        print(f"\n  Per-shard breakdown:")
        for shard_id, stats in static["shard_stats"].items():
            print(f"    Shard {shard_id}: control {stats['avg_control']:.1%}")

        print(f"\n" + "-" * 40)
        print("CFP COMMITTEE RESULTS:")
        print(f"  Blocks with capture: {cfp['blocks_with_capture']}/{cfp['total_blocks']}")
        print(f"  Capture rate: {cfp['overall_capture_rate']:.1%}")
        print(f"  Rho: {cfp['initial_rho']:.0%} -> {cfp['final_rho']:.0%}")
        print(f"  NOTE: Even when captured, rotation quickly recovers")

        print(f"\n  Per-shard breakdown:")
        for shard_id, stats in cfp["shard_stats"].items():
            cfi_info = ""
            if shard_id in cfp.get("cfi_stats", {}):
                cfi_info = f", avg CFI {cfp['cfi_stats'][shard_id]['avg_cfi']:.2f}"
            print(f"    Shard {shard_id}: avg control {stats['avg_control']:.1%}{cfi_info}")

        improvement = self.calculate_improvement()
        print(f"\n" + "-" * 40)
        print("COMPARISON:")
        print(f"  Static risk (probabilistic): {static.get('risk_across_trials', 0):.1%}")
        print(f"  CFP capture rate: {improvement['cfp_capture_rate']:.1%}")

        # The real comparison: Static risk vs CFP transient captures
        static_risk = static.get('risk_across_trials', 0)
        cfp_rate = improvement['cfp_capture_rate']

        if static_risk > 0:
            real_improvement = (static_risk - cfp_rate) / static_risk * 100
        else:
            real_improvement = 0 if cfp_rate == 0 else -100

        print(f"  Security improvement: {real_improvement:.1f}%")
        print(f"\n  Key insight:")
        print(f"    - Static: {static_risk:.1%} chance of PERMANENT capture")
        print(f"    - CFP: {cfp_rate:.1%} of blocks have TRANSIENT capture")

        print("=" * 60)


def theoretical_capture_probability(f, n, t):
    """
    Calculate theoretical probability of capturing a shard.

    P(capture) = sum over k>=t of C(n,k) * f^k * (1-f)^(n-k)

    f: adversarial fraction
    n: validators per shard
    t: threshold (number of validators needed for majority)
    """
    p = 0
    for k in range(t, n + 1):
        coef = math.comb(n, k)
        p += coef * (f ** k) * ((1 - f) ** (n - k))
    return p


# Backward compatibility
class AttackSimulator:
    """Wrapper for backward compatibility with test_phase6.py"""

    def __init__(self, network, fluidity_manager=None):
        self.network = network
        self.fluidity = fluidity_manager

    def simulate_static_attack(self, adversarial_fraction, num_rounds=100, threshold=0.5):
        static_sim = StaticCommitteeSimulator(self.network)
        static_sim.setup_adversary(adversarial_fraction)
        history = static_sim.simulate(num_rounds, threshold)
        results = static_sim.analyze_results(history, threshold)

        return {
            "type": "static",
            "adversarial_fraction": adversarial_fraction,
            "threshold": threshold,
            "compromised_rounds": results["blocks_with_capture"],
            "total_rounds": num_rounds,
            "success_rate": results["overall_capture_rate"]
        }

    def simulate_cfp_attack(self, adversarial_fraction, num_rounds=100, threshold=0.5):
        if self.fluidity is None:
            return {"error": "No fluidity manager"}

        from cfp.cfi import ConsensusFrictionIndex
        cfi_monitor = ConsensusFrictionIndex(self.fluidity, threshold=0.5, window=10)

        cfp_sim = CFPCommitteeSimulator(self.network, self.fluidity, cfi_monitor)
        cfp_sim.setup_adversary(adversarial_fraction)
        history = cfp_sim.simulate(num_rounds, threshold)
        results = cfp_sim.analyze_results(history, threshold)

        return {
            "type": "cfp",
            "adversarial_fraction": adversarial_fraction,
            "threshold": threshold,
            "rho": self.fluidity.rho,
            "avg_compromised_shards": sum(results["shard_stats"][s]["avg_control"]
                                          for s in results["shard_stats"]) / len(results["shard_stats"]),
            "rounds_with_compromise": results["blocks_with_capture"],
            "total_rounds": num_rounds,
            "success_rate": results["overall_capture_rate"]
        }

    def theoretical_capture_probability(self, f, n, t, num_shards):
        p_single = theoretical_capture_probability(f, n, t)
        p_any = 1 - ((1 - p_single) ** num_shards)

        return {
            "p_single_shard": p_single,
            "p_any_shard": p_any
        }


# For backward compatibility
EquilibriumCaptureSimulator = EquilibriumCaptureExperiment

# Test code
if __name__ == "__main__":
    print("Testing Equilibrium Capture Attack Simulation")
    print("=" * 60)

    # Run experiment
    experiment = EquilibriumCaptureExperiment()

    results = experiment.run_experiment(
        num_shards=4,
        validators_per_shard=25,
        total_validators=200,
        adversary_fraction=0.3,
        num_blocks=100,
        rho=0.2,
        threshold=0.5
    )

    experiment.print_results()

    # Theoretical comparison
    print("\nTheoretical Analysis:")
    p = theoretical_capture_probability(0.3, 25, 13)
    print(f"  P(capture single shard with 30% adversary): {p:.4f}")
    print(f"  P(capture any of 4 shards): {1 - (1 - p) ** 4:.4f}")

    print("\nAttack simulation test complete!")

# =====================================================================
# Revised security experiments (Sec. V-A benchmark, Sec. VI)
# =====================================================================

class RotationStrategyExperiment:
    """
    Runs one membership policy on a Network and records the adversarial seat count of
    every shard after every block. Policies share the same population of N validators
    (K active committees + standby pool) and the same adversarial identity set, so the
    comparison is like-for-like (Sec. V-A3):

      static  - assignment fixed for the whole run
      epoch   - every E blocks all K*n seats are redrawn uniformly from all N validators
      aecfr   - alpha-ECFR-style partial reshuffle [24]: each active validator is selected
                with probability alpha and the selected ones are permuted over the vacated seats
      cfp     - Eq. (14): r validators leave, r join from the standby pool, every block
    """

    def __init__(self, num_shards, validators_per_shard, total_validators,
                 adversary_fraction, seed):
        import numpy as np
        self.np = np
        self.seed = seed
        random.seed(seed)
        from sharding import Network
        self.net = Network(num_shards, validators_per_shard, total_validators)
        self.rng = random.Random(seed + 7919)
        pool = self.net.validator_pool.validators
        for v in pool:
            v.is_adversarial = False
        for v in self.rng.sample(pool, int(round(adversary_fraction * len(pool)))):
            v.is_adversarial = True
        self.K, self.n = num_shards, validators_per_shard

    def _counts(self):
        return [sum(v.is_adversarial for v in sh.validators) for sh in self.net.shards.values()]

    def _redraw_all(self):
        pool = self.net.validator_pool
        vals = list(pool.validators)
        self.rng.shuffle(vals)
        pool.assigned = set()
        for i, sh in enumerate(self.net.shards.values()):
            sh.validators = vals[i * self.n:(i + 1) * self.n]
            for v in sh.validators:
                v.current_shard = sh.shard_id
                pool.mark_assigned(v)
        return vals[self.K * self.n:]

    def run(self, policy, num_blocks, rho=0.2, epoch_length=10, alpha=0.2, fluidity=None):
        np = self.np
        out = np.zeros((num_blocks + 1, self.K), dtype=np.int16)
        out[0] = self._counts()
        if policy == 'cfp' and fluidity is None:
            from cfp.fluidity import ValidatorFluidity
            fluidity = ValidatorFluidity(self.net, rho=rho, fast=True, rng=self.rng)
        phase = self.rng.randrange(epoch_length) if policy == 'epoch' else 0
        for b in range(1, num_blocks + 1):
            if policy == 'cfp':
                fluidity.rotate_all_shards(block_hash='0' * 16)
            elif policy == 'epoch' and (b + phase) % epoch_length == 0:
                self._redraw_all()
            elif policy == 'aecfr':
                seats = [(sh, i) for sh in self.net.shards.values() for i in range(len(sh.validators))]
                sel = [s for s in seats if self.rng.random() < alpha]
                movers = [sh.validators[i] for sh, i in sel]
                self.rng.shuffle(movers)
                for (sh, i), v in zip(sel, movers):
                    sh.validators[i] = v
                    v.current_shard = sh.shard_id
            out[b] = self._counts()
        return out


def capture_episodes(counts, n):
    """
    Capture episodes (maximal runs of blocks with >= ceil(n/3) adversarial seats) per shard.
    Returns (completed_lengths, right_censored_lengths, left_censored_count). Episodes already
    active at block 0 are left-censored and excluded; episodes running at the horizon are
    right-censored and reported separately.
    """
    import numpy as np
    cap = counts >= capture_count(n)
    T = cap.shape[0]
    done, rc, lc = [], [], 0
    for k in range(cap.shape[1]):
        c = cap[:, k].astype(np.int8)
        if not c.any():
            continue
        d = np.diff(np.r_[0, c, 0])
        for st, en in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if st == 0:
                lc += 1
            elif en == T:
                rc.append(en - st)
            else:
                done.append(en - st)
    return np.array(done), np.array(rc), lc


def recovery_times(counts, n):
    """Blocks from capture onset until the shard is back below ceil(n/3) (completed episodes only)."""
    d, _, _ = capture_episodes(counts, n)
    return d


class ConsensusMonitoringExperiment:
    """
    Per-height voting model used to evaluate CFI and the bounded controller (Sec. IV-E).

    Each finalized height in each shard:
      * a leader is drawn uniformly from the committee;
      * honest leaders propose valid blocks; adversarial leaders propose an invalid block with
        probability p_invalid under the 'invalid' and 'withhold' strategies;
      * honest validators accept valid and reject invalid proposals (votes are never randomized);
        each honest vote is missing with probability eps;
      * 'coordinated': adversaries always accept; 'withhold': once the adversary holds
        >= ceil(n/3) seats its members send no votes; 'equivocate': each adversarial member
        double-signs with probability p_eq, producing a verifiable proof (flag E);
      * a timeout-related view change (flag D) happens when the accepting votes on a valid block
        do not reach the quorum q = n - ceil(n/3) + 1, or exogenously with probability p_net
        (network delay; no attack implied).
    After each height every shard replaces its currently authorized number of validators (Eq. 14).
    """

    def __init__(self, num_shards, validators_per_shard, total_validators, adversary_fraction, seed):
        self.base = RotationStrategyExperiment(num_shards, validators_per_shard, total_validators,
                                               adversary_fraction, seed)
        self.rng = self.base.rng
        self.n = validators_per_shard

    def run(self, heights, strategy='none', controller=True, r_fixed=None, r_min=None, B_rot=None,
            T=10, m_min=5, tau=0.5, w=3, eps=0.01, p_net=0.0, p_invalid=0.5, p_eq=0.002):
        import numpy as np
        from cfp.fluidity import ValidatorFluidity
        from cfp.cfi import BoundedRotationController
        n, rng = self.n, self.rng
        net = self.base.net
        flu = ValidatorFluidity(net, fast=True, rng=rng)
        r_min = r_min if r_min is not None else max(1, math.ceil(0.1 * n))
        B_rot = B_rot if B_rot is not None else math.ceil(0.6 * n)
        ctl = None
        if r_fixed is not None:
            for sid in net.shards:
                flu.shard_count[sid] = r_fixed
        else:
            ctl = BoundedRotationController(flu, n, r_min, B_rot, T=T, m_min=m_min, tau=tau, w=w)
        q = n - capture_count(n) + 1
        K = len(net.shards)
        counts = np.zeros((heights, K), dtype=np.int16)
        r_hist = np.zeros((heights, K), dtype=np.int16)
        rows = {sid: {} for sid in net.shards}
        E = {sid: False for sid in net.shards}
        D = {sid: False for sid in net.shards}
        hw = 0
        for h in range(heights):
            for sid, sh in net.shards.items():
                mem = sh.validators
                x = sum(v.is_adversarial for v in mem)
                counts[h, sid] = x
                r_hist[h, sid] = flu.calculate_rotation_count(sh)
                leader = mem[rng.randrange(len(mem))]
                valid = not (leader.is_adversarial and strategy in ('invalid', 'withhold')
                             and rng.random() < p_invalid)
                accept = 0
                obs = {}
                for v in mem:
                    if v.is_adversarial:
                        if strategy == 'withhold' and x >= capture_count(n):
                            vote = None
                        elif strategy == 'coordinated':
                            vote = 1
                        else:
                            vote = 1 if valid else -1
                    else:
                        vote = None if rng.random() < eps else (1 if valid else -1)
                    obs[v.validator_id] = vote
                    accept += vote == 1
                if (valid and accept < q) or rng.random() < p_net:
                    D[sid] = True
                if strategy == 'equivocate' and any(v.is_adversarial and rng.random() < p_eq for v in mem):
                    E[sid] = True
                wr = rows[sid]
                for vid, vote in obs.items():
                    wr.setdefault(vid, [None] * hw).append(vote)
                for vid, row in wr.items():
                    if len(row) < hw + 1:
                        row.append(None)
            hw += 1
            if hw == T:
                for sid in net.shards:
                    if ctl is not None:
                        ctl.close_window(sid, rows[sid], E[sid], True, D[sid])
                    else:
                        from cfp.cfi import cfi_window
                        c, m = cfi_window(rows[sid], T, m_min)
                        self._fixed_records = getattr(self, '_fixed_records', [])
                        self._fixed_records.append(dict(shard=sid, cfi=c, cohort=m, E=E[sid], D=D[sid],
                                                        A=E[sid] or (D[sid] and c is not None and c < tau)))
                    rows[sid] = {}
                    E[sid] = D[sid] = False
                hw = 0
            flu.rotate_all_shards(block_hash='0' * 16)
        records = ctl.records if ctl is not None else getattr(self, '_fixed_records', [])
        return dict(counts=counts, r_hist=r_hist, records=records)
