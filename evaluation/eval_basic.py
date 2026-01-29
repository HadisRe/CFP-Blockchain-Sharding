 

import time
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import from phase 1-2: Base infrastructure
from sharding import Network

# Import from phase 3: Validator fluidity and CFI
from cfp.fluidity import ValidatorFluidity
from cfp.cfi import ConsensusFrictionIndex

# Import from phase 4: Cross-shard transactions
from cfp.cross_shard import CrossShardCoordinator

# Import from phase 5: Account partitioning
from cfp.partitioning import AccountPartitioner

# Import from phase 6: Attack simulation
from cfp.attack import StaticCommitteeSimulator, CFPCommitteeSimulator

# Import from phase 7: Benchmarking
from cfp.benchmark import ThroughputBenchmark, RotationOverheadBenchmark

# Create output directory
PLOT_DIR = "evaluation_plots"
os.makedirs(PLOT_DIR, exist_ok=True)


class CFPEvaluator:
    """
    Main evaluator class that uses real components from phases 1-7.
    """

    def __init__(self, num_shards=4, validators_per_shard=25, total_validators=200):
        """Setup network with real components."""
        print("=" * 60)
        print("CFP SCIENTIFIC EVALUATION")
        print("Using REAL classes from phases 1-7")
        print("=" * 60)

        # Phase 1-2: Create real network
        print("\n[Setup] Creating network...")
        self.network = Network(
            num_shards=num_shards,
            validators_per_shard=validators_per_shard,
            total_validators=total_validators
        )
        print(f"  Network: {num_shards} shards, {validators_per_shard} validators each")
        print(f"  Total validators: {total_validators}")

        # Phase 3: Create fluidity manager with real VRF
        self.fluidity = ValidatorFluidity(self.network, rho=0.2)
        print(f"  Fluidity: rho={self.fluidity.rho}")

        # Phase 3: Create CFI monitor
        self.cfi_monitor = ConsensusFrictionIndex(self.fluidity, threshold=0.65, window=10)
        print(f"  CFI monitor: threshold={self.cfi_monitor.threshold}")

        # Phase 4: Create cross-shard coordinator
        self.coordinator = CrossShardCoordinator(self.network)
        print(f"  Cross-shard coordinator: ready")

        # Register accounts
        self.accounts = [f"user_{i}" for i in range(200)]
        for acc in self.accounts:
            self.network.register_account(acc)
        print(f"  Accounts registered: {len(self.accounts)}")

        # Results storage
        self.results = {}

        print("\n[Setup] Complete!")

    def evaluate_rotation_count(self):
        """Evaluate equation 12: rt = ceil(rho * n)"""
        print("\n" + "-" * 50)
        print("METRIC 1: Rotation Count (Equation 12)")
        print("-" * 50)

        results = []
        for shard_id, shard in self.network.shards.items():
            rt = self.fluidity.calculate_rotation_count(shard)
            n = shard.get_validator_count()
            results.append({"shard_id": shard_id, "n": n, "rho": self.fluidity.rho, "rt": rt})
            print(f"  Shard {shard_id}: n={n}, rho={self.fluidity.rho}, rt={rt}")

        self.results["rotation_count"] = results
        return results

    def evaluate_cumulative_refresh(self):
        """Evaluate equation 14: R_cum(L) = 1 - (1 - rho)^L"""
        print("\n" + "-" * 50)
        print("METRIC 2: Cumulative Refresh (Equation 14)")
        print("-" * 50)

        results = []
        for L in [1, 5, 10, 15, 20, 25, 30]:
            R_cum = self.fluidity.get_cumulative_refresh(L)
            results.append({"blocks": L, "R_cum": R_cum})
            print(f"  After {L:2d} blocks: {R_cum:.1%} committee refreshed")

        self.results["cumulative_refresh"] = results
        return results

    def evaluate_cfi(self, num_rounds=50):
        """Evaluate equation 15: CFI"""
        print("\n" + "-" * 50)
        print("METRIC 3: CFI (Equation 15)")
        print("-" * 50)

        results = {"honest": [], "coordinated_30": [], "coordinated_40": []}

        # Honest voting
        print("\n  Scenario 1: Honest voting...")
        for v in self.network.validator_pool.validators:
            v.is_adversarial = False
            v.vote_history = []

        for _ in range(num_rounds):
            self.cfi_monitor.simulate_voting_round(self.network, adversarial_coordination=False)
            for shard in self.network.shards.values():
                results["honest"].append(self.cfi_monitor.calculate_cfi(shard))

        avg_honest = sum(results["honest"]) / len(results["honest"])
        print(f"    Average CFI: {avg_honest:.3f}")

        # 30% coordinated
        print("\n  Scenario 2: 30% coordinated adversary...")
        self.network.validator_pool.set_adversarial_fraction(0.3)
        for v in self.network.validator_pool.validators:
            v.vote_history = []

        for _ in range(num_rounds):
            self.cfi_monitor.simulate_voting_round(self.network, adversarial_coordination=True)
            for shard in self.network.shards.values():
                results["coordinated_30"].append(self.cfi_monitor.calculate_cfi(shard))

        avg_30 = sum(results["coordinated_30"]) / len(results["coordinated_30"])
        print(f"    Average CFI: {avg_30:.3f}")

        # 40% coordinated
        print("\n  Scenario 3: 40% coordinated adversary...")
        self.network.validator_pool.set_adversarial_fraction(0.4)
        for v in self.network.validator_pool.validators:
            v.vote_history = []

        for _ in range(num_rounds):
            self.cfi_monitor.simulate_voting_round(self.network, adversarial_coordination=True)
            for shard in self.network.shards.values():
                results["coordinated_40"].append(self.cfi_monitor.calculate_cfi(shard))

        avg_40 = sum(results["coordinated_40"]) / len(results["coordinated_40"])
        print(f"    Average CFI: {avg_40:.3f}")

        for v in self.network.validator_pool.validators:
            v.is_adversarial = False

        self.results["cfi"] = {"avg_honest": avg_honest, "avg_30": avg_30, "avg_40": avg_40}
        return self.results["cfi"]

    def evaluate_capture_probability(self, num_blocks=30, num_trials=50):
        """Evaluate equations 22 and 23: Capture probability"""
        print("\n" + "-" * 50)
        print("METRIC 4: Capture Probability (Equations 22, 23)")
        print("-" * 50)

        adversary_fractions = [0.2, 0.25, 0.3, 0.35, 0.4]
        results = {"fractions": adversary_fractions, "static_rate": [], "cfp_rate": []}

        for f in adversary_fractions:
            print(f"\n  Testing {f:.0%} adversary ({num_trials} trials)...")

            static_rates = []
            cfp_rates = []

            for trial in range(num_trials):
                # Static system
                static_net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
                static_sim = StaticCommitteeSimulator(static_net)
                static_sim.setup_adversary(f)
                static_history = static_sim.simulate(num_blocks, threshold=0.5)
                static_analysis = static_sim.analyze_results(static_history, threshold=0.5)
                static_rates.append(static_analysis["overall_capture_rate"])

                # CFP system
                cfp_net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
                cfp_fluidity = ValidatorFluidity(cfp_net, rho=0.2)
                cfp_cfi = ConsensusFrictionIndex(cfp_fluidity, threshold=0.5, window=10)
                cfp_sim = CFPCommitteeSimulator(cfp_net, cfp_fluidity, cfp_cfi)
                cfp_sim.setup_adversary(f)
                cfp_history = cfp_sim.simulate(num_blocks, threshold=0.5)
                cfp_analysis = cfp_sim.analyze_results(cfp_history, threshold=0.5)
                cfp_rates.append(cfp_analysis["overall_capture_rate"])

            avg_static = sum(static_rates) / len(static_rates)
            avg_cfp = sum(cfp_rates) / len(cfp_rates)

            results["static_rate"].append(avg_static)
            results["cfp_rate"].append(avg_cfp)

            print(f"    Static rate: {avg_static:.1%}, CFP rate: {avg_cfp:.1%}")

        self.results["capture"] = results
        return results

    def evaluate_cross_shard(self, num_tx=50):
        """Evaluate cross-shard transaction success"""
        print("\n" + "-" * 50)
        print("METRIC 5: Cross-Shard Transaction Success")
        print("-" * 50)

        successful, failed, latencies = 0, 0, []

        for _ in range(num_tx):
            sender = random.choice(self.accounts)
            receiver = random.choice([a for a in self.accounts if a != sender])

            if self.network.is_cross_shard(sender, receiver):
                start = time.perf_counter()
                tx, _ = self.coordinator.process_transaction(sender, receiver, 50)
                latency = time.perf_counter() - start

                if tx and tx.status == "committed":
                    successful += 1
                    latencies.append(latency)
                else:
                    failed += 1

        total = successful + failed
        rate = successful / total if total > 0 else 0
        avg_lat = sum(latencies) / len(latencies) * 1000 if latencies else 0

        print(f"  Success rate: {rate:.1%}")
        print(f"  Avg latency: {avg_lat:.2f} ms")

        self.results["cross_shard"] = {"rate": rate, "latency_ms": avg_lat}
        return self.results["cross_shard"]

    def evaluate_performance(self, num_tx=300):
        """Evaluate TPS and rotation overhead"""
        print("\n" + "-" * 50)
        print("METRIC 6: Performance")
        print("-" * 50)

        tps_bench = ThroughputBenchmark(self.network)
        tps_result = tps_bench.run(num_tx)
        print(f"  TPS: {tps_result['throughput_tps']:.0f}")

        rot_bench = RotationOverheadBenchmark(self.network, self.fluidity)
        rot_result = rot_bench.run(30)
        print(f"  Rotation overhead: {rot_result['mean_time'] * 1000:.1f} ms")

        self.results["performance"] = {
            "tps": tps_result['throughput_tps'],
            "rotation_ms": rot_result['mean_time'] * 1000
        }
        return self.results["performance"]

    def evaluate_partitioning(self):
        """Evaluate account partitioning effect"""
        print("\n" + "-" * 50)
        print("METRIC 7: Account Partitioning")
        print("-" * 50)

        partitioner = AccountPartitioner(self.network)

        # 70% intra-cluster transactions, 30% inter-cluster
        for _ in range(500):
            if random.random() < 0.7:
                cluster = random.randint(0, 3)
                base = cluster * 50
                sender = f"user_{base + random.randint(0, 49)}"
                receiver = f"user_{base + random.randint(0, 49)}"
            else:
                sender = f"user_{random.randint(0, 199)}"
                receiver = f"user_{random.randint(0, 199)}"

            if sender != receiver:
                partitioner.record_transaction(sender, receiver)

        before_stats = partitioner.analyze_current_distribution()
        before = before_stats["cross_ratio"]
        migrations = partitioner.repartition()
        after_stats = partitioner.analyze_current_distribution()
        after = after_stats["cross_ratio"]

        improvement = (before - after) / before * 100 if before > 0 else 0

        print(f"  Before: {before:.1%}")
        print(f"  After: {after:.1%}")
        print(f"  Improvement: {improvement:.1f}%")

        self.results["partitioning"] = {"before": before, "after": after, "improvement": improvement}
        return self.results["partitioning"]

    def plot_all(self):
        """Generate publication-quality plots"""
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = False
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['font.weight'] = 'normal'

        # Plot 1: Cumulative Refresh
        if "cumulative_refresh" in self.results:
            data = self.results["cumulative_refresh"]
            fig, ax = plt.subplots(figsize=(7, 5))
            blocks = [d["blocks"] for d in data]
            R_cum = [d["R_cum"] * 100 for d in data]

            ax.fill_between(blocks, R_cum, alpha=0.25, color='#3498db')
            ax.plot(blocks, R_cum, 'o-', color='#2980b9', linewidth=2, markersize=6,
                    markerfacecolor='white', markeredgewidth=1.5)
            ax.axhline(y=95, color='#95a5a6', linestyle=':', linewidth=1)

            ax.set_xlabel('Number of Blocks (L)')
            ax.set_ylabel('Committee Refreshed (%)')
            ax.set_ylim(0, 105)
            ax.set_xlim(0, 32)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/01_cumulative_refresh.png", dpi=300, facecolor='white')
            plt.close()
            print("  Saved: 01_cumulative_refresh.png")

        # Plot 2: CFI Comparison
        if "cfi" in self.results:
            fig, ax = plt.subplots(figsize=(7, 4))

            categories = ['Honest Voting', '30% Adversary', '40% Adversary']
            values = [self.results["cfi"]["avg_honest"], self.results["cfi"]["avg_30"], self.results["cfi"]["avg_40"]]
            colors = ['#27ae60', '#f39c12', '#c0392b']

            y_pos = np.arange(len(categories))
            bars = ax.barh(y_pos, values, color=colors, height=0.5, edgecolor='white')
            ax.axvline(x=0.65, color='#7f8c8d', linestyle='--', linewidth=1.2, alpha=0.7)

            for bar, val in zip(bars, values):
                ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f'{val:.3f}',
                        va='center', fontsize=10)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories)
            ax.set_xlabel('CFI Value')
            ax.set_xlim(0, 1.0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/02_cfi_comparison.png", dpi=300, facecolor='white')
            plt.close()
            print("  Saved: 02_cfi_comparison.png")

        # Plot 3: Capture Probability
        if "capture" in self.results:
            fig, ax = plt.subplots(figsize=(7, 5))
            x = [f * 100 for f in self.results["capture"]["fractions"]]
            static = [r * 100 for r in self.results["capture"]["static_rate"]]
            cfp = [r * 100 for r in self.results["capture"]["cfp_rate"]]

            ax.fill_between(x, static, alpha=0.15, color='#e74c3c')
            ax.fill_between(x, cfp, alpha=0.15, color='#27ae60')
            ax.plot(x, static, 'o-', color='#c0392b', linewidth=2, markersize=6,
                    markerfacecolor='white', markeredgewidth=1.5, label='Static')
            ax.plot(x, cfp, 's-', color='#27ae60', linewidth=2, markersize=6,
                    markerfacecolor='white', markeredgewidth=1.5, label='CFP')

            ax.set_xlabel('Adversary Fraction (%)')
            ax.set_ylabel('Capture Rate (%)')
            ax.legend(loc='upper left', frameon=False, fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/03_capture_probability.png", dpi=300, facecolor='white')
            plt.close()
            print("  Saved: 03_capture_probability.png")

        # Plot 4: Performance - Semi-circle gauges
        if "performance" in self.results:
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            tps = self.results["performance"]["tps"]
            overhead = self.results["performance"]["rotation_ms"]

            theta = np.linspace(np.pi, 0, 100)
            r_outer, r_inner = 1.0, 0.6

            # TPS gauge
            ax1 = axes[0]
            ax1.fill_between(np.cos(theta) * r_outer, np.sin(theta) * r_outer,
                             np.sin(theta) * r_inner, alpha=0.2, color='#3498db')
            fill_ratio = min(tps / 100000, 1.0)
            theta_fill = np.linspace(np.pi, np.pi - fill_ratio * np.pi, 50)
            ax1.fill_between(np.cos(theta_fill) * r_outer, np.sin(theta_fill) * r_outer,
                             np.sin(theta_fill) * r_inner, color='#3498db', alpha=0.8)
            ax1.plot(np.cos(theta) * r_outer, np.sin(theta) * r_outer, 'k-', linewidth=1)
            ax1.plot(np.cos(theta) * r_inner, np.sin(theta) * r_inner, 'k-', linewidth=1)
            ax1.text(0, 0.15, f'{tps/1000:.1f}K', ha='center', va='center',
                     fontsize=18, fontweight='bold', color='#2c3e50')
            ax1.text(0, -0.15, 'TPS', ha='center', va='center', fontsize=11, color='#7f8c8d')
            ax1.set_xlim(-1.3, 1.3)
            ax1.set_ylim(-0.4, 1.2)
            ax1.set_aspect('equal')
            ax1.axis('off')

            # Overhead gauge
            ax2 = axes[1]
            ax2.fill_between(np.cos(theta) * r_outer, np.sin(theta) * r_outer,
                             np.sin(theta) * r_inner, alpha=0.2, color='#e67e22')
            fill_ratio2 = min(overhead / 1000, 1.0)
            theta_fill2 = np.linspace(np.pi, np.pi - fill_ratio2 * np.pi, 50)
            ax2.fill_between(np.cos(theta_fill2) * r_outer, np.sin(theta_fill2) * r_outer,
                             np.sin(theta_fill2) * r_inner, color='#e67e22', alpha=0.8)
            ax2.plot(np.cos(theta) * r_outer, np.sin(theta) * r_outer, 'k-', linewidth=1)
            ax2.plot(np.cos(theta) * r_inner, np.sin(theta) * r_inner, 'k-', linewidth=1)
            ax2.text(0, 0.15, f'{overhead:.0f}', ha='center', va='center',
                     fontsize=18, fontweight='bold', color='#2c3e50')
            ax2.text(0, -0.15, 'Rotation (ms)', ha='center', va='center', fontsize=11, color='#7f8c8d')
            ax2.set_xlim(-1.3, 1.3)
            ax2.set_ylim(-0.4, 1.2)
            ax2.set_aspect('equal')
            ax2.axis('off')

            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/04_performance.png", dpi=300, facecolor='white')
            plt.close()
            print("  Saved: 04_performance.png")

        # Plot 5: Partitioning - Donut charts
        if "partitioning" in self.results:
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            before = self.results["partitioning"]["before"] * 100
            after = self.results["partitioning"]["after"] * 100

            # Before donut
            ax1 = axes[0]
            wedges1, _ = ax1.pie([before, 100 - before], colors=['#e74c3c', '#fadbd8'],
                                  startangle=90, wedgeprops=dict(width=0.4, edgecolor='white'))
            ax1.text(0, 0, f'{before:.0f}%', ha='center', va='center',
                     fontsize=16, fontweight='bold', color='#c0392b')
            ax1.set_title('Before', fontsize=11, pad=10)

            # After donut
            ax2 = axes[1]
            wedges2, _ = ax2.pie([after, 100 - after], colors=['#27ae60', '#d5f5e3'],
                                  startangle=90, wedgeprops=dict(width=0.4, edgecolor='white'))
            ax2.text(0, 0, f'{after:.0f}%', ha='center', va='center',
                     fontsize=16, fontweight='bold', color='#1e8449')
            ax2.set_title('After', fontsize=11, pad=10)

            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/05_partitioning.png", dpi=300, facecolor='white')
            plt.close()
            print("  Saved: 05_partitioning.png")

        print(f"\nAll plots saved to: {PLOT_DIR}/")

    def run_full_evaluation(self):
        """Run complete evaluation"""
        self.evaluate_rotation_count()
        self.evaluate_cumulative_refresh()
        self.evaluate_cfi(num_rounds=30)
        self.evaluate_capture_probability(num_blocks=50, num_trials=50)
        self.evaluate_cross_shard(num_tx=50)
        self.evaluate_performance(num_tx=300)
        self.evaluate_partitioning()
        self.plot_all()

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)

        return self.results


if __name__ == "__main__":
    evaluator = CFPEvaluator()
    results = evaluator.run_full_evaluation()
