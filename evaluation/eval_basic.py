"""
CFP Scientific Evaluation Module
Uses REAL classes from phases 1-7 to measure paper metrics.

"""

import time
import random
import os

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

# Plotting
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

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
        self.cfi_monitor = ConsensusFrictionIndex(self.fluidity, threshold=0.5, window=10)
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

    def evaluate_capture_probability(self, num_blocks=30, num_trials=1):
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
                # Static system - run simulation without rotation
                static_net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
                static_sim = StaticCommitteeSimulator(static_net)
                static_sim.setup_adversary(f)
                static_history = static_sim.simulate(num_blocks, threshold=0.5)
                static_analysis = static_sim.analyze_results(static_history, threshold=0.5)
                static_rates.append(static_analysis["overall_capture_rate"])

                # CFP system - run simulation with rotation
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

        self.results["performance"] = {"tps": tps_result['throughput_tps'],
                                       "rotation_ms": rot_result['mean_time'] * 1000}
        return self.results["performance"]

    def evaluate_partitioning(self):
        """Evaluate account partitioning effect"""
        print("\n" + "-" * 50)
        print("METRIC 7: Account Partitioning")
        print("-" * 50)

        partitioner = AccountPartitioner(self.network)

        for _ in range(500):
            cluster = random.randint(0, 3)
            base = cluster * 50
            sender = f"user_{base + random.randint(0, 49)}"
            receiver = f"user_{base + random.randint(0, 49)}"
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
        """Generate all plots with improved design"""
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Plot 1: Cumulative Refresh - Clean without threshold line
        if "cumulative_refresh" in self.results:
            data = self.results["cumulative_refresh"]
            fig, ax = plt.subplots(figsize=(10, 6))
            blocks = [d["blocks"] for d in data]
            R_cum = [d["R_cum"] * 100 for d in data]

            ax.plot(blocks, R_cum, color='#2980b9', linewidth=3, marker='o', markersize=10,
                    markerfacecolor='white', markeredgewidth=2)
            ax.fill_between(blocks, R_cum, alpha=0.2, color='#2980b9')

            ax.set_xlabel('Number of Blocks (L)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Committee Refreshed (%)', fontsize=14, fontweight='bold')
            ax.set_title('Cumulative Refresh Rate\n(Equation 14: R_cum = 1 - (1-ρ)^L)',
                         fontsize=16, fontweight='bold', pad=15)
            ax.set_ylim(0, 105)
            ax.set_xlim(0, 32)

            # Add annotations
            ax.annotate(f'{R_cum[2]:.0f}%', xy=(blocks[2], R_cum[2]), xytext=(blocks[2] + 2, R_cum[2] - 10),
                        fontsize=12, fontweight='bold', color='#2980b9')
            ax.annotate(f'{R_cum[4]:.0f}%', xy=(blocks[4], R_cum[4]), xytext=(blocks[4] + 2, R_cum[4] - 5),
                        fontsize=12, fontweight='bold', color='#2980b9')

            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/01_cumulative_refresh.png", dpi=300, facecolor='white')
            plt.close()
            print("  Saved: 01_cumulative_refresh.png")

        # Plot 2: CFI - Horizontal bar chart
        if "cfi" in self.results:
            fig, ax = plt.subplots(figsize=(10, 6))

            categories = ['40% Adversary\n(Coordinated)', '30% Adversary\n(Coordinated)', 'Honest\nVoting']
            values = [self.results["cfi"]["avg_40"], self.results["cfi"]["avg_30"], self.results["cfi"]["avg_honest"]]
            colors = ['#e74c3c', '#f39c12', '#27ae60']

            bars = ax.barh(categories, values, color=colors, edgecolor='white', linewidth=2, height=0.6)

            ax.axvline(x=0.5, color='#2c3e50', linestyle='--', linewidth=2, label='Detection Threshold (0.5)')

            for bar, val in zip(bars, values):
                ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f'{val:.3f}',
                        va='center', fontsize=14, fontweight='bold')

            ax.set_xlabel('CFI Value', fontsize=14, fontweight='bold')
            ax.set_title('Consensus Friction Index (CFI)\nHigher = More Diverse Voting',
                         fontsize=16, fontweight='bold', pad=15)
            ax.set_xlim(0, 1.1)
            ax.legend(loc='lower right', fontsize=11)

            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/02_cfi_comparison.png", dpi=300, facecolor='white')
            plt.close()
            print("  Saved: 02_cfi_comparison.png")

        # Plot 3: Capture Probability - Area chart
        if "capture" in self.results:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = [f * 100 for f in self.results["capture"]["fractions"]]
            static = [r * 100 for r in self.results["capture"]["static_rate"]]
            cfp = [r * 100 for r in self.results["capture"]["cfp_rate"]]

            ax.fill_between(x, static, alpha=0.3, color='#e74c3c', label='Static System')
            ax.fill_between(x, cfp, alpha=0.3, color='#27ae60', label='CFP System')
            ax.plot(x, static, 'o-', color='#c0392b', linewidth=3, markersize=10, markerfacecolor='white',
                    markeredgewidth=2)
            ax.plot(x, cfp, 's-', color='#1e8449', linewidth=3, markersize=10, markerfacecolor='white',
                    markeredgewidth=2)

            ax.set_xlabel('Adversary Fraction (%)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Blocks with Capture (%)', fontsize=14, fontweight='bold')
            ax.set_title('Capture Rate Comparison\nStatic vs CFP (Equations 22-23)',
                         fontsize=16, fontweight='bold', pad=15)
            ax.legend(fontsize=12, loc='upper left')
            ax.set_ylim(0, 110)

            # Add improvement annotation
            if static[-1] > cfp[-1]:
                improvement = static[-1] - cfp[-1]
                ax.annotate(f'{improvement:.0f}% better', xy=(40, (static[-1] + cfp[-1]) / 2),
                            fontsize=12, fontweight='bold', color='#27ae60',
                            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#27ae60'))

            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/03_capture_probability.png", dpi=300, facecolor='white')
            plt.close()
            print("  Saved: 03_capture_probability.png")

        # Plot 4: Performance - Gauge style
        if "performance" in self.results:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # TPS gauge
            ax1 = axes[0]
            tps_val = self.results["performance"]["tps"]
            ax1.barh(['Throughput'], [tps_val], color='#3498db', edgecolor='white', height=0.5)
            ax1.set_xlim(0, tps_val * 1.3)
            ax1.set_xlabel('Transactions Per Second', fontsize=14, fontweight='bold')
            ax1.set_title(f'TPS: {tps_val:,.0f}', fontsize=18, fontweight='bold', color='#3498db')
            ax1.tick_params(left=False, labelleft=False)

            # Rotation gauge
            ax2 = axes[1]
            rot_val = self.results["performance"]["rotation_ms"]
            ax2.barh(['Overhead'], [rot_val], color='#e67e22', edgecolor='white', height=0.5)
            ax2.set_xlim(0, rot_val * 1.5)
            ax2.set_xlabel('Time (milliseconds)', fontsize=14, fontweight='bold')
            ax2.set_title(f'Rotation: {rot_val:.0f} ms', fontsize=18, fontweight='bold', color='#e67e22')
            ax2.tick_params(left=False, labelleft=False)

            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/04_performance.png", dpi=300, facecolor='white')
            plt.close()
            print("  Saved: 04_performance.png")

        # Plot 5: Partitioning - Before/After with arrow
        if "partitioning" in self.results:
            fig, ax = plt.subplots(figsize=(10, 6))

            before = self.results["partitioning"]["before"] * 100
            after = self.results["partitioning"]["after"] * 100

            x_pos = [0, 1]
            values = [before, after]
            colors = ['#e74c3c', '#27ae60']

            bars = ax.bar(x_pos, values, color=colors, edgecolor='white', linewidth=2, width=0.5)

            # Add arrow showing improvement
            ax.annotate('', xy=(1, after + 5), xytext=(0, before - 5),
                        arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=3))
            ax.text(0.5, before / 2 + 10, f'-{self.results["partitioning"]["improvement"]:.0f}%',
                    ha='center', fontsize=20, fontweight='bold', color='#27ae60')

            ax.set_xticks(x_pos)
            ax.set_xticklabels(['Before\nPartitioning', 'After\nPartitioning'], fontsize=14, fontweight='bold')
            ax.set_ylabel('Cross-Shard Ratio (%)', fontsize=14, fontweight='bold')
            ax.set_title('Effect of Louvain Clustering\non Cross-Shard Transactions',
                         fontsize=16, fontweight='bold', pad=15)
            ax.set_ylim(0, before + 15)

            # Add value labels
            ax.text(0, before + 2, f'{before:.1f}%', ha='center', fontsize=14, fontweight='bold', color='#c0392b')
            ax.text(1, after + 2, f'{after:.1f}%', ha='center', fontsize=14, fontweight='bold', color='#1e8449')

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
        self.evaluate_capture_probability(num_blocks=50)
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
