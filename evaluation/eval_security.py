"""
Extended metrics including attack scenarios, rho sensitivity, capture duration.
"""

import time
import random
import os
import numpy as np

from sharding import Network
from cfp.fluidity import ValidatorFluidity
from cfp.cfi import ConsensusFrictionIndex
from cfp.cross_shard import CrossShardCoordinator
from cfp.partitioning import AccountPartitioner
from cfp.attack import StaticCommitteeSimulator, CFPCommitteeSimulator

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

PLOT_DIR = "evaluation_plots"
os.makedirs(PLOT_DIR, exist_ok=True)


class AdvancedCFPEvaluator:
    """Advanced evaluation with attack scenarios and sensitivity analysis."""

    def __init__(self):
        print("=" * 70)
        print("   CFP ADVANCED SCIENTIFIC EVALUATION")
        print("=" * 70)
        self.results = {}

    def evaluate_rho_sensitivity(self):
        """Test different rotation rates and their effect on security and overhead."""
        print("\n" + "=" * 60)
        print("METRIC 8: Rotation Rate (rho) Sensitivity Analysis")
        print("=" * 60)

        rho_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        results = {
            "rho": rho_values,
            "capture_rate": [],
            "rotation_overhead_ms": [],
            "blocks_to_95_refresh": []
        }

        adversary_fraction = 0.35
        num_trials = 5
        num_blocks = 20

        for rho in rho_values:
            print(f"\n  Testing rho = {rho} ({num_trials} trials)...")

            capture_rates = []
            overhead_times = []

            for trial in range(num_trials):
                net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
                fluidity = ValidatorFluidity(net, rho=rho)
                cfi = ConsensusFrictionIndex(fluidity, threshold=0.5, window=10)

                sim = CFPCommitteeSimulator(net, fluidity, cfi)
                sim.setup_adversary(adversary_fraction)
                history = sim.simulate(num_blocks, threshold=0.5)
                analysis = sim.analyze_results(history, threshold=0.5)
                capture_rates.append(analysis["overall_capture_rate"])

                start = time.perf_counter()
                for _ in range(5):
                    fluidity.rotate_all_shards()
                overhead_times.append((time.perf_counter() - start) / 5 * 1000)

            avg_capture = np.mean(capture_rates) * 100
            avg_overhead = np.mean(overhead_times)
            results["capture_rate"].append(avg_capture)
            results["rotation_overhead_ms"].append(avg_overhead)

            blocks_95 = int(np.ceil(np.log(0.05) / np.log(1 - rho)))
            results["blocks_to_95_refresh"].append(blocks_95)

            print(f"    Capture rate: {avg_capture:.1f}%")
            print(f"    Overhead: {avg_overhead:.1f} ms")
            print(f"    Blocks to 95% refresh: {blocks_95}")

        self.results["rho_sensitivity"] = results
        return results

    def evaluate_shard_scalability(self):
        """Test how number of shards affects security and performance."""
        print("\n" + "=" * 60)
        print("METRIC 9: Shard Scalability Analysis")
        print("=" * 60)

        shard_counts = [2, 4, 8, 16]
        results = {
            "shards": shard_counts,
            "capture_rate": [],
            "tps": [],
            "validators_per_shard": []
        }

        total_validators = 200
        adversary_fraction = 0.35
        num_trials = 5

        for num_shards in shard_counts:
            validators_per_shard = total_validators // (num_shards * 2)
            print(f"\n  Testing {num_shards} shards ({validators_per_shard} validators each)...")

            capture_rates = []
            for trial in range(num_trials):
                net = Network(
                    num_shards=num_shards,
                    validators_per_shard=validators_per_shard,
                    total_validators=total_validators
                )
                fluidity = ValidatorFluidity(net, rho=0.2)
                cfi = ConsensusFrictionIndex(fluidity, threshold=0.5, window=10)

                sim = CFPCommitteeSimulator(net, fluidity, cfi)
                sim.setup_adversary(adversary_fraction)
                history = sim.simulate(30, threshold=0.5)
                analysis = sim.analyze_results(history, threshold=0.5)
                capture_rates.append(analysis["overall_capture_rate"])

            results["capture_rate"].append(np.mean(capture_rates) * 100)

            net = Network(num_shards=num_shards, validators_per_shard=validators_per_shard,
                          total_validators=total_validators)
            from cfp.benchmark import ThroughputBenchmark
            bench = ThroughputBenchmark(net)
            tps_result = bench.run(100)
            results["tps"].append(tps_result["throughput_tps"])
            results["validators_per_shard"].append(validators_per_shard)

            print(f"    Capture rate: {results['capture_rate'][-1]:.1f}%")
            print(f"    TPS: {tps_result['throughput_tps']:.0f}")

        self.results["shard_scalability"] = results
        return results

    def evaluate_capture_duration(self):
        """Measure how long adversary maintains control in Static vs CFP."""
        print("\n" + "=" * 60)
        print("METRIC 10: Capture Duration Analysis")
        print("=" * 60)

        results = {
            "static_durations": [],
            "cfp_durations": [],
            "static_avg": 0,
            "cfp_avg": 0
        }

        num_experiments = 10
        num_blocks = 50
        adversary_fraction = 0.4

        print(f"\n  Testing Static system ({num_experiments} experiments)...")
        for exp in range(num_experiments):
            net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
            sim = StaticCommitteeSimulator(net)
            sim.setup_adversary(adversary_fraction)

            captured_duration = 0
            for shard in net.shards.values():
                if sim.adversary.has_majority(shard, 0.5):
                    captured_duration = num_blocks
                    break

            results["static_durations"].append(captured_duration)

        print(f"  Testing CFP system ({num_experiments} experiments)...")
        for exp in range(num_experiments):
            net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
            fluidity = ValidatorFluidity(net, rho=0.2)
            cfi = ConsensusFrictionIndex(fluidity, threshold=0.5, window=10)
            sim = CFPCommitteeSimulator(net, fluidity, cfi)
            sim.setup_adversary(adversary_fraction)

            current_streak = 0
            max_streak = 0

            for block in range(num_blocks):
                fluidity.rotate_all_shards()
                captured = False
                for shard in net.shards.values():
                    if sim.adversary.has_majority(shard, 0.5):
                        captured = True
                        break

                if captured:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0

            results["cfp_durations"].append(max_streak)

        results["static_avg"] = np.mean(results["static_durations"])
        results["cfp_avg"] = np.mean(results["cfp_durations"])

        print(f"\n  Results:")
        print(f"    Static average duration: {results['static_avg']:.1f} blocks")
        print(f"    CFP average duration: {results['cfp_avg']:.1f} blocks")

        self.results["capture_duration"] = results
        return results

    def evaluate_attack_scenarios(self):
        """Test CFI under different attack strategies."""
        print("\n" + "=" * 60)
        print("METRIC 11: Attack Scenario Analysis")
        print("=" * 60)

        results = {
            "scenarios": ["No Attack", "Random Attack", "Coordinated Attack", "Adaptive Attack"],
            "cfi_values": [],
            "detection": []
        }

        num_rounds = 20
        adversary_fraction = 0.40
        threshold = 0.7

        print("\n  Scenario 1: No Attack (Baseline)...")
        cfi_values = []
        for _ in range(3):
            net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
            fluidity = ValidatorFluidity(net, rho=0.2)
            cfi_monitor = ConsensusFrictionIndex(fluidity, threshold=threshold, window=10)

            for _ in range(num_rounds):
                cfi_monitor.simulate_voting_round(net, adversarial_coordination=False)
                for shard in net.shards.values():
                    cfi_values.append(cfi_monitor.calculate_cfi(shard))

        avg_cfi = np.mean(cfi_values)
        results["cfi_values"].append(avg_cfi)
        results["detection"].append("No" if avg_cfi > threshold else "Yes")
        print(f"    Average CFI: {avg_cfi:.3f} - Detection: {results['detection'][-1]}")

        print("\n  Scenario 2: Random Attack...")
        cfi_values = []
        for _ in range(3):
            net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
            fluidity = ValidatorFluidity(net, rho=0.2)
            cfi_monitor = ConsensusFrictionIndex(fluidity, threshold=threshold, window=10)
            net.validator_pool.set_adversarial_fraction(adversary_fraction)

            for _ in range(num_rounds):
                cfi_monitor.simulate_voting_round(net, adversarial_coordination=False)
                for shard in net.shards.values():
                    cfi_values.append(cfi_monitor.calculate_cfi(shard))

        avg_cfi = np.mean(cfi_values)
        results["cfi_values"].append(avg_cfi)
        results["detection"].append("No" if avg_cfi > threshold else "Yes")
        print(f"    Average CFI: {avg_cfi:.3f} - Detection: {results['detection'][-1]}")

        print("\n  Scenario 3: Coordinated Attack...")
        cfi_values = []
        for _ in range(3):
            net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
            fluidity = ValidatorFluidity(net, rho=0.2)
            cfi_monitor = ConsensusFrictionIndex(fluidity, threshold=threshold, window=10)
            net.validator_pool.set_adversarial_fraction(adversary_fraction)

            for _ in range(num_rounds):
                cfi_monitor.simulate_voting_round(net, adversarial_coordination=True)
                for shard in net.shards.values():
                    cfi_values.append(cfi_monitor.calculate_cfi(shard))

        avg_cfi = np.mean(cfi_values)
        results["cfi_values"].append(avg_cfi)
        results["detection"].append("No" if avg_cfi > threshold else "Yes")
        print(f"    Average CFI: {avg_cfi:.3f} - Detection: {results['detection'][-1]}")

        print("\n  Scenario 4: Adaptive Attack (Evasion Attempt)...")
        cfi_values = []
        for _ in range(3):
            net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
            fluidity = ValidatorFluidity(net, rho=0.2)
            cfi_monitor = ConsensusFrictionIndex(fluidity, threshold=threshold, window=10)
            net.validator_pool.set_adversarial_fraction(adversary_fraction)

            for round_num in range(num_rounds):
                coordinate = (round_num % 3 == 0)
                cfi_monitor.simulate_voting_round(net, adversarial_coordination=coordinate)
                for shard in net.shards.values():
                    cfi_values.append(cfi_monitor.calculate_cfi(shard))

        avg_cfi = np.mean(cfi_values)
        results["cfi_values"].append(avg_cfi)
        results["detection"].append("No" if avg_cfi > threshold else "Yes")
        print(f"    Average CFI: {avg_cfi:.3f} - Detection: {results['detection'][-1]}")

        self.results["attack_scenarios"] = results
        return results

    def plot_all(self):
        """Generate plots."""
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)

        self._plot_rho_sensitivity()
        self._plot_shard_scalability()
        self._plot_attack_scenarios()

        print(f"\nAll plots saved to: {PLOT_DIR}/")

    def _plot_rho_sensitivity(self):
        """Plot rho sensitivity with dual y-axis."""
        if "rho_sensitivity" not in self.results:
            return

        data = self.results["rho_sensitivity"]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color_capture = '#e74c3c'
        color_overhead = '#3498db'

        x = np.array(data["rho"]) * 100
        ax1.fill_between(x, data["capture_rate"], alpha=0.3, color=color_capture)
        line1 = ax1.plot(x, data["capture_rate"], 'o-', color=color_capture,
                         linewidth=3, markersize=10, markerfacecolor='white',
                         markeredgewidth=2, label='Capture Rate (%)')
        ax1.set_xlabel('Rotation Rate rho (%)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Capture Rate (%)', fontsize=14, fontweight='bold', color=color_capture)
        ax1.tick_params(axis='y', labelcolor=color_capture)
        ax1.set_ylim(0, max(data["capture_rate"]) * 1.3)

        ax2 = ax1.twinx()
        line2 = ax2.plot(x, data["rotation_overhead_ms"], 's--', color=color_overhead,
                         linewidth=3, markersize=10, markerfacecolor='white',
                         markeredgewidth=2, label='Rotation Overhead (ms)')
        ax2.set_ylabel('Rotation Overhead (ms)', fontsize=14, fontweight='bold', color=color_overhead)
        ax2.tick_params(axis='y', labelcolor=color_overhead)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=11)

        plt.title('Impact of Rotation Rate on Security and Performance',
                  fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/08_rho_sensitivity.png", dpi=300, facecolor='white')
        plt.close()
        print("  Saved: 08_rho_sensitivity.png")

    def _plot_shard_scalability(self):
        """Plot shard scalability analysis."""
        if "shard_scalability" not in self.results:
            return

        data = self.results["shard_scalability"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(data["shards"])))
        bars = ax1.bar(range(len(data["shards"])), data["capture_rate"], color=colors,
                       edgecolor='white', linewidth=2)

        ax1.set_xticks(range(len(data["shards"])))
        ax1.set_xticklabels([f'{s} shards\n({v} val/shard)' for s, v in
                             zip(data["shards"], data["validators_per_shard"])], fontsize=11)
        ax1.set_ylabel('Capture Rate (%)', fontsize=14, fontweight='bold')
        ax1.set_title('Security vs Number of Shards', fontsize=14, fontweight='bold')

        for bar, val in zip(bars, data["capture_rate"]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')

        ax2 = axes[1]
        colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(data["shards"])))
        bars = ax2.bar(range(len(data["shards"])), [t / 1000 for t in data["tps"]],
                       color=colors, edgecolor='white', linewidth=2)

        ax2.set_xticks(range(len(data["shards"])))
        ax2.set_xticklabels([f'{s} shards' for s in data["shards"]], fontsize=11)
        ax2.set_ylabel('Throughput (K TPS)', fontsize=14, fontweight='bold')
        ax2.set_title('Performance vs Number of Shards', fontsize=14, fontweight='bold')

        for bar, val in zip(bars, data["tps"]):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{val / 1000:.1f}K', ha='center', fontsize=12, fontweight='bold')

        plt.suptitle('Shard Scalability Trade-offs', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/09_shard_scalability.png", dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        print("  Saved: 09_shard_scalability.png")

    def _plot_attack_scenarios(self):
        """Plot attack scenario comparison."""
        if "attack_scenarios" not in self.results:
            return

        data = self.results["attack_scenarios"]

        fig, ax = plt.subplots(figsize=(12, 6))

        y_pos = np.arange(len(data["scenarios"]))
        colors = ['#27ae60', '#3498db', '#e74c3c', '#f39c12']

        bars = ax.barh(y_pos, data["cfi_values"], color=colors, edgecolor='white',
                       linewidth=2, height=0.6)

        ax.axvline(x=0.7, color='#2c3e50', linestyle='--', linewidth=2.5, zorder=5)
        ax.text(0.7, len(y_pos), 'Detection\nThreshold', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#2c3e50')

        for i, (bar, cfi, det) in enumerate(zip(bars, data["cfi_values"], data["detection"])):
            ax.text(cfi + 0.02, bar.get_y() + bar.get_height() / 2, f'{cfi:.3f}',
                    va='center', fontsize=12, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(data["scenarios"], fontsize=12)
        ax.set_xlabel('Consensus Friction Index (CFI)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.1)
        ax.set_title('CFI Response to Different Attack Strategies',
                     fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/11_attack_scenarios.png", dpi=300, facecolor='white')
        plt.close()
        print("  Saved: 11_attack_scenarios.png")

    def run_full_evaluation(self):
        """Run all advanced evaluations."""
        self.evaluate_rho_sensitivity()
        self.evaluate_shard_scalability()
        self.evaluate_capture_duration()
        self.evaluate_attack_scenarios()
        self.plot_all()

        print("\n" + "=" * 70)
        print("   ADVANCED EVALUATION COMPLETE")
        print("=" * 70)

        return self.results


if __name__ == "__main__":
    evaluator = AdvancedCFPEvaluator()
    results = evaluator.run_full_evaluation()
