"""
CFP Advanced Evaluation Module
Extended metrics including attack scenarios, rho sensitivity, capture duration.
Beautiful publication-quality plots.
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
from matplotlib.patches import FancyBboxPatch
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
        print("   Attack Scenarios | Sensitivity Analysis | Beautiful Plots")
        print("=" * 70)
        self.results = {}

    # =========================================================
    # METRIC 8: Effect of Rotation Rate (rho)
    # =========================================================
    def evaluate_rho_sensitivity(self):
        """Test different rotation rates and their effect on security and overhead."""
        print("\n" + "=" * 60)
        print("METRIC 8: Rotation Rate (ρ) Sensitivity Analysis")
        print("=" * 60)

        rho_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        results = {
            "rho": rho_values,
            "capture_rate": [],
            "rotation_overhead_ms": [],
            "blocks_to_95_refresh": []
        }

        adversary_fraction = 0.35
        num_trials = 5  # Quick test
        num_blocks = 20

        for rho in rho_values:
            print(f"\n  Testing ρ = {rho} ({num_trials} trials)...")

            capture_rates = []
            overhead_times = []

            for trial in range(num_trials):
                # Create fresh network for each trial
                net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
                fluidity = ValidatorFluidity(net, rho=rho)
                cfi = ConsensusFrictionIndex(fluidity, threshold=0.5, window=10)

                # Setup adversary and simulate
                sim = CFPCommitteeSimulator(net, fluidity, cfi)
                sim.setup_adversary(adversary_fraction)
                history = sim.simulate(num_blocks, threshold=0.5)
                analysis = sim.analyze_results(history, threshold=0.5)
                capture_rates.append(analysis["overall_capture_rate"])

                # Measure rotation overhead once per trial
                start = time.perf_counter()
                for _ in range(5):
                    fluidity.rotate_all_shards()
                overhead_times.append((time.perf_counter() - start) / 5 * 1000)

            # Average results
            avg_capture = np.mean(capture_rates) * 100
            avg_overhead = np.mean(overhead_times)
            results["capture_rate"].append(avg_capture)
            results["rotation_overhead_ms"].append(avg_overhead)

            # Calculate blocks to 95% refresh (theoretical)
            blocks_95 = int(np.ceil(np.log(0.05) / np.log(1 - rho)))
            results["blocks_to_95_refresh"].append(blocks_95)

            print(f"    Capture rate: {avg_capture:.1f}% (std: {np.std(capture_rates) * 100:.1f}%)")
            print(f"    Overhead: {avg_overhead:.1f} ms")
            print(f"    Blocks to 95% refresh: {blocks_95}")

        self.results["rho_sensitivity"] = results
        return results

    # =========================================================
    # METRIC 9: Effect of Number of Shards
    # =========================================================
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

            # Measure TPS once
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

    # =========================================================
    # METRIC 10: Capture Duration (Static vs CFP)
    # =========================================================
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

        num_experiments = 10  # More experiments for stability
        num_blocks = 50
        adversary_fraction = 0.4

        # Static system
        print(f"\n  Testing Static system ({num_experiments} experiments)...")
        for exp in range(num_experiments):
            net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
            sim = StaticCommitteeSimulator(net)
            sim.setup_adversary(adversary_fraction)

            # In static, check if captured at start - if yes, it's permanent
            captured_duration = 0
            for shard in net.shards.values():
                if sim.adversary.has_majority(shard, 0.5):
                    captured_duration = num_blocks  # Permanent capture
                    break

            results["static_durations"].append(captured_duration)

        # CFP system
        print(f"  Testing CFP system ({num_experiments} experiments)...")
        for exp in range(num_experiments):
            net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
            fluidity = ValidatorFluidity(net, rho=0.2)
            cfi = ConsensusFrictionIndex(fluidity, threshold=0.5, window=10)
            sim = CFPCommitteeSimulator(net, fluidity, cfi)
            sim.setup_adversary(adversary_fraction)

            # Track maximum consecutive capture
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
        print(
            f"    Static average duration: {results['static_avg']:.1f} blocks (std: {np.std(results['static_durations']):.1f})")
        print(
            f"    CFP average duration: {results['cfp_avg']:.1f} blocks (std: {np.std(results['cfp_durations']):.1f})")

        self.results["capture_duration"] = results
        return results

    # =========================================================
    # METRIC 11: Attack Scenarios
    # =========================================================
    def evaluate_attack_scenarios(self):
        """Test CFI under different attack strategies."""
        print("\n" + "=" * 60)
        print("METRIC 11: Attack Scenario Analysis")
        print("=" * 60)

        results = {
            "scenarios": ["No Attack", "Random\nAttack", "Coordinated\nAttack", "Adaptive\nAttack"],
            "cfi_values": [],
            "detection": []
        }

        num_rounds = 20
        adversary_fraction = 0.40  # Stronger adversary
        threshold = 0.7  # Higher threshold for better detection

        # Scenario 1: No attack (baseline)
        print("\n  Scenario 1: No Attack (Baseline)...")
        cfi_values = []
        for _ in range(3):  # Multiple trials
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

        # Scenario 2: Random attack (adversaries vote randomly)
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

        # Scenario 3: Coordinated attack (adversaries vote together)
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

        # Scenario 4: Adaptive attack (adversaries try to evade detection)
        print("\n  Scenario 4: Adaptive Attack (Evasion Attempt)...")
        cfi_values = []
        for _ in range(3):
            net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
            fluidity = ValidatorFluidity(net, rho=0.2)
            cfi_monitor = ConsensusFrictionIndex(fluidity, threshold=threshold, window=10)
            net.validator_pool.set_adversarial_fraction(adversary_fraction)

            for round_num in range(num_rounds):
                # Adaptive: alternate between coordinated and random
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

    # =========================================================
    # BEAUTIFUL PLOTS
    # =========================================================
    def plot_all(self):
        """Generate beautiful publication-quality plots."""
        print("\n" + "=" * 60)
        print("GENERATING BEAUTIFUL PLOTS")
        print("=" * 60)

        self._plot_rho_sensitivity()
        self._plot_shard_scalability()
        self._plot_capture_duration()
        self._plot_attack_scenarios()
        self._plot_summary_dashboard()

        print(f"\nAll plots saved to: {PLOT_DIR}/")

    def _plot_rho_sensitivity(self):
        """Plot rho sensitivity with dual y-axis."""
        if "rho_sensitivity" not in self.results:
            return

        data = self.results["rho_sensitivity"]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Colors
        color_capture = '#e74c3c'
        color_overhead = '#3498db'
        color_refresh = '#27ae60'

        # Primary axis: Capture rate
        x = np.array(data["rho"]) * 100  # Convert to percentage
        ax1.fill_between(x, data["capture_rate"], alpha=0.3, color=color_capture)
        line1 = ax1.plot(x, data["capture_rate"], 'o-', color=color_capture,
                         linewidth=3, markersize=10, markerfacecolor='white',
                         markeredgewidth=2, label='Capture Rate (%)')
        ax1.set_xlabel('Rotation Rate ρ (%)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Capture Rate (%)', fontsize=14, fontweight='bold', color=color_capture)
        ax1.tick_params(axis='y', labelcolor=color_capture)
        ax1.set_ylim(0, max(data["capture_rate"]) * 1.3)

        # Secondary axis: Overhead
        ax2 = ax1.twinx()
        line2 = ax2.plot(x, data["rotation_overhead_ms"], 's--', color=color_overhead,
                         linewidth=3, markersize=10, markerfacecolor='white',
                         markeredgewidth=2, label='Rotation Overhead (ms)')
        ax2.set_ylabel('Rotation Overhead (ms)', fontsize=14, fontweight='bold', color=color_overhead)
        ax2.tick_params(axis='y', labelcolor=color_overhead)

        # Add blocks to 95% as annotations
        for i, (xi, blocks) in enumerate(zip(x, data["blocks_to_95_refresh"])):
            ax1.annotate(f'{blocks} blocks\nto 95%', xy=(xi, data["capture_rate"][i]),
                         xytext=(0, 25), textcoords='offset points',
                         fontsize=9, ha='center', color=color_refresh,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3', edgecolor=color_refresh))

        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=11)

        # Title
        plt.title('Impact of Rotation Rate (ρ) on Security and Performance\n35% Adversary',
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

        # Left: Capture rate vs Shards
        ax1 = axes[0]
        colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(data["shards"])))
        bars = ax1.bar(range(len(data["shards"])), data["capture_rate"], color=colors,
                       edgecolor='white', linewidth=2)

        ax1.set_xticks(range(len(data["shards"])))
        ax1.set_xticklabels([f'{s} shards\n({v} val/shard)' for s, v in
                             zip(data["shards"], data["validators_per_shard"])], fontsize=11)
        ax1.set_ylabel('Capture Rate (%)', fontsize=14, fontweight='bold')
        ax1.set_title('Security vs Number of Shards', fontsize=14, fontweight='bold')

        # Add value labels
        for bar, val in zip(bars, data["capture_rate"]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')

        # Right: TPS vs Shards
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

    def _plot_capture_duration(self):
        """Plot capture duration comparison."""
        if "capture_duration" not in self.results:
            return

        data = self.results["capture_duration"]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create visual comparison
        categories = ['Static System\n(No Rotation)', 'CFP System\n(With Rotation)']
        values = [data["static_avg"], data["cfp_avg"]]
        colors = ['#e74c3c', '#27ae60']

        # Draw circles representing duration
        for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
            # Background circle (max possible)
            circle_bg = plt.Circle((i * 3, 0), 1.2, color='#ecf0f1', zorder=1)
            ax.add_patch(circle_bg)

            # Value circle (proportional to duration)
            radius = 0.1 + (val / 50) * 1.1  # Scale to max radius of 1.2
            circle = plt.Circle((i * 3, 0), radius, color=color, alpha=0.8, zorder=2)
            ax.add_patch(circle)

            # Duration text inside
            if val > 0:
                ax.text(i * 3, 0, f'{val:.0f}\nblocks', ha='center', va='center',
                        fontsize=14 if val < 10 else 12, fontweight='bold', color='white', zorder=3)
            else:
                ax.text(i * 3, 0, '0', ha='center', va='center',
                        fontsize=16, fontweight='bold', color='white', zorder=3)

            # Label below
            ax.text(i * 3, -1.8, cat, ha='center', va='top', fontsize=12, fontweight='bold')

        # Arrow showing improvement
        if data["static_avg"] > data["cfp_avg"]:
            improvement = ((data["static_avg"] - data["cfp_avg"]) / data["static_avg"]) * 100
            ax.annotate('', xy=(3, 0.5), xytext=(0, 0.5),
                        arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=3))
            ax.text(1.5, 1.0, f'{improvement:.0f}% Reduction\nin Capture Duration',
                    ha='center', fontsize=14, fontweight='bold', color='#27ae60',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#d5f5e3', edgecolor='#27ae60'))

        ax.set_xlim(-2, 5)
        ax.set_ylim(-2.5, 2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Capture Duration: Static vs CFP\n(40% Adversary, 50 Blocks Simulation)',
                     fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/10_capture_duration.png", dpi=300, facecolor='white')
        plt.close()
        print("  Saved: 10_capture_duration.png")

    def _plot_attack_scenarios(self):
        """Plot attack scenario comparison."""
        if "attack_scenarios" not in self.results:
            return

        data = self.results["attack_scenarios"]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Horizontal bar chart with detection indicator
        y_pos = np.arange(len(data["scenarios"]))
        colors = ['#27ae60', '#3498db', '#e74c3c', '#f39c12']

        bars = ax.barh(y_pos, data["cfi_values"], color=colors, edgecolor='white',
                       linewidth=2, height=0.6)

        # Detection threshold line
        ax.axvline(x=0.5, color='#2c3e50', linestyle='--', linewidth=2.5, zorder=5)
        ax.text(0.5, len(y_pos), 'Detection\nThreshold', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#2c3e50')

        # Add CFI values and detection status
        for i, (bar, cfi, det) in enumerate(zip(bars, data["cfi_values"], data["detection"])):
            # CFI value
            ax.text(cfi + 0.02, bar.get_y() + bar.get_height() / 2, f'{cfi:.3f}',
                    va='center', fontsize=12, fontweight='bold')

            # Detection badge
            badge_color = '#27ae60' if det == "No" else '#e74c3c'
            badge_text = '✓ Safe' if det == "No" else '⚠ Detected'
            ax.text(0.95, bar.get_y() + bar.get_height() / 2, badge_text,
                    va='center', ha='left', fontsize=11, fontweight='bold', color=badge_color)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(data["scenarios"], fontsize=12)
        ax.set_xlabel('Consensus Friction Index (CFI)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.1)
        ax.set_title('CFI Response to Different Attack Strategies\n35% Adversary',
                     fontsize=16, fontweight='bold', pad=20)

        # Add explanation box
        explanation = "Higher CFI = More diverse voting\nCFI < 0.5 triggers detection"
        ax.text(0.02, -0.5, explanation, fontsize=10, style='italic', color='gray',
                transform=ax.get_yaxis_transform())

        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/11_attack_scenarios.png", dpi=300, facecolor='white')
        plt.close()
        print("  Saved: 11_attack_scenarios.png")

    def _plot_summary_dashboard(self):
        """Create a summary dashboard with key findings."""
        fig = plt.figure(figsize=(16, 10))

        # Title
        fig.suptitle('CFP Protocol: Key Findings Summary', fontsize=20, fontweight='bold', y=0.98)

        # Create grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Box 1: Best rho
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        if "rho_sensitivity" in self.results:
            # Find optimal rho (lowest capture rate)
            data = self.results["rho_sensitivity"]
            best_idx = np.argmin(data["capture_rate"])
            best_rho = data["rho"][best_idx]

            ax1.add_patch(FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.05",
                                         facecolor='#e8f6f3', edgecolor='#1abc9c', linewidth=3))
            ax1.text(0.5, 0.75, 'Optimal ρ', ha='center', fontsize=14, fontweight='bold', color='#1abc9c')
            ax1.text(0.5, 0.45, f'{best_rho:.0%}', ha='center', fontsize=32, fontweight='bold', color='#16a085')
            ax1.text(0.5, 0.2, f'Capture: {data["capture_rate"][best_idx]:.1f}%', ha='center', fontsize=11)

        # Box 2: Security improvement
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        if "capture_duration" in self.results:
            data = self.results["capture_duration"]
            improvement = ((data["static_avg"] - data["cfp_avg"]) / data["static_avg"]) * 100 if data[
                                                                                                     "static_avg"] > 0 else 100

            ax2.add_patch(FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.05",
                                         facecolor='#fef9e7', edgecolor='#f1c40f', linewidth=3))
            ax2.text(0.5, 0.75, 'Security Gain', ha='center', fontsize=14, fontweight='bold', color='#f39c12')
            ax2.text(0.5, 0.45, f'{improvement:.0f}%', ha='center', fontsize=32, fontweight='bold', color='#d68910')
            ax2.text(0.5, 0.2, 'Less capture duration', ha='center', fontsize=11)

        # Box 3: Attack detection
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        if "attack_scenarios" in self.results:
            data = self.results["attack_scenarios"]
            detected = sum(1 for d in data["detection"] if d == "Yes")

            ax3.add_patch(FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.05",
                                         facecolor='#fdedec', edgecolor='#e74c3c', linewidth=3))
            ax3.text(0.5, 0.75, 'Attacks Detected', ha='center', fontsize=14, fontweight='bold', color='#e74c3c')
            ax3.text(0.5, 0.45, f'{detected}/{len(data["detection"])}', ha='center', fontsize=32, fontweight='bold',
                     color='#c0392b')
            ax3.text(0.5, 0.2, 'by CFI monitoring', ha='center', fontsize=11)

        # Box 4: Scalability summary
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')
        if "shard_scalability" in self.results:
            data = self.results["shard_scalability"]
            max_tps = max(data["tps"])

            ax4.add_patch(FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.05",
                                         facecolor='#eaf2f8', edgecolor='#3498db', linewidth=3))
            ax4.text(0.5, 0.75, 'Peak TPS', ha='center', fontsize=14, fontweight='bold', color='#3498db')
            ax4.text(0.5, 0.45, f'{max_tps / 1000:.0f}K', ha='center', fontsize=32, fontweight='bold', color='#2980b9')
            ax4.text(0.5, 0.2, f'with {data["shards"][np.argmax(data["tps"])]} shards', ha='center', fontsize=11)

        # Box 5: Key insight
        ax5 = fig.add_subplot(gs[1, 1:])
        ax5.axis('off')
        ax5.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.05",
                                     facecolor='#f4ecf7', edgecolor='#9b59b6', linewidth=3))
        ax5.text(0.5, 0.7, '🔑 Key Insight', ha='center', fontsize=16, fontweight='bold', color='#8e44ad')
        ax5.text(0.5, 0.35, 'CFP transforms permanent shard capture into transient events.\n'
                            'Combined with CFI monitoring, coordinated attacks are detected\n'
                            'and mitigated through adaptive rotation rate adjustment.',
                 ha='center', fontsize=12, color='#6c3483', linespacing=1.5)

        plt.savefig(f"{PLOT_DIR}/12_summary_dashboard.png", dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        print("  Saved: 12_summary_dashboard.png")

    # =========================================================
    # MAIN
    # =========================================================
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