
import time
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sharding import Network
from cfp.fluidity import ValidatorFluidity
from cfp.cfi import ConsensusFrictionIndex
from cfp.attack import StaticCommitteeSimulator, CFPCommitteeSimulator

PLOT_DIR = "evaluation_plots"
os.makedirs(PLOT_DIR, exist_ok=True)


class AdvancedCFPEvaluator:
    """Advanced evaluation with attack scenarios and sensitivity analysis."""

    def __init__(self):
        print("=" * 70)
        print("   CFP ADVANCED SCIENTIFIC EVALUATION")
        print("   Attack Scenarios | Sensitivity Analysis")
        print("=" * 70)
        self.results = {}

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
        num_trials = 50
        num_blocks = 20

        for rho in rho_values:
            print(f"\n  Testing ρ = {rho} ({num_trials} trials)...")

            capture_rates = []
            overhead_times = []

            for trial in range(num_trials):
                net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
                fluidity = ValidatorFluidity(net, rho=rho)
                cfi = ConsensusFrictionIndex(fluidity, threshold=0.65, window=10)

                sim = CFPCommitteeSimulator(net, fluidity, cfi)
                sim.setup_adversary(adversary_fraction)
                history = sim.simulate(num_blocks, threshold=0.5)
                analysis = sim.analyze_results(history, threshold=0.5)
                capture_rates.append(analysis["overall_capture_rate"])

                start = time.perf_counter()
                for _ in range(15):
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
                cfi = ConsensusFrictionIndex(fluidity, threshold=0.65, window=10)

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
        adversary_fraction = 0.35

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
            cfi = ConsensusFrictionIndex(fluidity, threshold=0.65, window=10)
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
            "scenarios": ["No Attack", "Random\nAttack", "Coordinated\nAttack", "Adaptive\nAttack"],
            "cfi_values": [],
            "detection": []
        }

        num_rounds = 20
        adversary_fraction = 0.35
        threshold = 0.65

        # Scenario 1: No attack
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

        # Scenario 2: Random attack
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

        # Scenario 3: Coordinated attack
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

        # Scenario 4: Adaptive attack
        print("\n  Scenario 4: Adaptive Attack...")
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
        """Generate publication-quality plots"""
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = False
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['font.weight'] = 'normal'

        # Plot 8: Rho Sensitivity
        if "rho_sensitivity" in self.results:
            data = self.results["rho_sensitivity"]
            fig, ax1 = plt.subplots(figsize=(9, 5))

            rhos = [r * 100 for r in data["rho"]]
            capture = data["capture_rate"]
            overhead = data["rotation_overhead_ms"]
            blocks_95 = data["blocks_to_95_refresh"]

            ax1.fill_between(rhos, capture, alpha=0.2, color='#e74c3c')
            ax1.plot(rhos, capture, 'o-', color='#c0392b', linewidth=2, markersize=7,
                     markerfacecolor='white', markeredgewidth=1.5)
            ax1.set_xlabel('Rotation Rate ρ (%)')
            ax1.set_ylabel('Capture Rate (%)', color='#c0392b')
            ax1.tick_params(axis='y', labelcolor='#c0392b')

            ax2 = ax1.twinx()
            ax2.plot(rhos, overhead, 's--', color='#2980b9', linewidth=2, markersize=7,
                     markerfacecolor='white', markeredgewidth=1.5)
            ax2.set_ylabel('Rotation Overhead (ms)', color='#2980b9')
            ax2.tick_params(axis='y', labelcolor='#2980b9')

            for r, c, b in zip(rhos, capture, blocks_95):
                ax1.annotate(f'{b} blk', xy=(r, c), xytext=(0, 10),
                             textcoords='offset points', fontsize=8, ha='center', color='#7f8c8d')

            ax1.spines['top'].set_visible(False)
            ax2.spines['top'].set_visible(False)

            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/08_rho_sensitivity.png", dpi=300, facecolor='white')
            plt.close()
            print("  Saved: 08_rho_sensitivity.png")

        # Plot 9: Shard Scalability
        if "shard_scalability" in self.results:
            data = self.results["shard_scalability"]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

            shards = data["shards"]
            capture = data["capture_rate"]
            tps = [t / 1000 for t in data["tps"]]
            validators = data["validators_per_shard"]

            colors_red = ['#fadbd8', '#f1948a', '#e74c3c', '#922b21']
            bars1 = ax1.bar(range(len(shards)), capture, color=colors_red, width=0.6, edgecolor='white')
            ax1.set_xticks(range(len(shards)))
            ax1.set_xticklabels([f'{s} shards\n({v} val)' for s, v in zip(shards, validators)])
            ax1.set_ylabel('Capture Rate (%)')
            for bar, val in zip(bars1, capture):
                ax1.text(bar.get_x() + bar.get_width() / 2, val + 2, f'{val:.1f}%',
                         ha='center', fontsize=9)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

            colors_blue = ['#d6eaf8', '#85c1e9', '#3498db', '#1a5276']
            bars2 = ax2.bar(range(len(shards)), tps, color=colors_blue, width=0.6, edgecolor='white')
            ax2.set_xticks(range(len(shards)))
            ax2.set_xticklabels([f'{s} shards' for s in shards])
            ax2.set_ylabel('Throughput (K TPS)')
            for bar, val in zip(bars2, tps):
                ax2.text(bar.get_x() + bar.get_width() / 2, val + 1, f'{val:.1f}K',
                         ha='center', fontsize=9)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/09_shard_scalability.png", dpi=300, facecolor='white')
            plt.close()
            print("  Saved: 09_shard_scalability.png")

        # Plot 10: Capture Duration - Circles
        if "capture_duration" in self.results:
            data = self.results["capture_duration"]
            fig, ax = plt.subplots(figsize=(10, 5))

            static_dur = data["static_avg"]
            cfp_dur = data["cfp_avg"]
            max_dur = max(static_dur, cfp_dur)

            # Static circle
            circle1_bg = plt.Circle((-1.5, 0), 1.2, color='#fadbd8', zorder=1)
            radius1 = 0.4 + (static_dur / max_dur) * 0.7
            circle1 = plt.Circle((-1.5, 0), radius1, color='#e74c3c', alpha=0.85, zorder=2)
            ax.add_patch(circle1_bg)
            ax.add_patch(circle1)
            ax.text(-1.5, 0.05, f'{static_dur:.0f}', ha='center', va='center',
                    fontsize=20, color='white', fontweight='bold', zorder=3)
            ax.text(-1.5, -0.3, 'blocks', ha='center', va='center',
                    fontsize=10, color='white', zorder=3)
            ax.text(-1.5, -1.6, 'Static System', ha='center', fontsize=11, color='#2c3e50')

            # CFP circle
            circle2_bg = plt.Circle((1.5, 0), 1.2, color='#d5f5e3', zorder=1)
            radius2 = 0.4 + (cfp_dur / max_dur) * 0.7
            circle2 = plt.Circle((1.5, 0), radius2, color='#27ae60', alpha=0.85, zorder=2)
            ax.add_patch(circle2_bg)
            ax.add_patch(circle2)
            ax.text(1.5, 0.05, f'{cfp_dur:.0f}', ha='center', va='center',
                    fontsize=20, color='white', fontweight='bold', zorder=3)
            ax.text(1.5, -0.3, 'blocks', ha='center', va='center',
                    fontsize=10, color='white', zorder=3)
            ax.text(1.5, -1.6, 'CFP System', ha='center', fontsize=11, color='#2c3e50')

            # Arrow
            ax.annotate('', xy=(0.3, 0), xytext=(-0.3, 0),
                        arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

            ax.set_xlim(-3.2, 3.2)
            ax.set_ylim(-2.2, 1.8)
            ax.set_aspect('equal')
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/10_capture_duration.png", dpi=300, facecolor='white')
            plt.close()
            print("  Saved: 10_capture_duration.png")

        # Plot 11: Attack Scenarios
        if "attack_scenarios" in self.results:
            data = self.results["attack_scenarios"]
            fig, ax = plt.subplots(figsize=(9, 5))

            scenarios = data["scenarios"]
            cfis = data["cfi_values"]
            detected = data["detection"]

            y_pos = np.arange(len(scenarios))
            colors = ['#27ae60' if d == "No" else '#e74c3c' for d in detected]

            bars = ax.barh(y_pos, cfis, color=colors, height=0.5, edgecolor='white')
            ax.axvline(x=0.65, color='#7f8c8d', linestyle='--', linewidth=1.2, alpha=0.7)

            for bar, val, det in zip(bars, cfis, detected):
                marker = 'Safe' if det == "No" else 'Detected'
                ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f} ({marker})', va='center', fontsize=9)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(scenarios)
            ax.set_xlabel('Consensus Friction Index (CFI)')
            ax.set_xlim(0, 1.1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/11_attack_scenarios.png", dpi=300, facecolor='white')
            plt.close()
            print("  Saved: 11_attack_scenarios.png")

        print(f"\nAll plots saved to: {PLOT_DIR}/")

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
