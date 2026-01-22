"""
CFP Publication-Ready Evaluation
Creative plots + LaTeX tables for paper
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
from cfp.benchmark import ThroughputBenchmark

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects
import matplotlib

matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5

# Output directories
PLOT_DIR = "publication_plots"
os.makedirs(PLOT_DIR, exist_ok=True)


class PublicationEvaluator:
    """Publication-ready evaluation with creative visualizations."""
    
    def __init__(self):
        print("=" * 70)
        print("   CFP PUBLICATION-READY EVALUATION")
        print("   Creative Plots + LaTeX Tables")
        print("=" * 70)
        self.results = {}
    
    # ===========================================
    # EVALUATION METHODS
    # ===========================================
    
    def evaluate_all(self):
        """Run all evaluations."""
        self.evaluate_rho_sensitivity()
        self.evaluate_shard_scalability()
        self.evaluate_capture_duration()
        self.evaluate_attack_scenarios()
        self.evaluate_cross_shard()
        self.evaluate_performance()
    
    def evaluate_rho_sensitivity(self):
        """Metric 8: Effect of rotation rate."""
        print("\n" + "=" * 60)
        print("METRIC 8: Rotation Rate (ρ) Sensitivity")
        print("=" * 60)
        
        rho_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        results = {"rho": rho_values, "capture_rate": [], "overhead_ms": [], 
                   "blocks_95": [], "std": []}
        
        for rho in rho_values:
            print(f"  Testing ρ = {rho}...")
            captures = []
            overheads = []
            
            for _ in range(5):
                net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
                fluidity = ValidatorFluidity(net, rho=rho)
                cfi = ConsensusFrictionIndex(fluidity, threshold=0.5, window=10)
                sim = CFPCommitteeSimulator(net, fluidity, cfi)
                sim.setup_adversary(0.35)
                history = sim.simulate(20, threshold=0.5)
                analysis = sim.analyze_results(history, threshold=0.5)
                captures.append(analysis["overall_capture_rate"] * 100)
                
                start = time.perf_counter()
                for _ in range(5):
                    fluidity.rotate_all_shards()
                overheads.append((time.perf_counter() - start) / 5 * 1000)
            
            results["capture_rate"].append(np.mean(captures))
            results["std"].append(np.std(captures))
            results["overhead_ms"].append(np.mean(overheads))
            results["blocks_95"].append(int(np.ceil(np.log(0.05) / np.log(1 - rho))))
        
        self.results["rho"] = results
        return results
    
    def evaluate_shard_scalability(self):
        """Metric 9: Effect of shard count."""
        print("\n" + "=" * 60)
        print("METRIC 9: Shard Scalability")
        print("=" * 60)
        
        configs = [(2, 50), (4, 25), (8, 12), (16, 6)]
        results = {"shards": [], "validators": [], "capture_rate": [], "tps": []}
        
        for num_shards, val_per_shard in configs:
            print(f"  Testing {num_shards} shards...")
            captures = []
            
            for _ in range(5):
                net = Network(num_shards=num_shards, validators_per_shard=val_per_shard, 
                             total_validators=200)
                fluidity = ValidatorFluidity(net, rho=0.2)
                cfi = ConsensusFrictionIndex(fluidity, threshold=0.5, window=10)
                sim = CFPCommitteeSimulator(net, fluidity, cfi)
                sim.setup_adversary(0.35)
                history = sim.simulate(30, threshold=0.5)
                analysis = sim.analyze_results(history, threshold=0.5)
                captures.append(analysis["overall_capture_rate"] * 100)
            
            net = Network(num_shards=num_shards, validators_per_shard=val_per_shard, 
                         total_validators=200)
            bench = ThroughputBenchmark(net)
            tps = bench.run(100)["throughput_tps"]
            
            results["shards"].append(num_shards)
            results["validators"].append(val_per_shard)
            results["capture_rate"].append(np.mean(captures))
            results["tps"].append(tps)
        
        self.results["shards"] = results
        return results
    
    def evaluate_capture_duration(self):
        """Metric 10: Capture duration comparison."""
        print("\n" + "=" * 60)
        print("METRIC 10: Capture Duration")
        print("=" * 60)
        
        results = {"static": [], "cfp": []}
        
        print("  Testing Static...")
        for _ in range(10):
            net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
            sim = StaticCommitteeSimulator(net)
            sim.setup_adversary(0.4)
            captured = any(sim.adversary.has_majority(s, 0.5) for s in net.shards.values())
            results["static"].append(50 if captured else 0)
        
        print("  Testing CFP...")
        for _ in range(10):
            net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
            fluidity = ValidatorFluidity(net, rho=0.2)
            cfi = ConsensusFrictionIndex(fluidity, threshold=0.5, window=10)
            sim = CFPCommitteeSimulator(net, fluidity, cfi)
            sim.setup_adversary(0.4)
            
            streak, max_streak = 0, 0
            for _ in range(50):
                fluidity.rotate_all_shards()
                captured = any(sim.adversary.has_majority(s, 0.5) for s in net.shards.values())
                if captured:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
            results["cfp"].append(max_streak)
        
        results["static_avg"] = np.mean(results["static"])
        results["cfp_avg"] = np.mean(results["cfp"])
        results["improvement"] = (results["static_avg"] - results["cfp_avg"]) / results["static_avg"] * 100
        
        self.results["duration"] = results
        return results
    
    def evaluate_attack_scenarios(self):
        """Metric 11: Attack detection."""
        print("\n" + "=" * 60)
        print("METRIC 11: Attack Scenarios")
        print("=" * 60)
        
        scenarios = ["No Attack", "Random", "Coordinated", "Adaptive"]
        results = {"scenarios": scenarios, "cfi": [], "detected": []}
        threshold = 0.7
        
        for i, scenario in enumerate(scenarios):
            print(f"  Testing {scenario}...")
            cfi_vals = []
            
            for _ in range(3):
                net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
                fluidity = ValidatorFluidity(net, rho=0.2)
                cfi_mon = ConsensusFrictionIndex(fluidity, threshold=threshold, window=10)
                
                if i > 0:
                    net.validator_pool.set_adversarial_fraction(0.4)
                
                for r in range(20):
                    if i == 0:
                        coord = False
                    elif i == 1:
                        coord = False
                    elif i == 2:
                        coord = True
                    else:
                        coord = (r % 3 == 0)
                    
                    cfi_mon.simulate_voting_round(net, adversarial_coordination=coord)
                    for shard in net.shards.values():
                        cfi_vals.append(cfi_mon.calculate_cfi(shard))
            
            avg_cfi = np.mean(cfi_vals)
            results["cfi"].append(avg_cfi)
            results["detected"].append(avg_cfi < threshold)
        
        self.results["attacks"] = results
        return results
    
    def evaluate_cross_shard(self):
        """Metric: Cross-shard transactions."""
        print("\n" + "=" * 60)
        print("METRIC: Cross-Shard Transactions")
        print("=" * 60)
        
        net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
        coord = CrossShardCoordinator(net)
        
        # Register accounts to different shards
        for i in range(200):
            net.register_account(f"acc_{i}", i % 4)
        
        success, fail, latencies = 0, 0, []
        for i in range(50):
            src = f"acc_{i}"
            # Make sure dst is in a different shard
            dst_shard = (i % 4 + 1) % 4
            dst = f"acc_{dst_shard}"  # Account in different shard
            
            start = time.perf_counter()
            tx, result = coord.process_transaction(src, dst, 10)
            latencies.append((time.perf_counter() - start) * 1000)
            
            if tx is not None:
                success += 1
            else:
                fail += 1
        
        total = success + fail
        self.results["cross_shard"] = {
            "success_rate": (success / total * 100) if total > 0 else 0,
            "avg_latency": np.mean(latencies) if latencies else 0,
            "total": total
        }
        print(f"  Success rate: {self.results['cross_shard']['success_rate']:.1f}%")
        print(f"  Avg latency: {self.results['cross_shard']['avg_latency']:.1f} ms")
        return self.results["cross_shard"]
    
    def evaluate_performance(self):
        """Metric: System performance."""
        print("\n" + "=" * 60)
        print("METRIC: Performance")
        print("=" * 60)
        
        net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
        bench = ThroughputBenchmark(net)
        tps = bench.run(200)["throughput_tps"]
        
        fluidity = ValidatorFluidity(net, rho=0.2)
        times = []
        for _ in range(20):
            start = time.perf_counter()
            fluidity.rotate_all_shards()
            times.append((time.perf_counter() - start) * 1000)
        
        self.results["performance"] = {
            "tps": tps,
            "rotation_ms": np.mean(times)
        }
        return self.results["performance"]
    
    # ===========================================
    # CREATIVE PLOTS
    # ===========================================
    
    def generate_all_plots(self):
        """Generate all publication plots."""
        print("\n" + "=" * 60)
        print("GENERATING CREATIVE PLOTS")
        print("=" * 60)
        
        self._plot_rho_radar()
        self._plot_shard_security_gradient()
        self._plot_capture_timeline()
        self._plot_attack_shield()
        self._plot_performance_gauge()
        self._plot_comparison_infographic()
        
        print(f"\nAll plots saved to: {PLOT_DIR}/")
    
    def _plot_rho_radar(self):
        """Radar chart for rho analysis."""
        if "rho" not in self.results:
            return
        
        data = self.results["rho"]
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        categories = [f'ρ={r}' for r in data["rho"]]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Normalize values
        security = [100 - c for c in data["capture_rate"]]
        security += security[:1]
        
        speed = [100 - min(o/6, 100) for o in data["overhead_ms"]]
        speed += speed[:1]
        
        refresh = [100 - min(b/0.6, 100) for b in data["blocks_95"]]
        refresh += refresh[:1]
        
        ax.plot(angles, security, 'o-', linewidth=2.5, label='Security', color='#e74c3c')
        ax.fill(angles, security, alpha=0.25, color='#e74c3c')
        
        ax.plot(angles, speed, 's-', linewidth=2.5, label='Speed', color='#3498db')
        ax.fill(angles, speed, alpha=0.25, color='#3498db')
        
        ax.plot(angles, refresh, '^-', linewidth=2.5, label='Refresh Rate', color='#27ae60')
        ax.fill(angles, refresh, alpha=0.25, color='#27ae60')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12, fontweight='bold')
        ax.set_ylim(0, 100)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
        plt.title('Rotation Rate (ρ) Trade-off Analysis', size=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/01_rho_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 01_rho_radar")
    
    def _plot_shard_security_gradient(self):
        """Gradient heatmap for shard scalability."""
        if "shards" not in self.results:
            return
        
        data = self.results["shards"]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(data["shards"]))
        width = 0.8
        
        # Create gradient bars
        for i, (shards, val, cap, tps) in enumerate(zip(
            data["shards"], data["validators"], data["capture_rate"], data["tps"]
        )):
            # Security gradient: green (safe) to red (danger)
            color_intensity = cap / 100
            color = (color_intensity, 1 - color_intensity * 0.7, 0.2)
            
            bar = ax.bar(i, cap, width, color=color, edgecolor='black', linewidth=2)
            
            # Add shard icon on top
            ax.text(i, cap + 3, f'🔲 × {shards}', ha='center', fontsize=14)
            ax.text(i, -8, f'{val} validators\nper shard', ha='center', fontsize=10, 
                   style='italic', color='gray')
        
        # Add danger zone
        ax.axhspan(66, 105, alpha=0.15, color='red', label='Danger Zone (>66%)')
        ax.axhspan(33, 66, alpha=0.15, color='orange', label='Warning Zone')
        ax.axhspan(0, 33, alpha=0.15, color='green', label='Safe Zone (<33%)')
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s} Shards' for s in data["shards"]], fontsize=12, fontweight='bold')
        ax.set_ylabel('Capture Rate (%)', fontsize=14, fontweight='bold')
        ax.set_ylim(-15, 105)
        ax.set_title('Shard Count vs Security Trade-off\n(More Shards = Less Validators = Higher Risk)', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.legend(loc='upper left', fontsize=10)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/02_shard_security.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 02_shard_security")
    
    def _plot_capture_timeline(self):
        """Timeline comparison of capture duration."""
        if "duration" not in self.results:
            return
        
        data = self.results["duration"]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Timeline for Static
        static_y = 0.7
        ax.arrow(0, static_y, data["static_avg"], 0, head_width=0.08, head_length=1,
                fc='#e74c3c', ec='#c0392b', linewidth=3)
        ax.text(data["static_avg"]/2, static_y + 0.15, f'Static: {data["static_avg"]:.0f} blocks',
               ha='center', fontsize=14, fontweight='bold', color='#c0392b')
        
        # Danger symbols
        for i in range(0, int(data["static_avg"]), 5):
            ax.text(i, static_y - 0.12, '⚠', ha='center', fontsize=12, color='#e74c3c')
        
        # Timeline for CFP
        cfp_y = 0.3
        ax.arrow(0, cfp_y, data["cfp_avg"], 0, head_width=0.08, head_length=0.5,
                fc='#27ae60', ec='#1e8449', linewidth=3)
        ax.text(data["cfp_avg"]/2, cfp_y + 0.15, f'CFP: {data["cfp_avg"]:.0f} blocks',
               ha='center', fontsize=14, fontweight='bold', color='#1e8449')
        
        # Safe symbols
        for i in range(0, int(data["cfp_avg"]), 2):
            ax.text(i, cfp_y - 0.12, '✓', ha='center', fontsize=12, color='#27ae60')
        
        # Improvement annotation
        improvement = data["improvement"]
        mid_x = max(data["static_avg"], data["cfp_avg"]) + 5
        ax.annotate('', xy=(mid_x, cfp_y), xytext=(mid_x, static_y),
                   arrowprops=dict(arrowstyle='<->', color='#8e44ad', lw=3))
        ax.text(mid_x + 2, 0.5, f'{improvement:.0f}%\nImprovement',
               ha='left', va='center', fontsize=14, fontweight='bold', color='#8e44ad')
        
        ax.set_xlim(-2, mid_x + 15)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Blocks (Time →)', fontsize=14, fontweight='bold')
        ax.set_yticks([0.3, 0.7])
        ax.set_yticklabels(['CFP\n(With Rotation)', 'Static\n(No Rotation)'], fontsize=12)
        ax.set_title('Capture Duration: How Long Can Adversary Maintain Control?',
                    fontsize=16, fontweight='bold', pad=15)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/03_capture_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 03_capture_timeline")
    
    def _plot_attack_shield(self):
        """Shield visualization for attack detection."""
        if "attacks" not in self.results:
            return
        
        data = self.results["attacks"]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw shield shape
        shield_x = [0.5, 0.1, 0.1, 0.5, 0.9, 0.9, 0.5]
        shield_y = [0.95, 0.75, 0.3, 0.05, 0.3, 0.75, 0.95]
        ax.fill(shield_x, shield_y, color='#ecf0f1', edgecolor='#2c3e50', linewidth=4)
        
        # Title on shield
        ax.text(0.5, 0.85, 'CFI', ha='center', va='center', fontsize=32, 
               fontweight='bold', color='#2c3e50')
        ax.text(0.5, 0.75, 'Attack Detection', ha='center', va='center', fontsize=14,
               fontweight='bold', color='#7f8c8d')
        
        # Attack results
        y_positions = [0.6, 0.48, 0.36, 0.24]
        
        for i, (scenario, cfi, detected) in enumerate(zip(
            data["scenarios"], data["cfi"], data["detected"]
        )):
            y = y_positions[i]
            
            # Status icon
            if detected:
                icon = '🛡'  # Shield blocked
                color = '#27ae60'
                status = 'BLOCKED'
            else:
                icon = '✓'
                color = '#3498db'
                status = 'ALLOWED'
            
            ax.text(0.25, y, icon, ha='center', va='center', fontsize=16)
            ax.text(0.5, y, scenario, ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(0.75, y, f'CFI: {cfi:.2f}', ha='center', va='center', fontsize=10,
                   color=color, fontweight='bold')
        
        # Legend outside shield
        ax.text(0.5, -0.05, 'Threshold: CFI < 0.7 triggers detection', ha='center',
               fontsize=11, style='italic', color='#7f8c8d')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('CFI-Based Attack Detection System', fontsize=16, fontweight='bold', pad=10)
        
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/04_attack_shield.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 04_attack_shield")
    
    def _plot_performance_gauge(self):
        """Gauge-style performance metrics."""
        if "performance" not in self.results:
            return
        
        data = self.results["performance"]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # TPS Gauge
        ax1 = axes[0]
        tps = data["tps"]
        
        # Draw arc
        theta = np.linspace(0.75 * np.pi, 0.25 * np.pi, 100)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax1.plot(x, y, 'k-', linewidth=8)
        
        # Colored segments
        colors = ['#27ae60', '#f1c40f', '#e74c3c']
        for i, (start, end, c) in enumerate([(0.75, 0.58, colors[0]), 
                                              (0.58, 0.42, colors[1]),
                                              (0.42, 0.25, colors[2])]):
            t = np.linspace(start * np.pi, end * np.pi, 30)
            ax1.plot(np.cos(t), np.sin(t), '-', linewidth=20, color=c, solid_capstyle='butt')
        
        # Needle
        tps_norm = min(tps / 200000, 1)  # Normalize to 0-1
        needle_angle = (0.75 - tps_norm * 0.5) * np.pi
        ax1.arrow(0, 0, 0.7 * np.cos(needle_angle), 0.7 * np.sin(needle_angle),
                 head_width=0.08, head_length=0.05, fc='#2c3e50', ec='#2c3e50')
        ax1.plot(0, 0, 'ko', markersize=15)
        
        ax1.text(0, -0.3, f'{tps:,.0f}', ha='center', fontsize=28, fontweight='bold', color='#2c3e50')
        ax1.text(0, -0.5, 'TPS', ha='center', fontsize=16, color='#7f8c8d')
        
        ax1.set_xlim(-1.3, 1.3)
        ax1.set_ylim(-0.7, 1.3)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title('Throughput', fontsize=16, fontweight='bold')
        
        # Rotation Overhead Gauge
        ax2 = axes[1]
        overhead = data["rotation_ms"]
        
        # Draw arc
        ax2.plot(x, y, 'k-', linewidth=8)
        
        # Colored segments (reversed - low is good)
        for i, (start, end, c) in enumerate([(0.75, 0.58, colors[0]), 
                                              (0.58, 0.42, colors[1]),
                                              (0.42, 0.25, colors[2])]):
            t = np.linspace(start * np.pi, end * np.pi, 30)
            ax2.plot(np.cos(t), np.sin(t), '-', linewidth=20, color=c, solid_capstyle='butt')
        
        # Needle
        overhead_norm = min(overhead / 1000, 1)
        needle_angle = (0.75 - overhead_norm * 0.5) * np.pi
        ax2.arrow(0, 0, 0.7 * np.cos(needle_angle), 0.7 * np.sin(needle_angle),
                 head_width=0.08, head_length=0.05, fc='#2c3e50', ec='#2c3e50')
        ax2.plot(0, 0, 'ko', markersize=15)
        
        ax2.text(0, -0.3, f'{overhead:.0f}', ha='center', fontsize=28, fontweight='bold', color='#2c3e50')
        ax2.text(0, -0.5, 'ms per rotation', ha='center', fontsize=16, color='#7f8c8d')
        
        ax2.set_xlim(-1.3, 1.3)
        ax2.set_ylim(-0.7, 1.3)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title('Rotation Overhead', fontsize=16, fontweight='bold')
        
        plt.suptitle('System Performance Metrics', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/05_performance_gauge.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 05_performance_gauge")
    
    def _plot_comparison_infographic(self):
        """Infographic comparing Static vs CFP."""
        if "duration" not in self.results or "shards" not in self.results:
            return
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Background
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)
        
        # Left side: Static System
        left_box = FancyBboxPatch((0.5, 1), 7, 8, boxstyle="round,pad=0.1",
                                   facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=3)
        ax.add_patch(left_box)
        
        ax.text(4, 8.5, 'Static System', ha='center', fontsize=20, fontweight='bold', color='#c0392b')
        ax.text(4, 7.5, '(No Rotation)', ha='center', fontsize=14, color='#7f8c8d')
        
        ax.text(4, 6.3, '❌ Permanent Capture', ha='center', fontsize=14, color='#e74c3c')
        ax.text(4, 5.5, '❌ No Attack Detection', ha='center', fontsize=14, color='#e74c3c')
        ax.text(4, 4.7, '❌ Fixed Committees', ha='center', fontsize=14, color='#e74c3c')
        
        ax.text(4, 3.2, f'Capture Duration:', ha='center', fontsize=12, fontweight='bold')
        ax.text(4, 2.5, f'{self.results["duration"]["static_avg"]:.0f} blocks', 
               ha='center', fontsize=24, fontweight='bold', color='#c0392b')
        
        # Right side: CFP System
        right_box = FancyBboxPatch((8.5, 1), 7, 8, boxstyle="round,pad=0.1",
                                    facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=3)
        ax.add_patch(right_box)
        
        ax.text(12, 8.5, 'CFP System', ha='center', fontsize=20, fontweight='bold', color='#1e8449')
        ax.text(12, 7.5, '(With Rotation)', ha='center', fontsize=14, color='#7f8c8d')
        
        ax.text(12, 6.3, '✓ Transient Capture Only', ha='center', fontsize=14, color='#27ae60')
        ax.text(12, 5.5, '✓ CFI Attack Detection', ha='center', fontsize=14, color='#27ae60')
        ax.text(12, 4.7, '✓ Dynamic Committees', ha='center', fontsize=14, color='#27ae60')
        
        ax.text(12, 3.2, f'Capture Duration:', ha='center', fontsize=12, fontweight='bold')
        ax.text(12, 2.5, f'{self.results["duration"]["cfp_avg"]:.0f} blocks', 
               ha='center', fontsize=24, fontweight='bold', color='#1e8449')
        
        # Center: Improvement
        center_box = FancyBboxPatch((6.5, 3.5), 3, 3, boxstyle="round,pad=0.1",
                                     facecolor='#f9e79f', edgecolor='#f1c40f', linewidth=3)
        ax.add_patch(center_box)
        
        ax.text(8, 5.8, 'Improvement', ha='center', fontsize=14, fontweight='bold', color='#9a7d0a')
        ax.text(8, 4.5, f'{self.results["duration"]["improvement"]:.0f}%', 
               ha='center', fontsize=32, fontweight='bold', color='#b7950b')
        
        # Arrows
        ax.annotate('', xy=(6.5, 5), xytext=(5.5, 5),
                   arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
        ax.annotate('', xy=(10.5, 5), xytext=(9.5, 5),
                   arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
        
        ax.axis('off')
        ax.set_title('Static vs CFP: Security Comparison', fontsize=22, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/06_comparison_infographic.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 06_comparison_infographic")
    
 
    
    def run(self):
        """Run complete evaluation."""
        self.evaluate_all()
        self.generate_all_plots()
        
        print("\n" + "=" * 70)
        print("   PUBLICATION EVALUATION COMPLETE")
        print(f"   Plots: {PLOT_DIR}/")
        print("=" * 70)


if __name__ == "__main__":
    evaluator = PublicationEvaluator()
    evaluator.run()
