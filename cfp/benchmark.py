"""
Benchmarking module for CFP simulation.
Measures performance metrics for the sharded blockchain.

Metrics:
- Throughput: transactions per second
- Latency: time per transaction
- Scalability: performance vs number of shards
- Rotation overhead: cost of validator rotation
- Cross-shard ratio: fraction of cross-shard transactions
"""

import time
import random
import statistics
from collections import defaultdict


class TransactionGenerator:
    """
    Generates realistic transaction workloads for benchmarking.
    """
    
    def __init__(self, accounts, cross_shard_ratio=0.75):
        """
        Setup generator.
        
        accounts: list of account IDs
        cross_shard_ratio: target ratio of cross-shard transactions
        """
        self.accounts = accounts
        self.cross_shard_ratio = cross_shard_ratio
    
    def generate_batch(self, count):
        """Generate a batch of random transactions."""
        transactions = []
        for _ in range(count):
            sender = random.choice(self.accounts)
            receiver = random.choice([a for a in self.accounts if a != sender])
            amount = random.randint(1, 100)
            transactions.append({
                "sender": sender,
                "receiver": receiver,
                "amount": amount
            })
        return transactions


class LatencyTracker:
    """
    Tracks latency of operations.
    """
    
    def __init__(self):
        self.measurements = defaultdict(list)
    
    def start(self, operation):
        """Start timing an operation."""
        return time.perf_counter()
    
    def end(self, operation, start_time):
        """End timing and record."""
        elapsed = time.perf_counter() - start_time
        self.measurements[operation].append(elapsed)
        return elapsed
    
    def get_stats(self, operation):
        """Get statistics for an operation."""
        data = self.measurements.get(operation, [])
        if not data:
            return None
        
        return {
            "count": len(data),
            "total": sum(data),
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "min": min(data),
            "max": max(data),
            "stdev": statistics.stdev(data) if len(data) > 1 else 0
        }
    
    def get_all_stats(self):
        """Get statistics for all operations."""
        return {op: self.get_stats(op) for op in self.measurements}


class ThroughputBenchmark:
    """
    Measures transaction throughput.
    """
    
    def __init__(self, network):
        self.network = network
        self.results = []
    
    def run(self, num_transactions, batch_size=100):
        """
        Run throughput benchmark.
        
        Processes transactions and measures how many per second.
        """
        from core import Transaction
        
        accounts = [f"user_{i}" for i in range(200)]
        for acc in accounts:
            if acc not in self.network.account_to_shard:
                self.network.register_account(acc)
        
        generator = TransactionGenerator(accounts)
        
        # Generate all transactions
        all_txs = generator.generate_batch(num_transactions)
        
        # Process in batches and measure time
        start_time = time.perf_counter()
        processed = 0
        
        for i in range(0, len(all_txs), batch_size):
            batch = all_txs[i:i+batch_size]
            
            for tx_data in batch:
                # Route to correct shard
                shard_id = self.network.get_account_shard(tx_data["sender"])
                if shard_id is not None:
                    shard = self.network.shards[shard_id]
                    # Create proper Transaction object
                    tx = Transaction(
                        sender=tx_data["sender"],
                        receiver=tx_data["receiver"],
                        amount=tx_data["amount"]
                    )
                    shard.pending_txs.append(tx)
                    processed += 1
            
            # Process block for each shard
            for shard in self.network.shards.values():
                if shard.pending_txs:
                    shard.process_block()
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        throughput = processed / elapsed if elapsed > 0 else 0
        
        result = {
            "transactions": processed,
            "time_seconds": elapsed,
            "throughput_tps": throughput,
            "batch_size": batch_size
        }
        
        self.results.append(result)
        return result


class ScalabilityBenchmark:
    """
    Measures how performance scales with number of shards.
    """
    
    def __init__(self):
        self.results = []
    
    def run(self, shard_counts, transactions_per_test=1000):
        """
        Run scalability benchmark.
        
        Tests throughput with different numbers of shards.
        """
        from sharding import Network
        
        for num_shards in shard_counts:
            # Create network with this many shards
            validators_per_shard = 25
            total_validators = num_shards * validators_per_shard * 2
            
            network = Network(
                num_shards=num_shards,
                validators_per_shard=validators_per_shard,
                total_validators=total_validators
            )
            
            # Register accounts
            accounts = [f"user_{i}" for i in range(num_shards * 50)]
            for acc in accounts:
                network.register_account(acc)
            
            # Run throughput test
            benchmark = ThroughputBenchmark(network)
            result = benchmark.run(transactions_per_test)
            
            result["num_shards"] = num_shards
            result["validators_per_shard"] = validators_per_shard
            result["total_validators"] = len(network.validator_pool.validators)
            
            self.results.append(result)
        
        return self.results
    
    def print_results(self):
        """Print scalability results."""
        print("\nScalability Benchmark Results:")
        print("-" * 60)
        print(f"{'Shards':<10} {'TPS':<15} {'Time (s)':<15} {'Transactions':<15}")
        print("-" * 60)
        
        for r in self.results:
            print(f"{r['num_shards']:<10} {r['throughput_tps']:<15.1f} "
                  f"{r['time_seconds']:<15.3f} {r['transactions']:<15}")


class RotationOverheadBenchmark:
    """
    Measures the overhead of validator rotation.
    """
    
    def __init__(self, network, fluidity_manager):
        self.network = network
        self.fluidity = fluidity_manager
        self.results = []
    
    def run(self, num_rotations=100):
        """
        Measure time taken for validator rotations.
        """
        tracker = LatencyTracker()
        
        for i in range(num_rotations):
            start = tracker.start("rotation")
            self.fluidity.rotate_all_shards()
            tracker.end("rotation", start)
        
        stats = tracker.get_stats("rotation")
        
        result = {
            "num_rotations": num_rotations,
            "total_time": stats["total"],
            "mean_time": stats["mean"],
            "median_time": stats["median"],
            "min_time": stats["min"],
            "max_time": stats["max"]
        }
        
        self.results.append(result)
        return result


class CrossShardBenchmark:
    """
    Measures cross-shard transaction performance.
    """
    
    def __init__(self, network, coordinator):
        self.network = network
        self.coordinator = coordinator
        self.results = []
    
    def run(self, num_transactions=100):
        """
        Benchmark cross-shard transaction processing.
        """
        accounts = list(self.network.account_to_shard.keys())
        if len(accounts) < 10:
            accounts = [f"user_{i}" for i in range(200)]
            for acc in accounts:
                if acc not in self.network.account_to_shard:
                    self.network.register_account(acc)
        
        tracker = LatencyTracker()
        
        successful = 0
        failed = 0
        cross_shard_count = 0
        same_shard_count = 0
        
        for i in range(num_transactions):
            sender = random.choice(accounts)
            receiver = random.choice([a for a in accounts if a != sender])
            
            if self.network.is_cross_shard(sender, receiver):
                cross_shard_count += 1
                
                start = tracker.start("cross_shard_tx")
                tx, result = self.coordinator.process_transaction(sender, receiver, 50)
                tracker.end("cross_shard_tx", start)
                
                if tx and tx.status == "committed":
                    successful += 1
                else:
                    failed += 1
            else:
                same_shard_count += 1
        
        stats = tracker.get_stats("cross_shard_tx")
        
        result = {
            "total_transactions": num_transactions,
            "cross_shard": cross_shard_count,
            "same_shard": same_shard_count,
            "cross_shard_ratio": cross_shard_count / num_transactions if num_transactions > 0 else 0,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / cross_shard_count if cross_shard_count > 0 else 0,
            "latency_stats": stats
        }
        
        self.results.append(result)
        return result


class ComprehensiveBenchmark:
    """
    Runs all benchmarks and produces a complete report.
    """
    
    def __init__(self):
        self.results = {}
    
    def run_all(self, num_shards=4, validators_per_shard=25, 
                total_validators=200, num_transactions=500):
        """
        Run comprehensive benchmark suite.
        """
        from sharding import Network
        from cfp.fluidity import ValidatorFluidity
        from cfp.cross_shard import CrossShardCoordinator
        
        print("=" * 60)
        print("CFP COMPREHENSIVE BENCHMARK")
        print("=" * 60)
        
        # Setup
        print("\n[1] Setting up network...")
        network = Network(
            num_shards=num_shards,
            validators_per_shard=validators_per_shard,
            total_validators=total_validators
        )
        
        fluidity = ValidatorFluidity(network, rho=0.2)
        coordinator = CrossShardCoordinator(network)
        
        # Register accounts
        accounts = [f"user_{i}" for i in range(200)]
        for acc in accounts:
            network.register_account(acc)
        
        print(f"  Network: {num_shards} shards, {validators_per_shard} validators each")
        print(f"  Accounts: {len(accounts)}")
        
        # Throughput benchmark
        print("\n[2] Running throughput benchmark...")
        throughput_bench = ThroughputBenchmark(network)
        throughput_result = throughput_bench.run(num_transactions)
        self.results["throughput"] = throughput_result
        print(f"  Throughput: {throughput_result['throughput_tps']:.1f} TPS")
        
        # Rotation overhead benchmark
        print("\n[3] Running rotation overhead benchmark...")
        rotation_bench = RotationOverheadBenchmark(network, fluidity)
        rotation_result = rotation_bench.run(100)
        self.results["rotation"] = rotation_result
        print(f"  Mean rotation time: {rotation_result['mean_time']*1000:.2f} ms")
        
        # Cross-shard benchmark
        print("\n[4] Running cross-shard benchmark...")
        cross_bench = CrossShardBenchmark(network, coordinator)
        cross_result = cross_bench.run(100)
        self.results["cross_shard"] = cross_result
        print(f"  Cross-shard ratio: {cross_result['cross_shard_ratio']:.1%}")
        print(f"  Success rate: {cross_result['success_rate']:.1%}")
        if cross_result['latency_stats']:
            print(f"  Mean latency: {cross_result['latency_stats']['mean']*1000:.2f} ms")
        
        # Scalability benchmark
        print("\n[5] Running scalability benchmark...")
        scale_bench = ScalabilityBenchmark()
        scale_results = scale_bench.run([2, 4, 8], transactions_per_test=200)
        self.results["scalability"] = scale_results
        
        print(f"  2 shards: {scale_results[0]['throughput_tps']:.1f} TPS")
        print(f"  4 shards: {scale_results[1]['throughput_tps']:.1f} TPS")
        print(f"  8 shards: {scale_results[2]['throughput_tps']:.1f} TPS")
        
        return self.results
    
    def print_report(self):
        """Print comprehensive benchmark report."""
        print("\n" + "=" * 60)
        print("BENCHMARK REPORT")
        print("=" * 60)
        
        if "throughput" in self.results:
            r = self.results["throughput"]
            print(f"\nThroughput:")
            print(f"  Transactions: {r['transactions']}")
            print(f"  Time: {r['time_seconds']:.3f} seconds")
            print(f"  TPS: {r['throughput_tps']:.1f}")
        
        if "rotation" in self.results:
            r = self.results["rotation"]
            print(f"\nRotation Overhead:")
            print(f"  Rotations: {r['num_rotations']}")
            print(f"  Mean time: {r['mean_time']*1000:.2f} ms")
            print(f"  Total time: {r['total_time']:.3f} seconds")
        
        if "cross_shard" in self.results:
            r = self.results["cross_shard"]
            print(f"\nCross-Shard Transactions:")
            print(f"  Cross-shard ratio: {r['cross_shard_ratio']:.1%}")
            print(f"  Success rate: {r['success_rate']:.1%}")
            if r['latency_stats']:
                print(f"  Mean latency: {r['latency_stats']['mean']*1000:.2f} ms")
        
        if "scalability" in self.results:
            print(f"\nScalability:")
            for r in self.results["scalability"]:
                print(f"  {r['num_shards']} shards: {r['throughput_tps']:.1f} TPS")
        
        print("\n" + "=" * 60)


# Test code
if __name__ == "__main__":
    benchmark = ComprehensiveBenchmark()
    benchmark.run_all()
    benchmark.print_report()
