"""
Consensus Friction Index (CFI) module for CFP simulation.
Measures voting diversity to detect predictable behavior.

Based on paper equation (15):
CFI(T) = 1 - (1/n) * sum((streak_v / T)^2)

High CFI = diverse voting (good)
Low CFI = predictable voting (bad, possible attack)
"""


class ConsensusFrictionIndex:
    """
    Calculates and monitors CFI for shards.
    When CFI drops below threshold, increases rotation rate.
    """
    
    def __init__(self, fluidity_manager, threshold=0.5, window=10):
        """
        Setup CFI monitor.
        
        fluidity_manager: the ValidatorFluidity object to adjust
        threshold: CFI below this triggers response
        window: how many blocks to look at (T in paper)
        """
        self.fluidity = fluidity_manager
        self.threshold = threshold
        self.window = window
        self.cfi_history = {}  # track CFI per shard over time
    
    def calculate_streak(self, votes):
        """
        Find longest consecutive same votes.
        Example: [T,T,T,F,F,T] -> streak is 3 (the three T's)
        """
        if not votes:
            return 0
        
        max_streak = 1
        current_streak = 1
        
        for i in range(1, len(votes)):
            if votes[i] == votes[i-1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        
        return max_streak
    
    def calculate_cfi(self, shard):
        """
        Calculate CFI for a shard.
        Uses equation (15) from paper.
        """
        validators = shard.validators
        n = len(validators)
        
        if n == 0:
            return 1.0  # no validators means no risk
        
        T = self.window
        sum_squared = 0
        
        for v in validators:
            # get recent votes
            recent = v.get_recent_votes(T)
            
            if len(recent) < 2:
                # not enough history, assume diverse
                streak = 1
            else:
                # extract just the vote values
                vote_values = [vote["vote"] for vote in recent]
                streak = self.calculate_streak(vote_values)
            
            # add to sum: (streak/T)^2
            ratio = streak / T
            sum_squared += ratio * ratio
        
        # CFI formula: 1 - (1/n) * sum
        cfi = 1 - (sum_squared / n)
        
        return cfi
    
    def check_shard(self, shard):
        """
        Check CFI for a shard and respond if needed.
        Returns CFI value and whether action was taken.
        """
        cfi = self.calculate_cfi(shard)
        
        # record history
        if shard.shard_id not in self.cfi_history:
            self.cfi_history[shard.shard_id] = []
        self.cfi_history[shard.shard_id].append(cfi)
        
        # check threshold
        action_taken = False
        if cfi < self.threshold:
            # low CFI detected, increase rotation
            self.fluidity.increase_rho(0.05)
            action_taken = True
        
        return {
            "shard_id": shard.shard_id,
            "cfi": cfi,
            "threshold": self.threshold,
            "below_threshold": cfi < self.threshold,
            "action_taken": action_taken
        }
    
    def check_all_shards(self, network):
        """Check CFI for all shards in network."""
        results = []
        for shard_id, shard in network.shards.items():
            result = self.check_shard(shard)
            results.append(result)
        return results
    
    def get_average_cfi(self, network):
        """Get average CFI across all shards."""
        total = 0
        count = 0
        
        for shard in network.shards.values():
            total += self.calculate_cfi(shard)
            count += 1
        
        return total / count if count > 0 else 1.0
    
    def simulate_voting_round(self, network, adversarial_coordination=False):
        """
        Simulate validators casting votes.
        If adversarial_coordination=True, adversarial validators vote same way.
        """
        block_hash = f"block_{len(self.cfi_history)}"
        
        for shard in network.shards.values():
            for v in shard.validators:
                if v.is_adversarial and adversarial_coordination:
                    # adversarial validators always vote True (coordinated)
                    vote = True
                else:
                    # honest validators vote randomly
                    import random
                    vote = random.choice([True, False])
                
                v.cast_vote(block_hash, vote)
    
    def print_status(self, network):
        """Print CFI status for all shards."""
        print(f"\nCFI Status (threshold={self.threshold}):")
        print("-" * 45)
        
        for shard_id, shard in network.shards.items():
            cfi = self.calculate_cfi(shard)
            status = "OK" if cfi >= self.threshold else "WARNING"
            adv = shard.get_adversarial_fraction()
            print(f"  Shard {shard_id}: CFI={cfi:.3f} [{status}] (adv={adv:.0%})")
        
        avg = self.get_average_cfi(network)
        print(f"\n  Average CFI: {avg:.3f}")
        print(f"  Current rho: {self.fluidity.rho:.1%}")


# test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    
    from sharding import Network
    from cfp.fluidity import ValidatorFluidity
    
    print("Testing CFI")
    print("=" * 50)
    
    # setup network
    net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
    net.validator_pool.set_adversarial_fraction(0.3)
    
    fluidity = ValidatorFluidity(net, rho=0.2)
    cfi_monitor = ConsensusFrictionIndex(fluidity, threshold=0.5, window=10)
    
    print("Running simulation with honest voting...")
    
    # simulate 10 rounds of honest voting
    for i in range(10):
        cfi_monitor.simulate_voting_round(net, adversarial_coordination=False)
    
    cfi_monitor.print_status(net)
    
    print("\n" + "=" * 50)
    print("Now simulating adversarial coordination...")
    
    # simulate 10 rounds with adversarial coordination
    for i in range(10):
        cfi_monitor.simulate_voting_round(net, adversarial_coordination=True)
    
    cfi_monitor.print_status(net)
    
    # check if CFI dropped
    results = cfi_monitor.check_all_shards(net)
    warnings = sum(1 for r in results if r["below_threshold"])
    print(f"\nShards with low CFI: {warnings}")
    
    print("\nCFI test complete!")
