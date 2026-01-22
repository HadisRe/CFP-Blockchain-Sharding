"""
Network module for CFP simulation.
Manages the entire sharded blockchain network.
"""

import random
from sharding.validator import ValidatorPool
from sharding.shard import Shard
from core.transaction import Transaction


class Network:
   
    
    def __init__(self, num_shards, validators_per_shard, total_validators=None):
        " 
        if total_validators is None:
            # need extra validators for rotation
            total_validators = num_shards * validators_per_shard * 2
        
        self.num_shards = num_shards
        self.validators_per_shard = validators_per_shard
        
        # create global validator pool
        self.validator_pool = ValidatorPool(total_validators)
        
        # create shards
        self.shards = {}
        self._setup_shards()
        
        # track accounts
        self.account_to_shard = {}
    
    def _setup_shards(self):
         all_validators = self.validator_pool.validators.copy()
        random.shuffle(all_validators)
        
        for i in range(self.num_shards):
            # get validators for this shard
            start = i * self.validators_per_shard
            end = start + self.validators_per_shard
            shard_validators = all_validators[start:end]
            
            # create shard
            shard = Shard(shard_id=i, validators=shard_validators)
            self.shards[i] = shard
            
            # mark validators as assigned
            for v in shard_validators:
                self.validator_pool.mark_assigned(v)
    
    def register_account(self, account_id, shard_id=None):
       
        if shard_id is None:
            shard_id = hash(account_id) % self.num_shards
        
        self.account_to_shard[account_id] = shard_id
        self.shards[shard_id].add_account(account_id)
        return shard_id
    
    def get_account_shard(self, account_id):
         return self.account_to_shard.get(account_id)
    
    def is_cross_shard(self, sender, receiver):
        """Check if transaction would be cross-shard."""
        sender_shard = self.get_account_shard(sender)
        receiver_shard = self.get_account_shard(receiver)
        
        if sender_shard is None or receiver_shard is None:
            return False
        
        return sender_shard != receiver_shard
    
    def submit_transaction(self, tx):
        
        sender_shard = self.get_account_shard(tx.sender)
        
        if sender_shard is None:
            # register sender if new
            sender_shard = self.register_account(tx.sender)
        
        # for now, just send to sender's shard
        # cross-shard handling comes in later phase
        self.shards[sender_shard].submit_transaction(tx)
        
        return sender_shard
    
    def process_round(self):
        
        blocks = []
        for shard_id, shard in self.shards.items():
            block = shard.process_block()
            if block:
                blocks.append((shard_id, block))
        return blocks
    
    def get_network_stats(self):
         total_blocks = sum(s.get_chain_height() for s in self.shards.values())
        total_validators = sum(s.get_validator_count() for s in self.shards.values())
        
        adv_fractions = [s.get_adversarial_fraction() for s in self.shards.values()]
        avg_adv = sum(adv_fractions) / len(adv_fractions) if adv_fractions else 0
        
        return {
            "num_shards": self.num_shards,
            "total_blocks": total_blocks,
            "total_validators": total_validators,
            "pool_size": len(self.validator_pool),
            "avg_adversarial": avg_adv,
            "accounts": len(self.account_to_shard)
        }
    
    def print_status(self):
         stats = self.get_network_stats()
        
        print(f"\n{'='*50}")
        print("Network Status")
        print(f"{'='*50}")
        print(f"Shards: {stats['num_shards']}")
        print(f"Total validators in shards: {stats['total_validators']}")
        print(f"Global pool size: {stats['pool_size']}")
        print(f"Registered accounts: {stats['accounts']}")
        print(f"Average adversarial: {stats['avg_adversarial']:.1%}")
        print(f"\nPer-shard status:")
        
        for shard_id, shard in self.shards.items():
            print(f"  {shard}")
    
    def __str__(self):
        return f"Network(shards={self.num_shards}, validators={len(self.validator_pool)})"


# test
if __name__ == "__main__":
    print("Testing Network")
    print("=" * 50)
    
    # create network: 4 shards, 25 validators each
    net = Network(num_shards=4, validators_per_shard=25)
    print(f"Created: {net}")
    
    # register some accounts
    accounts = [f"user_{i}" for i in range(100)]
    for acc in accounts:
        net.register_account(acc)
    print(f"Registered {len(accounts)} accounts")
    
    # create transactions
    for i in range(80):
        sender = random.choice(accounts)
        receiver = random.choice([a for a in accounts if a != sender])
        tx = Transaction(sender, receiver, random.randint(1, 100))
        net.submit_transaction(tx)
    print(f"Submitted 80 transactions")
    
    # count cross-shard
    cross_count = 0
    for i in range(100):
        s = random.choice(accounts)
        r = random.choice([a for a in accounts if a != s])
        if net.is_cross_shard(s, r):
            cross_count += 1
    print(f"Cross-shard ratio (sample): {cross_count}%")
    
    # process a round
    blocks = net.process_round()
    print(f"Processed {len(blocks)} blocks")
    
    # show status
    net.print_status()
    
    print("\nNetwork test complete!")
