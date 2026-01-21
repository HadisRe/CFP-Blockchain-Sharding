"""
Shard module for CFP simulation.
Each shard is an independent partition of the blockchain network.
"""

import random
from core.blockchain import Blockchain
from core.transaction import Transaction


class Shard:
    """
    A shard is a partition of the network with its own blockchain.
    Each shard has a committee of validators that process transactions.
    """
    
    def __init__(self, shard_id, validators=None):
        """
        Create new shard.
        shard_id: unique identifier for this shard
        validators: initial list of validators for this shard
        """
        self.shard_id = shard_id
        self.blockchain = Blockchain(chain_id=f"shard_{shard_id}")
        self.validators = validators if validators else []
        self.pending_txs = []
        self.accounts = set()  # accounts assigned to this shard
        
        # assign shard reference to validators
        for v in self.validators:
            v.current_shard = self.shard_id
    
    def add_validator(self, validator):
        """Add a validator to this shard's committee."""
        if validator not in self.validators:
            self.validators.append(validator)
            validator.current_shard = self.shard_id
    
    def remove_validator(self, validator):
        """Remove a validator from this shard."""
        if validator in self.validators:
            self.validators.remove(validator)
            validator.current_shard = None
    
    def get_validator_count(self):
        """Return number of validators in this shard."""
        return len(self.validators)
    
    def add_account(self, account_id):
        """Assign an account to this shard."""
        self.accounts.add(account_id)
    
    def has_account(self, account_id):
        """Check if account belongs to this shard."""
        return account_id in self.accounts
    
    def submit_transaction(self, tx):
        """Add transaction to pending pool."""
        self.pending_txs.append(tx)
    
    def process_block(self):
        """
        Create a new block from pending transactions.
        In real system, this would involve consensus voting.
        Returns the new block or None if no transactions.
        """
        if not self.pending_txs:
            return None
        
        # take transactions for this block
        block_txs = self.pending_txs[:50]  # max 50 per block
        self.pending_txs = self.pending_txs[50:]
        
        # create block
        block = self.blockchain.create_block(block_txs)
        return block
    
    def get_chain_height(self):
        """Get current blockchain height."""
        return self.blockchain.get_chain_length()
    
    def get_adversarial_fraction(self):
        """Calculate fraction of adversarial validators in this shard."""
        if not self.validators:
            return 0
        adv_count = sum(1 for v in self.validators if v.is_adversarial)
        return adv_count / len(self.validators)
    
    def get_state_snapshot(self, height=None):
        """
        Get a snapshot of shard state at given height.
        Used for cross-shard transactions (from paper section 3.1).
        """
        if height is None:
            height = self.get_chain_height() - 1
        
        block = self.blockchain.get_block_by_number(height)
        if block is None:
            return None
        
        return {
            "shard_id": self.shard_id,
            "height": height,
            "block_hash": block.block_hash,
            "validators": [v.validator_id for v in self.validators],
            "validator_count": len(self.validators)
        }
    
    def __str__(self):
        adv = self.get_adversarial_fraction()
        return f"Shard[{self.shard_id}](validators={len(self.validators)}, adv={adv:.1%}, blocks={self.get_chain_height()})"
    
    def __repr__(self):
        return self.__str__()


# test
if __name__ == "__main__":
    from validator import Validator, ValidatorPool
    
    print("Testing Shard")
    print("-" * 40)
    
    # create validators
    pool = ValidatorPool(50)
    validators = pool.get_random_validators(10)
    
    # create shard
    shard = Shard(shard_id=0, validators=validators)
    print(f"Created: {shard}")
    
    # add some accounts
    for i in range(20):
        shard.add_account(f"acc_{i}")
    print(f"Added 20 accounts")
    
    # create transactions
    for i in range(15):
        tx = Transaction(f"acc_{i}", f"acc_{i+1}", random.randint(10, 100))
        shard.submit_transaction(tx)
    print(f"Submitted 15 transactions")
    
    # process block
    block = shard.process_block()
    print(f"Processed block: {block}")
    
    # get snapshot
    snap = shard.get_state_snapshot()
    print(f"State snapshot: height={snap['height']}, hash={snap['block_hash'][:8]}...")
    
    print("\nShard test complete!")
