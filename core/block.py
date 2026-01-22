"""
Block module for CFP blockchain simulation.
This module defines the block structure that contains transactions.
"""

import hashlib
import time


class Block:
    
    
    def __init__(self, block_number, transactions, previous_hash):
       
        self.block_number = block_number
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.timestamp = time.time()
        self.nonce = 0
        self.block_hash = self.calculate_hash()
    
    def calculate_hash(self):
        
        # Include full transaction data, not just IDs
        tx_data = "".join([
            f"{tx.sender}{tx.receiver}{tx.amount}{tx.tx_id}" 
            for tx in self.transactions
        ])
        
        block_content = (
            f"{self.block_number}"
            f"{tx_data}"
            f"{self.previous_hash}"
            f"{self.timestamp}"
            f"{self.nonce}"
        )
        
        return hashlib.sha256(block_content.encode()).hexdigest()
    
    def get_transaction_count(self):
        """Return the number of transactions in this block."""
        return len(self.transactions)
    
    def to_dict(self):
       
        return {
            "block_number": self.block_number,
            "block_hash": self.block_hash,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "transaction_count": self.get_transaction_count(),
            "transactions": [tx.to_dict() for tx in self.transactions]
        }
    
    def __str__(self):
        """Human readable representation of the block."""
        return (
            f"Block #{self.block_number} "
            f"[{self.block_hash[:8]}...] "
            f"({self.get_transaction_count()} txs) "
            f"-> prev: [{self.previous_hash[:8]}...]"
        )
    
    def __repr__(self):
        return self.__str__()


# Test code for Phase 1.2
if __name__ == "__main__":
    # Import Transaction class for testing
    from transaction import Transaction, create_random_transaction
    
    print("=" * 60)
    print("Phase 1.2 Test: Block Class")
    print("=" * 60)
    
    # Create test accounts
    test_accounts = [f"account_{i}" for i in range(10)]
    
    # Test 1: Create a genesis block (first block with no previous hash)
    print("\nTest 1: Create genesis block")
    genesis_transactions = [
        Transaction("system", "account_0", 1000),
        Transaction("system", "account_1", 1000),
    ]
    genesis_block = Block(
        block_number=0,
        transactions=genesis_transactions,
        previous_hash="0" * 64  # Genesis block has no real previous hash
    )
    print(f"Created: {genesis_block}")
    print(f"Full hash: {genesis_block.block_hash}")
    
    # Test 2: Create a block with 5 random transactions
    print("\n" + "-" * 40)
    print("Test 2: Create block with 5 transactions")
    print("-" * 40)
    
    random_transactions = [create_random_transaction(test_accounts) for _ in range(5)]
    block_1 = Block(
        block_number=1,
        transactions=random_transactions,
        previous_hash=genesis_block.block_hash
    )
    print(f"Created: {block_1}")
    
    print("\nTransactions in block:")
    for i, tx in enumerate(block_1.transactions):
        print(f"  {i+1}. {tx}")
    
    # Test 3: Verify block links to previous block
    print("\n" + "-" * 40)
    print("Test 3: Verify block chain linkage")
    print("-" * 40)
    
    if block_1.previous_hash == genesis_block.block_hash:
        print("SUCCESS: Block 1 correctly references genesis block")
    else:
        print("FAILED: Block linkage is broken")
    
    # Test 4: Verify hash changes when content changes
    print("\n" + "-" * 40)
    print("Test 4: Verify hash integrity")
    print("-" * 40)
    
    original_hash = block_1.block_hash
    block_1.nonce = 999  # Change nonce
    new_hash = block_1.calculate_hash()
    
    if original_hash != new_hash:
        print("SUCCESS: Hash changes when block content changes")
        print(f"  Original: {original_hash[:16]}...")
        print(f"  After modification: {new_hash[:16]}...")
    else:
        print("FAILED: Hash should change when content changes")
    
    print("\n" + "=" * 60)
    print("Phase 1.2 Complete!")
    print("=" * 60)
