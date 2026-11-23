"""
Blockchain module for CFP blockchain simulation.
This module defines the blockchain structure that manages the chain of blocks.
"""

from .transaction import Transaction, create_random_transaction
from .block import Block


class Blockchain:
    """
    Represents a blockchain - a chain of linked blocks.
    
    The blockchain maintains the entire history of transactions
    and provides methods to add new blocks and validate the chain.
    """
    
    def __init__(self, chain_id="main"):
        """
        Initialize a new blockchain with a genesis block.
        
        Args:
            chain_id: Identifier for this blockchain (useful when we have multiple shards)
        """
        self.chain_id = chain_id
        self.chain = []
        self.pending_transactions = []
        
        # Create the genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """
        Create the first block in the chain.
        
        The genesis block has no previous hash and contains
        initial system transactions.
        """
        genesis_tx = Transaction("system", "genesis", 0)
        genesis_block = Block(
            block_number=0,
            transactions=[genesis_tx],
            previous_hash="0" * 64
        )
        self.chain.append(genesis_block)
    
    def get_latest_block(self):
        """
        Get the most recent block in the chain.
        
        Returns:
            The last Block object in the chain
        """
        return self.chain[-1]
    
    def get_block_by_number(self, block_number):
        """
        Get a block by its block number.
        
        Args:
            block_number: The height of the block to retrieve
            
        Returns:
            The Block object at that height, or None if not found
        """
        if 0 <= block_number < len(self.chain):
            return self.chain[block_number]
        return None
    
    def add_transaction(self, transaction):
        """
        Add a transaction to the pending pool.
        
        Args:
            transaction: Transaction object to add
        """
        self.pending_transactions.append(transaction)
    
    def create_block(self, transactions=None):
        """
        Create a new block with given transactions or pending transactions.
        
        Args:
            transactions: List of transactions to include. 
                         If None, uses pending transactions.
        
        Returns:
            The newly created Block object
        """
        if transactions is None:
            transactions = self.pending_transactions
            self.pending_transactions = []
        
        new_block = Block(
            block_number=len(self.chain),
            transactions=transactions,
            previous_hash=self.get_latest_block().block_hash
        )
        
        self.chain.append(new_block)
        return new_block
    
    def is_chain_valid(self):
        """
        Validate the integrity of the entire blockchain.
        
        Checks that:
        1. Each block's stored hash matches recalculated hash
        2. Each block correctly references the previous block's hash
        
        Returns:
            True if the chain is valid, False otherwise
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Recalculate hash and compare with stored hash
            # This detects any tampering with block contents
            recalculated_hash = current_block.calculate_hash()
            if current_block.block_hash != recalculated_hash:
                return False
            
            # Check if current block points to correct previous hash
            if current_block.previous_hash != previous_block.block_hash:
                return False
        
        return True
    
    def get_chain_length(self):
        """Return the number of blocks in the chain."""
        return len(self.chain)
    
    def get_total_transactions(self):
        """Return the total number of transactions across all blocks."""
        return sum(block.get_transaction_count() for block in self.chain)
    
    def print_chain(self):
        """Print a summary of the entire blockchain."""
        print(f"\nBlockchain '{self.chain_id}' Summary:")
        print(f"Total blocks: {self.get_chain_length()}")
        print(f"Total transactions: {self.get_total_transactions()}")
        print("-" * 50)
        
        for block in self.chain:
            print(f"  {block}")
    
    def __str__(self):
        return f"Blockchain[{self.chain_id}]: {self.get_chain_length()} blocks"
    
    def __repr__(self):
        return self.__str__()


# Test code for Phase 1.3
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1.3 Test: Blockchain Class")
    print("=" * 60)
    
    # Create test accounts
    test_accounts = [f"account_{i}" for i in range(10)]
    
    # Test 1: Create a new blockchain
    print("\nTest 1: Create new blockchain")
    blockchain = Blockchain(chain_id="test_chain")
    print(f"Created: {blockchain}")
    print(f"Genesis block: {blockchain.get_latest_block()}")
    
    # Test 2: Add transactions and create blocks
    print("\n" + "-" * 40)
    print("Test 2: Add blocks with transactions")
    print("-" * 40)
    
    for block_num in range(1, 4):
        # Create random transactions
        transactions = [create_random_transaction(test_accounts) for _ in range(5)]
        
        # Create a new block
        new_block = blockchain.create_block(transactions)
        print(f"Added: {new_block}")
    
    # Test 3: Print full chain
    print("\n" + "-" * 40)
    print("Test 3: Print full chain")
    print("-" * 40)
    blockchain.print_chain()
    
    # Test 4: Validate chain integrity
    print("\n" + "-" * 40)
    print("Test 4: Validate chain integrity")
    print("-" * 40)
    
    if blockchain.is_chain_valid():
        print("SUCCESS: Blockchain is valid")
    else:
        print("FAILED: Blockchain validation failed")
    
    # Test 5: Try to tamper with a block
    print("\n" + "-" * 40)
    print("Test 5: Detect tampering")
    print("-" * 40)
    
    # Save original hash
    original_hash = blockchain.chain[1].block_hash
    
    # Tamper with block 1
    blockchain.chain[1].transactions[0].amount = 9999
    
    # Check if tampering is detected
    if not blockchain.is_chain_valid():
        print("SUCCESS: Tampering was detected")
    else:
        print("FAILED: Tampering was not detected")
    
    # Restore for clean state (recalculate hash after tampering)
    blockchain.chain[1].block_hash = blockchain.chain[1].calculate_hash()
    
    print("\n" + "=" * 60)
    print("Phase 1.3 Complete!")
    print("=" * 60)
