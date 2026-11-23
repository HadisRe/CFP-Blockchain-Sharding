"""
Transaction module for CFP blockchain simulation.
This module defines the basic transaction structure used throughout the system.
"""

import hashlib
import time
import uuid


class Transaction:
    """
    Represents a single transaction in the blockchain.
    
    A transaction transfers value from one account to another.
    Each transaction has a unique identifier and timestamp.
    """
    
    def __init__(self, sender, receiver, amount):
        """
        Initialize a new transaction.
        
        Args:
            sender: The account address sending the funds
            receiver: The account address receiving the funds
            amount: The amount being transferred
        """
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.timestamp = time.time()
        self.tx_id = self.generate_tx_id()
    
    def generate_tx_id(self):
        """
        Generate a unique transaction ID based on transaction data.
        
        Returns:
            A hexadecimal string representing the transaction hash
        """
        tx_content = f"{self.sender}{self.receiver}{self.amount}{self.timestamp}{uuid.uuid4()}"
        return hashlib.sha256(tx_content.encode()).hexdigest()
    
    def to_dict(self):
        """
        Convert transaction to dictionary format.
        
        Returns:
            Dictionary containing all transaction fields
        """
        return {
            "tx_id": self.tx_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "timestamp": self.timestamp
        }
    
    def __str__(self):
        """Human readable representation of the transaction."""
        return f"TX[{self.tx_id[:8]}...]: {self.sender} -> {self.receiver} : {self.amount}"
    
    def __repr__(self):
        return self.__str__()


def create_random_transaction(accounts):
    """
    Create a random transaction between two accounts.
    
    Args:
        accounts: List of account addresses to choose from
        
    Returns:
        A new Transaction object with random sender, receiver and amount
    """
    import random
    
    sender = random.choice(accounts)
    receiver = random.choice([acc for acc in accounts if acc != sender])
    amount = random.randint(1, 100)
    
    return Transaction(sender, receiver, amount)


# Test code for Phase 1.1
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1.1 Test: Transaction Class")
    print("=" * 60)
    
    # Create some test accounts
    test_accounts = [f"account_{i}" for i in range(10)]
    
    # Test 1: Create a single transaction manually
    print("\nTest 1: Create single transaction")
    tx = Transaction("alice", "bob", 50)
    print(f"Created: {tx}")
    print(f"Full TX ID: {tx.tx_id}")
    print(f"Dictionary format: {tx.to_dict()}")
    
    # Test 2: Create 10 random transactions
    print("\n" + "-" * 40)
    print("Test 2: Create 10 random transactions")
    print("-" * 40)
    
    transactions = []
    for i in range(10):
        tx = create_random_transaction(test_accounts)
        transactions.append(tx)
        print(f"{i+1}. {tx}")
    
    # Test 3: Verify all transactions have unique IDs
    print("\n" + "-" * 40)
    print("Test 3: Verify unique transaction IDs")
    print("-" * 40)
    
    tx_ids = [tx.tx_id for tx in transactions]
    unique_ids = set(tx_ids)
    
    if len(tx_ids) == len(unique_ids):
        print(f"SUCCESS: All {len(transactions)} transactions have unique IDs")
    else:
        print(f"FAILED: Found duplicate IDs")
    
    print("\n" + "=" * 60)
    print("Phase 1.1 Complete!")
    print("=" * 60)
