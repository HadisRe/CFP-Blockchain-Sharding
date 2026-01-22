"""
Validator module for CFP simulation.
Defines the validator entity that participates in consensus.

Each validator has:
- Unique ID
- ECDSA key pair for signing (used in VRF)
- Vote history for CFI tracking
"""

import hashlib
import uuid
import random
from ecdsa import SigningKey, VerifyingKey, SECP256k1, BadSignatureError


class Validator:
   

    def __init__(self, validator_id=None):
        
        if validator_id is None:
            validator_id = str(uuid.uuid4())[:8]

        self.validator_id = validator_id

        # Generate real ECDSA key pair
        self.private_key = SigningKey.generate(curve=SECP256k1)
        self.public_key = self.private_key.get_verifying_key()

        self.is_adversarial = False
        self.vote_history = []
        self.current_shard = None

    def get_public_key_bytes(self):
         return self.public_key.to_string()

    def get_public_key_hex(self):
         return self.public_key.to_string().hex()[:32]

    def sign(self, message):
        
        if isinstance(message, str):
            message = message.encode()
        return self.private_key.sign_deterministic(message, hashfunc=hashlib.sha256)

    def verify_signature(self, message, signature, public_key=None):
       
        if public_key is None:
            public_key = self.public_key

        if isinstance(message, str):
            message = message.encode()

        try:
            public_key.verify(signature, message, hashfunc=hashlib.sha256)
            return True
        except BadSignatureError:
            return False

    def vrf_generate(self, input_data):
       
        if isinstance(input_data, str):
            input_data = input_data.encode()

        # Step 1: Hash the input
        input_hash = hashlib.sha256(input_data).digest()

        # Step 2: Sign the hash deterministically
        # hashfunc=hashlib.sha256 makes it deterministic per RFC6979
        proof = self.private_key.sign_deterministic(
            input_hash,
            hashfunc=hashlib.sha256
        )

        # Step 3: Hash the proof to get the random output
        random_bytes = hashlib.sha256(proof).digest()
        random_value = int.from_bytes(random_bytes[:8], 'big') / (2 ** 64)

        return {
            "value": random_value,  # number between 0 and 1
            "proof": proof,
            "input_hash": input_hash
        }

    def vrf_verify(self, input_data, vrf_output, public_key=None):
        
        if public_key is None:
            public_key = self.public_key

        if isinstance(input_data, str):
            input_data = input_data.encode()

        # Recompute input hash
        input_hash = hashlib.sha256(input_data).digest()

        # Check that input hash matches
        if input_hash != vrf_output["input_hash"]:
            return False

        # Verify the signature (proof)
        try:
            public_key.verify(vrf_output["proof"], input_hash, hashfunc=hashlib.sha256)
        except BadSignatureError:
            return False

        # Recompute the random value and check it matches
        expected_bytes = hashlib.sha256(vrf_output["proof"]).digest()
        expected_value = int.from_bytes(expected_bytes[:8], 'big') / (2 ** 64)

        if abs(expected_value - vrf_output["value"]) > 1e-10:
            return False

        return True

    def cast_vote(self, block_hash, vote):
       
        self.vote_history.append({
            "block": block_hash[:8] if len(block_hash) > 8 else block_hash,
            "vote": vote
        })
        return vote

    def get_recent_votes(self, count=10):
        """Get the last N votes cast by this validator."""
        return self.vote_history[-count:] if self.vote_history else []

    def mark_as_adversarial(self):
        """Mark this validator as controlled by adversary."""
        self.is_adversarial = True

    def __str__(self):
        status = "[ADV]" if self.is_adversarial else ""
        return f"Validator({self.validator_id}){status}"

    def __repr__(self):
        return self.__str__()


class ValidatorPool:
    

    def __init__(self, total_validators):
        """Create a pool with specified number of validators."""
        self.validators = []
        self.assigned = set()

        for i in range(total_validators):
            v = Validator(f"v_{i:04d}")
            self.validators.append(v)

    def get_validator_by_id(self, validator_id):
         for v in self.validators:
            if v.validator_id == validator_id:
                return v
        return None

    def get_available(self):
         return [v for v in self.validators if v.validator_id not in self.assigned]

    def mark_assigned(self, validator):
         self.assigned.add(validator.validator_id)

    def mark_unassigned(self, validator):
         self.assigned.discard(validator.validator_id)

    def get_random_validators(self, count, exclude=None):
        
        if exclude is None:
            exclude = []

        exclude_ids = {v.validator_id for v in exclude}
        available = [v for v in self.validators if v.validator_id not in exclude_ids]

        if len(available) < count:
            count = len(available)

        return random.sample(available, count)

    def set_adversarial_fraction(self, fraction):
       
        # First reset all
        for v in self.validators:
            v.is_adversarial = False

        count = int(len(self.validators) * fraction)
        adversarial = random.sample(self.validators, count)

        for v in adversarial:
            v.mark_as_adversarial()

        return adversarial

    def get_adversarial_count(self):
         return sum(1 for v in self.validators if v.is_adversarial)

    def __len__(self):
        return len(self.validators)

    def __str__(self):
        adv = self.get_adversarial_count()
        return f"ValidatorPool(total={len(self)}, adversarial={adv})"


# test code
if __name__ == "__main__":
    print("Testing Validator with VRF")
    print("=" * 50)

    # Create validator
    v = Validator("test_001")
    print(f"Created: {v}")
    print(f"Public key: {v.get_public_key_hex()}")

    # Test signing
    message = "Hello, blockchain!"
    signature = v.sign(message)
    print(f"\nSigned message: '{message}'")
    print(f"Signature valid: {v.verify_signature(message, signature)}")
    print(f"Wrong message valid: {v.verify_signature('Wrong!', signature)}")

    # Test VRF
    print(f"\n" + "-" * 40)
    print("Testing VRF")

    input_data = "block_hash_12345"
    vrf_output = v.vrf_generate(input_data)

    print(f"Input: {input_data}")
    print(f"VRF value: {vrf_output['value']:.6f}")
    print(f"Proof length: {len(vrf_output['proof'])} bytes")

    # Verify VRF
    is_valid = v.vrf_verify(input_data, vrf_output)
    print(f"VRF verification: {is_valid}")

    # Try with wrong input
    is_valid_wrong = v.vrf_verify("wrong_input", vrf_output)
    print(f"VRF with wrong input: {is_valid_wrong}")

    # Test determinism - same input should give same output
    vrf_output2 = v.vrf_generate(input_data)
    print(f"\nDeterminism test:")
    print(f"Same input, same output: {vrf_output['value'] == vrf_output2['value']}")

    # Test ValidatorPool
    print(f"\n" + "=" * 50)
    print("Testing ValidatorPool")

    pool = ValidatorPool(10)
    print(f"Created: {pool}")

    # Each validator should have unique VRF output for same input
    print(f"\nVRF outputs for same input from different validators:")
    for validator in pool.validators[:3]:
        output = validator.vrf_generate("same_input")
        print(f"  {validator.validator_id}: {output['value']:.6f}")

    print("\nValidator test complete!")
