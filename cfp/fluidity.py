"""
Validator Fluidity module for CFP simulation.
Implements the core CFP mechanism: rotating validators every block using VRF.

Based on paper:
- Equation (12): rt = ceil(rho * n) - validators to rotate per block
- Equation (14): R_cum(L) = 1 - (1-rho)^L - cumulative refresh after L blocks

VRF (Verifiable Random Function) is used to:
1. Select which validators leave a shard
2. Select which validators join a shard
3. Ensure selection is unpredictable but verifiable
"""

import math
import hashlib


class VRFBasedSelector:
    """
    Selects validators using VRF outputs.

    How it works:
    1. Each validator generates a VRF output using the block hash
    2. Sort validators by their VRF output
    3. Select top/bottom N validators for rotation

    This is verifiable because:
    - Anyone can check each validator's VRF output
    - The selection is deterministic based on VRF outputs
    """

    def __init__(self):
        self.selection_history = []

    def select_validators_to_leave(self, shard, block_hash, count):
        """
        Select validators to leave the shard using VRF.

        Process:
        1. Each validator in shard generates VRF output with block_hash
        2. Sort by VRF value
        3. Lowest values leave (could also be highest, just needs to be consistent)

        Returns list of (validator, vrf_output) tuples for verification.
        """
        if count <= 0 or count >= len(shard.validators):
            return []

        # Each validator generates VRF
        vrf_results = []
        for v in shard.validators:
            vrf_input = f"{block_hash}:leave:{shard.shard_id}:{v.validator_id}"
            vrf_output = v.vrf_generate(vrf_input)
            vrf_results.append({
                "validator": v,
                "vrf_output": vrf_output,
                "vrf_input": vrf_input
            })

        # Sort by VRF value
        vrf_results.sort(key=lambda x: x["vrf_output"]["value"])

        # Select lowest values to leave
        selected = vrf_results[:count]

        # Record for history
        self.selection_history.append({
            "type": "leave",
            "shard_id": shard.shard_id,
            "block_hash": block_hash[:16],
            "count": count,
            "selected": [s["validator"].validator_id for s in selected]
        })

        return selected

    def select_validators_to_join(self, available_validators, shard, block_hash, count):
        """
        Select validators to join the shard using VRF.

        Same process but from the pool of available validators.
        """
        if count <= 0 or len(available_validators) == 0:
            return []

        if count > len(available_validators):
            count = len(available_validators)

        # Each available validator generates VRF
        vrf_results = []
        for v in available_validators:
            vrf_input = f"{block_hash}:join:{shard.shard_id}:{v.validator_id}"
            vrf_output = v.vrf_generate(vrf_input)
            vrf_results.append({
                "validator": v,
                "vrf_output": vrf_output,
                "vrf_input": vrf_input
            })

        # Sort by VRF value
        vrf_results.sort(key=lambda x: x["vrf_output"]["value"])

        # Select lowest values to join
        selected = vrf_results[:count]

        # Record for history
        self.selection_history.append({
            "type": "join",
            "shard_id": shard.shard_id,
            "block_hash": block_hash[:16],
            "count": count,
            "selected": [s["validator"].validator_id for s in selected]
        })

        return selected

    def verify_selection(self, selection_result):
        """
        Verify that a selection was done correctly.

        Anyone can call this to verify:
        1. Each VRF output is valid for the validator's public key
        2. The selection follows the sorting rule
        """
        if not selection_result:
            return True

        # Verify each VRF output
        for item in selection_result:
            v = item["validator"]
            vrf_output = item["vrf_output"]
            vrf_input = item["vrf_input"]

            if not v.vrf_verify(vrf_input, vrf_output):
                return False

        # Verify sorting
        values = [item["vrf_output"]["value"] for item in selection_result]
        if values != sorted(values):
            return False

        return True


class ValidatorFluidity:
    """
    Manages validator rotation within shards.
    This is the core mechanism that makes CFP different from other protocols.

    Key properties:
    - Unpredictable: No one can predict who will be selected before the block
    - Verifiable: Anyone can verify the selection was fair using VRF proofs
    - Configurable: Rotation rate (rho) can be adjusted based on security needs
    """

    def __init__(self, network, rho=0.2, rho_min=0.1, rho_max=0.5):
        """
        Setup fluidity manager.

        network: the Network object containing shards
        rho: rotation rate (default 20% per block)
        rho_min: minimum rotation rate
        rho_max: maximum rotation rate
        """
        self.network = network
        self.rho = rho
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.selector = VRFBasedSelector()
        self.rotation_history = []
        self.block_number = 0

    def calculate_rotation_count(self, shard):
        """
        Calculate how many validators to rotate.
        From paper equation (12): rt = ceil(rho * n)

        Parameters:
        - rho: rotation rate (e.g., 0.2 = 20%)
        - n: number of validators in shard

        Returns:
        - rt: number of validators to rotate
        """
        n = shard.get_validator_count()
        rt = math.ceil(self.rho * n)
        return rt

    def get_current_block_hash(self):
        """
        Get hash to use for VRF input.
        In real system, this would be the previous block hash.
        """
        # Simulate block hash
        self.block_number += 1
        block_data = f"block_{self.block_number}_{id(self)}"
        return hashlib.sha256(block_data.encode()).hexdigest()

    def rotate_shard(self, shard, block_hash=None):
        """
        Perform one rotation on a shard using VRF-based selection.

        Process:
        1. Calculate how many to rotate (rt)
        2. Use VRF to select who leaves
        3. Use VRF to select who joins from available pool
        4. Execute the rotation
        5. Return info with proofs for verification
        """
        if block_hash is None:
            block_hash = self.get_current_block_hash()

        rt = self.calculate_rotation_count(shard)

        if rt == 0:
            return None

        # VRF-based selection of who leaves
        leaving_results = self.selector.select_validators_to_leave(shard, block_hash, rt)

        if not leaving_results:
            return None

        # Get available validators (not in any shard)
        leaving_ids = {r["validator"].validator_id for r in leaving_results}
        current_shard_ids = {v.validator_id for v in shard.validators}

        available = [
            v for v in self.network.validator_pool.validators
            if v.validator_id not in self.network.validator_pool.assigned
               or v.validator_id in leaving_ids  # Will be unassigned soon
        ]
        available = [v for v in available if v.validator_id not in current_shard_ids or v.validator_id in leaving_ids]

        # VRF-based selection of who joins
        joining_results = self.selector.select_validators_to_join(
            available, shard, block_hash, rt
        )

        # Execute rotation
        for result in leaving_results:
            v = result["validator"]
            shard.remove_validator(v)
            self.network.validator_pool.mark_unassigned(v)

        for result in joining_results:
            v = result["validator"]
            shard.add_validator(v)
            self.network.validator_pool.mark_assigned(v)

        # Record rotation with proofs
        rotation_info = {
            "shard_id": shard.shard_id,
            "block_hash": block_hash[:16],
            "removed_count": len(leaving_results),
            "added_count": len(joining_results),
            "removed": [r["validator"].validator_id for r in leaving_results],
            "added": [r["validator"].validator_id for r in joining_results],
            "leaving_proofs": leaving_results,
            "joining_proofs": joining_results,
            "new_shard_size": shard.get_validator_count()
        }

        self.rotation_history.append(rotation_info)
        return rotation_info

    def verify_rotation(self, rotation_info):
        """
        Verify that a rotation was performed correctly.

        Checks:
        1. VRF outputs for leaving validators are valid
        2. VRF outputs for joining validators are valid
        3. Selection follows the sorting rule
        """
        if rotation_info is None:
            return True

        # Verify leaving selection
        if not self.selector.verify_selection(rotation_info.get("leaving_proofs", [])):
            return False

        # Verify joining selection
        if not self.selector.verify_selection(rotation_info.get("joining_proofs", [])):
            return False

        return True

    def rotate_all_shards(self, block_hash=None):
        """
        Rotate validators in all shards.
        Called once per block.
        """
        if block_hash is None:
            block_hash = self.get_current_block_hash()

        results = []
        for shard_id, shard in self.network.shards.items():
            # Each shard uses same block hash but different shard ID in VRF input
            info = self.rotate_shard(shard, block_hash)
            if info:
                results.append(info)

        return results

    def adjust_rho(self, new_rho):
        """
        Adjust rotation rate.
        Keeps it within min/max bounds.
        """
        old_rho = self.rho
        self.rho = max(self.rho_min, min(self.rho_max, new_rho))
        return self.rho

    def increase_rho(self, delta=0.05):
        """Increase rotation rate by delta."""
        return self.adjust_rho(self.rho + delta)

    def decrease_rho(self, delta=0.05):
        """Decrease rotation rate by delta."""
        return self.adjust_rho(self.rho - delta)

    def get_cumulative_refresh(self, blocks):
        """
        Calculate cumulative refresh ratio over L blocks.
        From paper equation (14): R_cum(L) = 1 - (1-rho)^L

        This tells us what fraction of the committee has been replaced
        after L blocks.

        Example: with rho=0.2 (20% per block)
        - After 5 blocks: 1 - 0.8^5 = 67.2% refreshed
        - After 10 blocks: 1 - 0.8^10 = 89.3% refreshed
        """
        refresh = 1 - ((1 - self.rho) ** blocks)
        return refresh

    def get_stats(self):
        """Get statistics about rotations."""
        if not self.rotation_history:
            return {"total_rotations": 0}

        total_removed = sum(r["removed_count"] for r in self.rotation_history)
        total_added = sum(r["added_count"] for r in self.rotation_history)

        return {
            "total_rotations": len(self.rotation_history),
            "total_validators_removed": total_removed,
            "total_validators_added": total_added,
            "current_rho": self.rho,
            "block_number": self.block_number
        }

    def print_status(self):
        """Print current fluidity status."""
        stats = self.get_stats()
        print(f"\nValidator Fluidity Status:")
        print(f"  Rotation rate (rho): {self.rho:.1%}")
        print(f"  Total rotations: {stats['total_rotations']}")
        print(f"  Validators rotated: {stats['total_validators_removed']}")

        print(f"  Cumulative refresh:")
        for blocks in [5, 10, 20]:
            refresh = self.get_cumulative_refresh(blocks)
            print(f"    After {blocks} blocks: {refresh:.1%}")


# Test code
if __name__ == "__main__":
    import sys

    sys.path.insert(0, '..')

    from sharding import Network

    print("Testing VRF-Based Validator Fluidity")
    print("=" * 55)

    # Create network
    net = Network(num_shards=4, validators_per_shard=25, total_validators=200)
    print(f"Network created: {net}")

    # Create fluidity manager
    fluidity = ValidatorFluidity(net, rho=0.2)
    print(f"Fluidity manager created with rho={fluidity.rho}")

    # Show initial state
    print(f"\nInitial shard state:")
    for sid, shard in net.shards.items():
        print(f"  Shard {sid}: {shard.get_validator_count()} validators")

    # Do rotation on one shard
    print(f"\n" + "-" * 40)
    print("Testing single shard rotation with VRF:")

    shard = net.shards[0]
    block_hash = "test_block_hash_12345"

    rotation_info = fluidity.rotate_shard(shard, block_hash)

    print(f"  Block hash: {rotation_info['block_hash']}")
    print(f"  Validators removed: {rotation_info['removed_count']}")
    print(f"  Validators added: {rotation_info['added_count']}")
    print(f"  Removed IDs: {rotation_info['removed']}")
    print(f"  Added IDs: {rotation_info['added']}")

    # Verify rotation
    is_valid = fluidity.verify_rotation(rotation_info)
    print(f"  Rotation verification: {'PASSED' if is_valid else 'FAILED'}")

    # Do multiple rotations
    print(f"\n" + "-" * 40)
    print("Testing 5 rounds of rotation:")

    for round_num in range(5):
        results = fluidity.rotate_all_shards()
        total_rotated = sum(r["removed_count"] for r in results)

        # Verify all
        all_valid = all(fluidity.verify_rotation(r) for r in results)
        status = "OK" if all_valid else "FAILED"

        print(f"  Round {round_num + 1}: rotated {total_rotated} validators, verification: {status}")

    # Show cumulative refresh
    print(f"\nCumulative refresh after 5 blocks: {fluidity.get_cumulative_refresh(5):.1%}")

    # Show stats
    fluidity.print_status()

    # Test determinism
    print(f"\n" + "-" * 40)
    print("Testing VRF determinism:")

    v = net.validator_pool.validators[0]
    vrf1 = v.vrf_generate("same_input")
    vrf2 = v.vrf_generate("same_input")
    print(f"  Same input produces same output: {vrf1['value'] == vrf2['value']}")

    vrf3 = v.vrf_generate("different_input")
    print(f"  Different input produces different output: {vrf1['value'] != vrf3['value']}")

    print("\nVRF-Based Validator Fluidity test complete!")