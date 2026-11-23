"""
CFP Protocol module.
Contains: Validator Fluidity, CFI, Cross-Shard, Partitioning, Attack Simulation, Benchmark.
"""

from .fluidity import ValidatorFluidity, VRFBasedSelector
from .cfi import ConsensusFrictionIndex
from .cross_shard import CrossShardCoordinator, CrossShardTransaction, AsyncCommitSimulator
from .partitioning import AccountPartitioner, InteractionGraph
from .attack import (StaticCommitteeSimulator, CFPCommitteeSimulator, AdversaryStrategy,
                     EquilibriumCaptureExperiment, RotationStrategyExperiment,
                     ConsensusMonitoringExperiment, capture_episodes, capture_count)
from .benchmark import ThroughputBenchmark, RotationOverheadBenchmark, CrossShardBenchmark