# CFP: Consensus Fluidity Protocol for Sharded Blockchains

Simulation code for the paper "Consensus Fluidity Protocol: Preserving Unpredictability and Atomicity in Sharded Blockchains" (M. Sohail, H. Rezaei).

CFP replaces a small number of validators in every shard committee at each block, drawing the incoming members from a standby pool. Cross-shard transfers are decided with certified votes bound to committee snapshots, so a committee change during a transaction cannot break atomicity. A bounded controller can raise the replacement count when a window shows equivocation proofs or missing progress.

## Layout

```
core/         transactions, blocks, chain
sharding/     validators, shards, network
cfp/
  fluidity.py     committee rotation (Eq. 14) and the sampling rule of Alg. 3
  attack.py       capture experiments: static, epoch, alpha-ECFR, CFP; controller scenarios
  cfi.py          CFI (Eq. 16) and the bounded rotation controller (Alg. 4)
  cross_shard.py  discrete-event simulation of the certified commit (Alg. 5) and a timeout 2PC baseline
  analysis.py     closed forms of Section V (Eqs. 31-39)
  placement.py    account placement (Alg. 1) with hash, Tangram (jump hash), CLPA and Metis baselines
  throughput.py   throughput/latency model for relay, Atomix, RapidChain, broker and CFP commit
  messages.py     per-block message and byte accounting
  partitioning.py, benchmark.py   earlier helpers
eval_revision.py          runs all experiments of Section VI, writes revision_results/*.csv
make_revision_figures.py  draws the figures from the CSV files
verify_formulas.py        compares the Section V formulas with simulation
tests/                    functional tests
data/                     place the Ethereum transfer trace here (see below)
```

## Running

```
pip install -r requirements.txt
python verify_formulas.py
python eval_revision.py                 # all experiments (several hours on 4 cores)
python eval_revision.py fig6 fig7_12    # selected experiments only
python make_revision_figures.py
```

Add `--quick` for a short run with fewer seeds and blocks. Each test in `tests/` can be run directly, e.g. `python tests/test_phase1.py`.

## Figures in the paper

Results are written to `revision_results/` and figures to `revision_results/figures/`.

| Paper figure | File in revision_results/figures |
|---|---|
| Fig. 3  | fig3_cumulative_refresh |
| Fig. 4  | fig4_rho_sensitivity |
| Fig. 5  | fig5_cfi_values |
| Fig. 6  | fig6_capture_rate |
| Fig. 7  | fig7_capture_duration |
| Fig. 8  | fig8_controller_response |
| Fig. 9  | fig9_performance |
| Fig. 10 | fig10_placement |
| Fig. 11 | fig11_shards |
| Fig. 12 | fig12_comparison |
| Fig. 13 | fig13_stress_timeline |
| Fig. 14 | fig14_profile |

## Data

The placement and throughput experiments use `selectedTxs_300K.csv`, 300,000 Ethereum transfers distributed with BlockEmulator under the MIT licence (see `data/BlockEmulator_LICENSE`). Download it into `data/`:

```
curl -L -o data/selectedTxs_300K.csv https://raw.githubusercontent.com/HuangLab-SYSU/block-emulator/main/selectedTxs_300K.csv
```

Placement plans use the first 150,000 transfers and are evaluated on the remaining 150,000.

## Licence

See `LICENSE`.
