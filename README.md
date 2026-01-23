# CFP: Continuous Fluidity Protocol for Blockchain Sharding

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Implementation of the Continuous Fluidity Protocol (CFP) for secure blockchain sharding.

## Overview

Sharded blockchains face the **committee capture problem**: adversaries can gradually infiltrate shard committees to compromise consensus. CFP solves this through:

- **Continuous Validator Rotation**: VRF-based probabilistic rotation prevents static committee targeting
- **Consensus Friction Index (CFI)**: Real-time anomaly detection identifies coordinated attacks
- **Cross-Shard Transactions**: Secure 2PC protocol with cryptographic proofs

## Project Structure

```
CFP-Blockchain-Sharding/
├── core/
│   ├── block.py
│   ├── blockchain.py
│   └── transaction.py
├── sharding/
│   ├── validator.py
│   ├── shard.py
│   └── network.py
├── cfp/
│   ├── fluidity.py
│   ├── cfi.py
│   ├── cross_shard.py
│   ├── partitioning.py
│   ├── attack.py
│   └── benchmark.py
├── evaluation/
│   └── publication_evaluation.py
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone
git clone https://github.com/HadisRe/CFP-Blockchain-Sharding.git
cd CFP-Blockchain-Sharding

# Virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Run Evaluation

```bash
python -m evaluation.publication_evaluation
```

**Output:**
- `publication_plots/` - Visualization files 


