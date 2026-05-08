# Offline RL Scratch Laboratory 🧪

This directory contains the consolidated diagnostic and verification suites for the Charity Gas Optimization pipeline. All legacy proof-of-concept scripts have been merged into these 4 professional suites.

## 🚀 The Omega Training Guard
Before running any training experiment, ensure all suites pass.

| Suite | Purpose | Key Checks |
| :--- | :--- | :--- |
| `logic_integrity_suite.py` | Physical Parity | Validates that Dataset Builder and Environment share 100% identical physics. |
| `training_audit_suite.py` | Signal Integrity | Checks normalization sync, reward magnitudes, and D3RLPy v2.x compatibility. |
| `oracle_verification_suite.py` | Economic Alignment | Verifies that the Hindsight Oracle's optimal choices match Environment rewards. |
| `data_diagnostics_suite.py` | Data Health | Detects schema anomalies, missing references, and NaN values in parquet files. |

## 🛠️ Usage
To run the full suite:
```bash
./venv/bin/python scratch/logic_integrity_suite.py
./venv/bin/python scratch/training_audit_suite.py
./venv/bin/python scratch/oracle_verification_suite.py
./venv/bin/python scratch/data_diagnostics_suite.py
```

## ⚠️ Economic Calibration (Hardened)
The pipeline has been calibrated with the following "Hardened" constants:
- **Deadline Penalty**: `2,000,000.0` (2.0 scaled points).
- **Urgency Beta**: `100.0`.
- **Reward Scale**: `1,000,000.0`.

These values ensure that the agent perceives a deadline miss as much more costly than gas fees, preventing passive waiting behavior.

## 📈 Status: 100% HARDENED
As of the last audit, all suites pass with 100% logical and mathematical parity.
