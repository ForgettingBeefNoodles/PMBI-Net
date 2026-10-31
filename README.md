# PMBI-Net

**Periodic Prior-Guided Multi-Granularity Bidirectional Interaction Network for Traffic Flow Forecasting**

## Overview and relationship to TQNet
PMBI-Net builds upon TQNet and extends it with multi-granularity temporal feature encoding and heterogeneous bidirectional interaction.

The formal implementation is **models/PMBI_Net.py**, registered as **PMBI_Net**.
It defines the complete model constructor and forward; it is not an alias or wrapper
around the legacy V2. It inherits the real original backbone in models/TQNet.py.
The original TQNet paper and repository are credited below.

### MTFE: multi-granularity temporal feature encoding
- P24: four ordered 24-step patches from the same normalized 96-step history;
  shared projection, position embedding and per-variable local patch attention.
- P96: independent full-window representation, not an assertion of a 96-step natural period.
- TQ context: original periodic Query / history Key and Value cross-sensor attention,
  then its original context projection. There is only one full cross-sensor TQ attention.

### HBI: heterogeneous bidirectional interaction
Bottom-Up aggregates P24 into P96 and then the TQ-context representation.
Top-Down feeds the updated context back to the window and individual patches, with
historical periodic phase Phi. Joint Readout retains the exact order
[UP, D96, flatten(D24)] (6r) and the original residual/forecast backbone.
RevIN, initialization, parameter names, RNG preservation, dropout and operations
are unchanged. utils/pmbi_helpers.py contains the original preserve_rng and mlp functions.

## Environment Setup
Verified Python 3.11.10, PyTorch 2.6.0+cu124, CUDA 12.4, RTX 4060 Ti.
Install a suitable official CPU/CUDA PyTorch build, activate your environment, then:
```
python -m pip install -r requirements.txt
```
GPU0 is the default; CPU fallback is supported. Use --use_gpu false explicitly if needed.
No machine-specific interpreter path is stored in scripts. Exact versions:
provenance/environment.json.

## Dataset Preparation
See dataset/README.md. Place files locally: PEMS03.npz, PEMS04.npz, PEMS07.npz,
PEMS08.npz and Traffic.csv (capital T, important on Linux).
No cross-project imports, automatic copying, symlinks or resampling.
The original split, training-only StandardScaler, RevIN setting and forecast-start
row-based cycle_index are preserved. Dataset contents are excluded from Git.

## Formal configuration
configs/pmbi_hparams.py combines COMMON_CONFIG, DATASET_CONFIGS and TASK_OVERRIDES.
CLI > selected configuration > parser defaults. All 20 formal configurations remain
unchanged except model=TQNet_Pyramid_V2 -> model=PMBI_Net. Provenance is preserved in
provenance/formal_configs.json (its historical model names are intentionally not rewritten).
Supported: PEMS03/04/07/08 each H12/H24/H48/H96; Traffic H96/H192/H336/H720.
Internal pyramid_v2_variant=v2_full remains for exact compatibility.
```
python run.py --config configs/pmbi_hparams.py --exp_key PEMS04_96 --check_config
```

## Training
```
python run.py --config configs/pmbi_hparams.py --exp_key PEMS04_96 --is_training 1
```
Equivalent explicit model selection:
```
python run.py --config configs/pmbi_hparams.py --exp_key PEMS04_96 --model PMBI_Net --is_training 1
```
Windows: scripts/train_pems04_h96.bat
Linux: bash scripts/train_pems04_h96.sh
Change exp_key for another supported task. Append --resume for complete epoch-boundary
model/optimizer/scheduler/AMP/early-stopping/RNG recovery. No per-epoch test selection.
New output IDs contain PMBI_Net and configuration/source/data fingerprints under runs/.
Completed exact runs are verified and skipped, never silently overwritten.

## Evaluation of new runs
```
python run.py --config configs/pmbi_hparams.py --exp_key PEMS04_96 --is_training 0
```
Windows: scripts/test_pems04_h96.bat
Linux: bash scripts/test_pems04_h96.sh
Loads the matching new run's best checkpoint. A missing checkpoint causes an explicit
error; test-only never trains. Completed evaluation artifacts are preserved.
Commands also work from another working directory when run.py is specified by its path.

## Explicit compatibility evaluation of an old checkpoint
The historical reference remains separate from new training. It is not renamed or edited.
```
python run.py --config configs/pmbi_hparams.py --exp_key PEMS04_96 --model PMBI_Net --is_training 0 --pyramid_checkpoint checkpoints/reference_PEMS04_96/best.pth
```
This route verifies original config, checkpoint SHA256, dataset SHA256 and saved scaler;
uses the unchanged evaluation algorithm on an isolated checkpoint copy; and writes only
under runs/compat_evaluations/. Historical epochs are identified as imported provenance,
not new training. It is forbidden with is_training=1. The reference is excluded from Git.
For the legacy-model comparison, use the same command with --model TQNet_Pyramid_V2.
The old model files remain unchanged solely for compatibility checks.

## Reproducibility checks
```
python scripts/verify_pmbi_refactor.py
python scripts/verify_portability.py --evaluate_reference
```
No training or optimizer step occurs. The refactor test checks independent-seed
initialization, every state_dict entry/shape/value and requires_grad, strict checkpoint
loading, CPU prediction equivalence, matched-dropout forward/backward equivalence,
and paired complete-test evaluation when local data/weights are available.
Absent artifacts are reported NOT TESTED, never fabricated.
Measured PEMS04 H96: both models 1,564,320 parameters, 62 state_dict entries;
initialization exact; CPU and full-test prediction max absolute difference 0;
MSE 0.10294593979487245 and MAE 0.20917778096668083 on all 3,303 windows.
See REFACTOR_REPORT.txt and refactor_checks/. No full training was run during refactoring.
New training on other environments may exhibit numerical variation.

## Code organization and preserved history
models/PMBI_Net.py is the formal complete implementation, inheriting models/TQNet.py.
utils/pmbi_helpers.py is its only extracted helper dependency; it does not import an
early Pyramid model. TQNet_Pyramid_V2.py and TQNet_Pyramid.py remain legacy references.
The training class in exp/exp_pyramid.py, data loaders, metrics, optimizer and learning
rate algorithm are untouched. Only registry/configuration/output compatibility changed.
Private pre-migration backups and historical reports are retained, excluded from Git,
and are not runtime dependencies. Do not include them in a public release.

## License and Citation Notes
Original TQNet Apache-2.0 LICENSE and attribution are retained; see NOTICE and
PUBLICATION_REVIEW.md. This does not assign authorship of upstream code to PMBI-Net.
Authorization for local research additions and redistribution of data/weights still
requires author review. No GitHub repository or upload was created.
[TQNet official code](https://github.com/ACAT-SCUT/TQNet)
```
@inproceedings{lin2025TQNet,
  title={Temporal Query Network for Efficient Multivariate Time Series Forecasting},
  author={Lin, Shengsheng and Chen, Haojun and Wu, Haijie and Qiu, Chunyun and Lin, Weiwei},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```
PMBI-Net publication citation: pending; no DOI, acceptance or publication date is asserted.
