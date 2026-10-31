# Local release preparation review (no publication performed)

- LICENSE is byte-identical to the source Apache-2.0 file. NOTICE retains upstream
  attribution and identifies local modifications. No third-party license was replaced.
- The PMBI-Net additions' authorship and permission to distribute need author confirmation;
  dataset and checkpoint rights require separate review. Do not assume Apache-2.0 covers them.
- Active Python/config/BAT/shell/JSON files were scanned for machine-specific absolute
  paths and obvious credential assignments; no hits. This is a bounded pattern scan,
  not a guarantee that human review is unnecessary.
- No account credentials, user email or machine interpreter paths were intentionally copied
  into release code. Environment metadata records package/GPU versions only.
- Exclude dataset contents, runs, checkpoints, results, migration_checks, and
  _pre_migration_backup. .gitignore covers these. The backup is a private preservation
  area and contains old code/logs; it is NOT part of the PMBI-Net release.
- Formal PMBI_Net plus original TQNet backbone and two unchanged legacy reference files are retained. No unrelated model registration.
- No git init, remote creation, upload or push was performed.

## Source signature caveat
All 20 completed runs retain matching model, loader and Exp_Pyramid training-class
source signatures. Historical run.py / registry / V2 orchestration signatures differ
from the later source project after additional experiments were added.
utils/tools.py differs from the historical hash by exactly np.Inf -> np.inf; the
preserved earlier file matches the recorded historical hash. The migrated file is
byte-identical to the current source. The formal Exp_Pyramid early-stopping logic
is inside the unchanged trainer, so it does not call this legacy EarlyStopping class.
The current formal training class, optimizer/data methods and LR function were retained;
full PEMS04 H96 evaluation and exact fixed-input checkpoint outputs were verified.
Do not describe every historical entry file as byte-identical.

Naming refactor checks and private backups are excluded under refactor_checks/.
The original LICENSE is unchanged. See REFACTOR_REPORT.txt for this refactor.
