# Dataset preparation
Place legally obtained files in this directory:
- PEMS03.npz (358 sensors)
- PEMS04.npz (307 sensors)
- PEMS07.npz (883 sensors)
- PEMS08.npz (170 sensors)
- Traffic.csv (862 variables; capitalization is required on Linux)

No download, copying from another project, symlink, or automatic interpolation is performed.
PEMS uses NPZ["data"][:,:,0], chronological 60/20/20 with original history overlap;
Traffic uses the unchanged Dataset_Custom 70/10/20 split. StandardScaler is fit only
on training data. The original row-based forecast-start cycle index is preserved.
Do not redistribute datasets without checking their separate permissions.
Existing local dataset files were retained; no large dataset was copied in migration.
