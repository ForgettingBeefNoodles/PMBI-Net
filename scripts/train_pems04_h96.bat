@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
python -c "import torch, numpy, pandas, sklearn, matplotlib"
if errorlevel 1 (
  echo ERROR: Activate your Python environment and install requirements.
  pause
  exit /b 1
)
python run.py --config configs/pmbi_hparams.py --exp_key PEMS04_96 --model PMBI_Net --is_training 1 %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo Experiment failed; inspect the error above.
pause
exit /b %RC%
