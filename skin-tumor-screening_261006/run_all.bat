@echo off
REM 3rd round (10/20): ConvNeXt-Tiny 6 + ResNet50 2, 50 epochs, batch 16, AdamW, AMP(bf16).
REM Run from Anaconda Prompt after 'conda activate <env>'. Results go to outputs_3cls_v3\ (2nd round outputs_3cls\ untouched).
REM Each experiment: train -> evaluate -> compare. If one fails, the next one still runs.
REM Internet is needed on the first run (ConvNeXt weights, about 110MB).
cd /d %~dp0
set PYTHONUTF8=1
set OUT=outputs_3cls_v3

echo ===== [0/8] prepare split =====
python src/prepare_split.py --data data
if errorlevel 1 (
  echo prepare_split.py FAILED. Send a screenshot of the error above.
  pause
  exit /b 1
)

echo ===== [smoke] 1-epoch practice run (ConvNeXt, combo aug) - about 1-2 min =====
python src/train.py --data data --out %OUT%/_smoke --arch convnext_tiny --epochs 1 --aug flip rotate crop cutmix
if errorlevel 1 (
  echo Practice run FAILED. Main experiments NOT started. Send a screenshot of the error above.
  pause
  exit /b 1
)
python src/evaluate.py --data data --ckpt %OUT%/_smoke/best.pt
if errorlevel 1 (
  echo Practice evaluate FAILED. Main experiments NOT started. Send a screenshot of the error above.
  pause
  exit /b 1
)
echo Practice run OK. Starting main experiments.

echo ===== [1/8] 00_baseline =====
python src/train.py --data data --out %OUT%/00_baseline --arch convnext_tiny
python src/evaluate.py --data data --ckpt %OUT%/00_baseline/best.pt
python src/compare.py --out %OUT%

echo ===== [2/8] 01_flip =====
python src/train.py --data data --out %OUT%/01_flip --arch convnext_tiny --aug flip
python src/evaluate.py --data data --ckpt %OUT%/01_flip/best.pt
python src/compare.py --out %OUT%

echo ===== [3/8] 02_rotate =====
python src/train.py --data data --out %OUT%/02_rotate --arch convnext_tiny --aug rotate
python src/evaluate.py --data data --ckpt %OUT%/02_rotate/best.pt
python src/compare.py --out %OUT%

echo ===== [4/8] 03_crop =====
python src/train.py --data data --out %OUT%/03_crop --arch convnext_tiny --aug crop
python src/evaluate.py --data data --ckpt %OUT%/03_crop/best.pt
python src/compare.py --out %OUT%

echo ===== [5/8] 04_cutmix =====
python src/train.py --data data --out %OUT%/04_cutmix --arch convnext_tiny --aug cutmix
python src/evaluate.py --data data --ckpt %OUT%/04_cutmix/best.pt
python src/compare.py --out %OUT%

echo ===== [6/8] 05_combo =====
python src/train.py --data data --out %OUT%/05_combo --arch convnext_tiny --aug flip rotate crop cutmix
python src/evaluate.py --data data --ckpt %OUT%/05_combo/best.pt
python src/compare.py --out %OUT%

echo ===== [7/8] 06_resnet50_baseline =====
python src/train.py --data data --out %OUT%/06_resnet50_baseline --arch resnet50
python src/evaluate.py --data data --ckpt %OUT%/06_resnet50_baseline/best.pt
python src/compare.py --out %OUT%

echo ===== [8/8] 07_resnet50_combo =====
python src/train.py --data data --out %OUT%/07_resnet50_combo --arch resnet50 --aug flip rotate crop cutmix
python src/evaluate.py --data data --ckpt %OUT%/07_resnet50_combo/best.pt
python src/compare.py --out %OUT%

echo ===== ALL DONE. Open outputs_3cls_v3\(result table .md file) =====
pause
