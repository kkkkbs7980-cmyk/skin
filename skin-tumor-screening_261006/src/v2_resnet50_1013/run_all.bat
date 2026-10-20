@echo off
REM 3-class (normal/benign/malignant) experiments. Run from Anaconda Prompt after 'conda activate <env>'.
REM Each experiment: train -> evaluate -> compare. If one fails, the next one still runs.
cd /d %~dp0
set PYTHONUTF8=1

echo ===== [0/10] prepare split =====
python src/prepare_split.py --data data
if errorlevel 1 (
  echo prepare_split.py FAILED. Send a screenshot of the error above.
  pause
  exit /b 1
)

echo ===== [1/10] 00_baseline =====
python src/train.py --data data --out outputs_3cls/00_baseline
python src/evaluate.py --data data --ckpt outputs_3cls/00_baseline/best.pt
python src/compare.py --out outputs_3cls

echo ===== [2/10] 01_flip =====
python src/train.py --data data --out outputs_3cls/01_flip --aug flip
python src/evaluate.py --data data --ckpt outputs_3cls/01_flip/best.pt
python src/compare.py --out outputs_3cls

echo ===== [3/10] 02_crop =====
python src/train.py --data data --out outputs_3cls/02_crop --aug crop
python src/evaluate.py --data data --ckpt outputs_3cls/02_crop/best.pt
python src/compare.py --out outputs_3cls

echo ===== [4/10] 03_color =====
python src/train.py --data data --out outputs_3cls/03_color --aug color
python src/evaluate.py --data data --ckpt outputs_3cls/03_color/best.pt
python src/compare.py --out outputs_3cls

echo ===== [5/10] 04_cutmix =====
python src/train.py --data data --out outputs_3cls/04_cutmix --aug cutmix
python src/evaluate.py --data data --ckpt outputs_3cls/04_cutmix/best.pt
python src/compare.py --out outputs_3cls

echo ===== [6/10] 05_combo =====
python src/train.py --data data --out outputs_3cls/05_combo --aug flip crop cutmix
python src/evaluate.py --data data --ckpt outputs_3cls/05_combo/best.pt
python src/compare.py --out outputs_3cls

echo ===== [7/10] 06_hair =====
python src/train.py --data data --out outputs_3cls/06_hair --aug hair
python src/evaluate.py --data data --ckpt outputs_3cls/06_hair/best.pt
python src/compare.py --out outputs_3cls

echo ===== [8/10] 07_vignette =====
python src/train.py --data data --out outputs_3cls/07_vignette --aug vignette
python src/evaluate.py --data data --ckpt outputs_3cls/07_vignette/best.pt
python src/compare.py --out outputs_3cls

echo ===== [9/10] 08_erase =====
python src/train.py --data data --out outputs_3cls/08_erase --aug erase
python src/evaluate.py --data data --ckpt outputs_3cls/08_erase/best.pt
python src/compare.py --out outputs_3cls

echo ===== [10/10] 09_elastic =====
python src/train.py --data data --out outputs_3cls/09_elastic --aug elastic
python src/evaluate.py --data data --ckpt outputs_3cls/09_elastic/best.pt
python src/compare.py --out outputs_3cls

echo ===== ALL DONE. Open outputs_3cls\(result table .md file) =====
pause
