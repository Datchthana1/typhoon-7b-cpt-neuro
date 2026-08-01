@echo off
cd /d "%~dp0"
"C:\Users\Windows 11\AppData\Local\Programs\Python\Python314\python.exe" src\train_cpt.py --config configs\cpt_large_v2.yaml --override epochs=3 --out outputs_large_v2\run1 > train_v2.log 2>&1
