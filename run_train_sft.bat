@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo กำลังรัน/กู้คืนการเทรน SFT (สอนให้ตอบคำถาม) outputs_sft/run1 ...
echo ต่อยอดจาก adapter ของ CPT (outputs_large_v3/run1/checkpoint-200)
echo ถ้ามี checkpoint เดิมอยู่แล้วจะกู้คืนอัตโนมัติ ไม่ต้องกังวลว่าจะเริ่มนับหนึ่งใหม่
echo กำลังโหลดโมเดล... หน้าจอจะว่างสักครู่ (ปกติ) อย่าเพิ่งปิดหน้าต่างนี้
echo.
set PYTHONUNBUFFERED=1
"C:\Users\Windows 11\AppData\Local\Programs\Python\Python314\python.exe" src\train_sft.py --config configs\sft.yaml --out outputs_sft\run1 > train_sft.log 2>&1
echo.
echo จบการรัน (หรือ error) — ดูรายละเอียดที่ train_sft.log
pause
