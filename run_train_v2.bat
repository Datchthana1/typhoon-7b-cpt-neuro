@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo กำลังรัน/กู้คืนการเทรน outputs_large_v2/run1 ...
echo ถ้ามี checkpoint เดิมอยู่แล้วจะกู้คืนอัตโนมัติ ไม่ต้องกังวลว่าจะเริ่มนับหนึ่งใหม่
echo กำลังโหลดโมเดล... หน้าจอจะว่างสักครู่ (ปกติ) อย่าเพิ่งปิดหน้าต่างนี้
echo.
"C:\Users\Windows 11\AppData\Local\Programs\Python\Python314\python.exe" src\train_cpt.py --config configs\cpt_large_v2.yaml --override epochs=3 --out outputs_large_v2\run1 > train_v2.log 2>&1
echo.
echo จบการรัน (หรือ error) — ดูรายละเอียดที่ train_v2.log
pause
 




  