cd "C:\Users\Blue Code technology\Downloads\ETI_working_core_v0_1\egreed-trade-intelligence"
call .venv\Scripts\pyinstaller --name eti_desktop --onefile --windowed app.py
echo Build complete %errorlevel%
pause