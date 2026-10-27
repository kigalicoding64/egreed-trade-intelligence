@echo off
cd "C:\Users\Blue Code technology\Downloads\ETI_working_core_v0_1\egreed-trade-intelligence"
call "C:\Users\Blue Code technology\Downloads\ETI_working_core_v0_1\egreed-trade-intelligence\.venv\Scripts\python.exe" -m streamlit run app.py --server.headless=true --server.port=8501 > streamlit_output.txt 2>&1
timeout 10
type streamlit_output.txt
timeout 5
taskkill /f /im python.exe >nul 2>&1