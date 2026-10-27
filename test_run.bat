@echo off
cd "C:\Users\Blue Code technology\Downloads\ETI_working_core_v0_1\egreed-trade-intelligence"
C:\Users\Blue Code technology\Downloads\ETI_working_core_v0_1\egreed-trade-intelligence\.venv\Scripts\python.exe -m pytest tests/ -q > test_output.txt 2>&1
type test_output.txt