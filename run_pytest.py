import subprocess, sys
venv = r'C:\Users\Blue Code technology\Downloads\ETI_working_core_v0_1\egreed-trade-intelligence\.venv\Scripts\python.exe'
result = subprocess.run(
    [venv, 'tests.py', '-q'],
    capture_output=True, text=True,
    cwd=r'C:\Users\Blue Code technology\Downloads\ETI_working_core_v0_1\egreed-trade-intelligence'
)
print('STDOUT:', result.stdout[-300:])
print('STDERR:', result.stderr[-300:])
print('RC:', result.returncode)