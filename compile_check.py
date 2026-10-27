import py_compile
import os
import sys

project_dir = r'C:\Users\Blue Code technology\Downloads\ETI_working_core_v0_1\egreed-trade-intelligence'
errors = []

for root, dirs, files in os.walk(project_dir):
    # Skip .venv and .git
    if '.venv' in root or '.git' in root:
        continue
    for f in files:
        if f.endswith('.py'):
            filepath = os.path.join(root, f)
            try:
                py_compile.compile(filepath, doraise=True)
            except py_compile.PyCompileError as e:
                errors.append(f"{filepath}: {e}")

if errors:
    print(f"Syntax errors found in {len(errors)} files:")
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print("All project .py files compile successfully.")
    sys.exit(0)