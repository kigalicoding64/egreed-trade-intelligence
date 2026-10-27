import os
project = r'C:\Users\Blue Code technology\Downloads\ETI_working_core_v0_1\egreed-trade-intelligence'
mm_path = os.path.join(project, 'eti', 'market_data', '__init__.py')
_mm_path = os.path.join(project, 'eti', 'market_data', '_', '__init__.py')

for p in [mm_path, _mm_path]:
    if os.path.exists(p):
        with open(p, 'r') as f:
            content = f.read()
        print(f"File: {p}")
        print(content)
    else:
        print(f"File not found: {p}")