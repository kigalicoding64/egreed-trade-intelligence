import ast
with open(r'C:\Users\Blue Code technology\Downloads\ETI_working_core_v0_1\egreed-trade-intelligence\app.py', 'r') as f:
    ast.parse(f.read())
print('Syntax OK')