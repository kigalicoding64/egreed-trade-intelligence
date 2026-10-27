import sys
import os
os.chdir(r'C:\Users\Blue Code technology\Downloads\ETI_working_core_v0_1\egreed-trade-intelligence')
sys.path.insert(0, '.')
import pytest
sys.exit(pytest.main(['tests/', '-q']))