import os

project = r'C:\Users\Blue Code technology\Downloads\ETI_working_core_v0_1\egreed-trade-intelligence'
init_path = os.path.join(project, 'eti', 'market_data', '__init__.py')

with open(init_path, 'r') as f:
    content = f.read()

# Add the import and export if not present
if 'make_provider_config' not in content:
    # Add import after the twelve_data import
    content = content.replace(
        'from .twelve_data import TwelveDataProvider',
        'from .twelve_data import TwelveDataProvider, make_provider_config, register_twelve_data_provider'
    )
    # Add to __all__
    content = content.replace(
        '"TwelveDataProvider",',
        '"TwelveDataProvider",\n    "make_provider_config",\n    "register_twelve_data_provider",'
    )
    with open(init_path, 'w') as f:
        f.write(content)
    print('File updated successfully')
else:
    print('make_provider_config already present')