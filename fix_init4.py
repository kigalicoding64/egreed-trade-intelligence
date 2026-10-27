with open('egreed-trade-intelligence/eti/market_data/__init__.py', 'r') as f:
    content = f.read()

# Add MT5Adapter import after the twelve_data import
content = content.replace(
    'from .twelve_data import TwelveDataProvider, make_provider_config, register_twelve_data_provider',
    'from .twelve_data import TwelveDataProvider, make_provider_config, register_twelve_data_provider\nfrom .mt5_adapter import MT5Adapter'
)

# Add MT5Adapter to __all__
content = content.replace(
    '"TwelveDataProvider",',
    '\"TwelveDataProvider\",\n    \"MT5Adapter\",'
)

with open('egreed-trade-intelligence/eti/market_data/__init__.py', 'w') as f:
    f.write(content)

print('MT5Adapter added to __init__.py')