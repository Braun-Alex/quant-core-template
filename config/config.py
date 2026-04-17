import os
from dotenv import load_dotenv

load_dotenv()

BINANCE_CONFIG = {
    'apiKey': os.getenv('BINANCE_TESTNET_API_KEY'),
    'secret': os.getenv('BINANCE_TESTNET_SECRET'),
    'sandbox': True,
    'options': {
        'defaultType': 'spot',
    },
    'enableRateLimit': True
}
