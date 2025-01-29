currency_pairs = [
    'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDUSD',
    'CADCHF', 'CADJPY',
    'CHFJPY',
    'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 'EURUSD',
    'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD', 'GBPUSD',
    'NZDCAD', 'NZDJPY', 'NZDUSD',
    'USDCAD', 'USDCHF', 'USDJPY'
]

currency_pairs_by_timezone = {
    "Tokyo": [
        'AUDJPY', 'CADJPY', 'CHFJPY', 'EURJPY', 'GBPJPY', 'NZDJPY', 'USDJPY'
    ],
    "Zurich": [
        'AUDCHF', 'CADCHF', 'CHFJPY', 'EURCHF', 'GBPCHF', 'USDCHF', 'NZDCHF'
    ],
    "New York": [
        'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'EURUSD', 'GBPUSD'
    ],
    "Sydney": [
        'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDUSD', 'NZDCAD', 'NZDJPY', 'NZDUSD'
    ],
    "Frankfurt": [
        'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 'EURUSD'
    ],
    "London": [
        'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD', 'GBPUSD'
    ],
    "Toronto": [
        'CADCHF', 'CADJPY', 'EURCAD', 'GBPCAD', 'NZDCAD', 'USDCAD'
    ]
}

# Define the open/close times for each market (in 24-hour format)
market_hours = {
    "New York": {'open': '08:30', 'close': '18:00'},
    "Tokyo": {'open': '18:30', 'close': '04:00'},
    "London": {'open': '03:30', 'close': '13:00'},
    "Toronto": {'open': '08:30', 'close': '18:00'},
    "Zurich": {'open': '03:30', 'close': '12:00'},
    "Frankfurt": {'open': '02:30', 'close': '11:00'},
    "Sydney": {'open': '15:30', 'close': '01:00'}
}

from datetime import datetime

def get_current_sessions():
    """Returns a list of active market sessions based on the current time."""
    current_time = datetime.now().strftime('%H:%M')
    active_sessions = []

    for market, hours in market_hours.items():
        open_time = hours['open']
        close_time = hours['close']

        open_time_int = int(open_time.replace(":", ""))
        close_time_int = int(close_time.replace(":", ""))
        current_time_int = int(current_time.replace(":", ""))

        # Handle overlap cases (e.g., Tokyo open at 19:00 and close at 04:00)
        if open_time_int < close_time_int:
            if open_time_int <= current_time_int <= close_time_int:
                active_sessions.append(market)
        else:
            if current_time_int >= open_time_int or current_time_int <= close_time_int:
                active_sessions.append(market)

    return active_sessions


def get_active_currency_pairs():
    """Returns unique active currency pairs based on the current market sessions."""
    active_sessions = get_current_sessions()

    active_currency_pairs = []

    for session in active_sessions:
        if session in currency_pairs_by_timezone:
            active_currency_pairs.extend(currency_pairs_by_timezone[session])

    # Remove duplicates by converting the list to a set, then back to a list
    unique_currency_pairs = list(set(active_currency_pairs))

    return unique_currency_pairs

from tradingview_ta import Interval
M1 = Interval.INTERVAL_1_MINUTE
M5 = Interval.INTERVAL_5_MINUTES
M15 = Interval.INTERVAL_15_MINUTES
M30 = Interval.INTERVAL_30_MINUTES
H1 = Interval.INTERVAL_1_HOUR
H2 = Interval.INTERVAL_2_HOURS
H4 = Interval.INTERVAL_4_HOURS
D1 = Interval.INTERVAL_1_DAY
W1 = Interval.INTERVAL_1_WEEK
MN1 = Interval.INTERVAL_1_MONTH

CONFIG = {
    "exchange": "FX",  # Example exchange
    "screener": "forex",  # Example screener
    "interval": [M1,M5,M15,M30,H1,H2,H4,D1,W1,MN1],  # Example interval placeholder
    "cp_by_time_zone": get_active_currency_pairs()
}
