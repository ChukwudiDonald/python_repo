from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
import pandas as pd
import MetaTrader5 as MeTa
import numpy as np
import re

@retry(retry=retry_if_exception_type(Exception), stop=stop_after_attempt(5), wait=wait_fixed(0.5))
def fetch_ohlc_data(s, data_points=10, timeframe=MeTa.TIMEFRAME_H1):
    """Fetch OHLC data from MetaTrader."""
    MeTa.initialize()
    ohlc_data = MeTa.copy_rates_from_pos(s, timeframe, 1, data_points)

    if ohlc_data is not None and len(ohlc_data) > 0:
        return pd.DataFrame(ohlc_data)  # Return raw OHLC data
    else:
        raise RuntimeError("Data is not available")

def expand_candlestick_features(df):
    """Expand OHLC features with percentage differences & candlestick type."""
    df['High_vs_Open_%'] = (df['high'] - df['open']) / df['open'] * 100
    df['High_vs_Close_%'] = (df['high'] - df['close']) / df['close'] * 100
    df['High_vs_Low_%'] = (df['high'] - df['low']) / df['low'] * 100

    df['Open_vs_High_%'] = (df['open'] - df['high']) / df['high'] * 100
    df['Open_vs_Close_%'] = (df['open'] - df['close']) / df['close'] * 100
    df['Open_vs_Low_%'] = (df['open'] - df['low']) / df['low'] * 100

    df['Close_vs_High_%'] = (df['close'] - df['high']) / df['high'] * 100
    df['Close_vs_Open_%'] = (df['close'] - df['open']) / df['open'] * 100
    df['Close_vs_Low_%'] = (df['close'] - df['low']) / df['low'] * 100

    df['Low_vs_High_%'] = (df['low'] - df['high']) / df['high'] * 100
    df['Low_vs_Open_%'] = (df['low'] - df['open']) / df['open'] * 100
    df['Low_vs_Close_%'] = (df['low'] - df['close']) / df['close'] * 100

    # **Balanced Candlestick Encoding**: +1 = Bullish, -1 = Bearish, 0 = Neutral
    df['Candle_Type'] = np.where(df['open'] < df['close'], 1,
                                 np.where(df['open'] > df['close'], -1, 0))

    return df

def encode_candlesticks(df):
    """Encode candlestick patterns: 1 for bullish, 0 for bearish, 2 for neutral.

    Returns a string representation without modifying the original DataFrame.
    """
    candle_types = np.where(df['open'] < df['close'], '1',
                            np.where(df['open'] > df['close'], '0', '2'))

    return ''.join(candle_types.tolist())  # Convert to string

def prepare_rnn_data(symbol, patterns, HTF_timeframe, LTF_timeframe):
    """
    Prepares the dataset for RNN by detecting patterns in the higher timeframe (HTF),
    mapping them to the lower timeframe (LTF), and returning a structured DataFrame.

    Parameters:
    - symbol (str): Trading symbol (e.g., 'EUR-USD').
    - patterns (list of str): List of regex patterns to detect in HTF-coded data.
    - HTF_timeframe (str): Higher timeframe for pattern detection (e.g., MeTa.TIMEFRAME_H4).
    - LTF_timeframe (str): Lower timeframe for mapping (e.g., MeTa.TIMEFRAME_H1).

    Returns:
    - pd.DataFrame: LTF data with 'Pattern_Flag' column.
    """
    # Fetch HTF data
    HTF_dataframe = fetch_ohlc_data(s=symbol, data_points=20, timeframe=HTF_timeframe)
    HTF_coded = encode_candlesticks(HTF_dataframe)
    HTF_ranges = []
    HTF_last_index = len(HTF_dataframe) - 1

    # Fetch LTF data and expand features
    LTF_dataframe = fetch_ohlc_data(s=symbol, data_points=80, timeframe=LTF_timeframe)
    LTF_dataframe = expand_candlestick_features(LTF_dataframe)
    LTF_dataframe['Pattern_ID'] = 0


    for pattern_id, pattern_regex in enumerate(patterns, start=1):
        compiled_pattern = re.compile(pattern_regex)
        matches = compiled_pattern.finditer(HTF_coded)

        for match in matches:
            s, e = match.start(), match.end()  # Start and end indices

            # Boundary check for start timestamp
            start_index = min(s + 2, HTF_last_index)
            start_timestamp = HTF_dataframe.iloc[start_index]['time']

            # Boundary check for end timestamp
            end_index = min(e, HTF_last_index)
            end_timestamp = HTF_dataframe.iloc[end_index]['time']

            HTF_ranges.append((start_timestamp, end_timestamp, pattern_id))  # Store with unique pattern ID


    # Apply pattern IDs to LTF data
    for start_ts, end_ts, pattern_id in HTF_ranges:
        LTF_dataframe.loc[(LTF_dataframe['time'] >= start_ts) & (LTF_dataframe['time'] <= end_ts), 'Pattern_ID'] = pattern_id

    return LTF_dataframe


if __name__ == "__main__":

    bullish_pattern = r"01[12]1{3,}"
    bearish_pattern = r"10[02]0{3,}"


    X_data_frame = prepare_rnn_data(
        symbol="EURUSD",
        patterns=[bullish_pattern,bearish_pattern],
        HTF_timeframe=MeTa.TIMEFRAME_H4,
        LTF_timeframe=MeTa.TIMEFRAME_H1
    )

    print(X_data_frame)




