import numpy as np
import MetaTrader5 as MeTa
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

@retry(retry=retry_if_exception_type(Exception), stop=stop_after_attempt(5), wait=wait_fixed(0.5))
def fetch_data(symbol, data_points = 200, timeframe = MeTa.TIMEFRAME_D1):
    MeTa.initialize()
    ohlc_data = MeTa.copy_rates_from_pos(symbol, timeframe, 1, data_points)

    # Ensure data is available
    if ohlc_data is not None and len(ohlc_data) > 0:
        h_o_c_l = [
            np.array([data['high'], data['open'], data['close'], data['low']]).reshape(4,1)
            for data in ohlc_data
        ]

        # Encode candlestick data: 1 for bullish, 0 for bearish, and 2 for neutral
        encoded_candles = []
        for data in ohlc_data:
            if data['open'] < data['close']:
                encoded_candles.append(1)  # Bullish
            elif data['open'] > data['close']:
                encoded_candles.append(0)  # Bearish
            else:
                encoded_candles.append(2)  # Neutral

        encoded_candles_str = ''.join(map(str, encoded_candles))

        # Generate Features Matrix
        features_matrix = []
        for each_vector in h_o_c_l:
            diff_matrix = each_vector - each_vector.T  # Restore your original logic
            percent_change_matrix = (diff_matrix / each_vector.flatten()) * 100
            features_matrix.append(percent_change_matrix)

        return features_matrix, ohlc_data, encoded_candles_str

    else:
        raise RuntimeError("Data is not available")





if __name__ =="__main__":
    s ="GBPCAD"
    X_features,X_data,X_encoded = fetch_data(symbol=s)