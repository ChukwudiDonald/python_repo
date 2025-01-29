from tradingview_ta import TA_Handler
import logging,threading,time,math
import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd
import matplotlib.dates as m_dates
from tenacity import retry,retry_if_exception_type,stop_after_attempt,wait_fixed,RetryError

class MT5Account:
    def __init__(self):
        # Account details initialized to None
        self.__report = {}             # A report dictionary to store relevant account data
        self.__positions = []          # List to hold positions associated with this account
        self.__account_ID = 0       # Account ID
        self.__open_position_status = 0       # Total profit/loss status of the account
        self.__account_balance = 0
        self.__account_server = None   # Server details
        self.__account_password = None # Account password
        self.__closed_position_status = 0

    def get_account_id(self):
        """
        Returns the account ID.
        """
        return self.__account_ID

    def get_report(self):
        """
        Returns the account report.
        """
        self.update_report()  # Automatically update positions before returning
        return self.__report

    def get_balance(self):
        """Return the balance of the current mt5_account """
        self.update_balance()  # Automatically update balance before returning
        return self.__account_balance

    def get_positions(self):
        """
        Returns the list of positions.
        """
        self.update_positions()  # Automatically update positions before returning
        return self.__positions

    def update_report(self):
        """
        Updates the account report with the current positions and profit status.
        """
        self.__report = {
            "positions": self.__positions,
        }

    def update_balance(self):
        """Update the balance of the current mt5_account """
        self.__account_balance = mt5.account_info().balance

    def update_positions(self):
        """
        Updates the positions associated with this account by retrieving them from MT5.
        """
        # Retrieve all open positions from MT5
        positions = mt5.positions_get()

        # If there are no positions, exit without updating anything
        if not positions:
            return

        # Update the __positions with all details from MT5 positions
        self.__positions = [{
            "symbol": position.symbol,
            "profit": position.profit,
            "lot_size": position.volume,
            "entry_price": position.price_open,
            "current_price": position.price_current,
            "stop_loss": position.sl,
            "take_profit": position.tp,
            "type": position.type,
            "ticket": position.ticket,
            "open_time": position.time,
            "swap": position.swap,
        } for position in positions]

    def get_open_position_status(self):
        """
        Returns the total profit/loss status.
        """
        self.update_open_position_status()  # Automatically update profit_status before returning
        return self.__open_position_status

    def update_open_position_status(self):
        """
        Updates the profit status based on the current positions.
        """
        self.update_positions()
        self.__open_position_status = sum(position["profit"] for position in self.__positions)

    def display_account_info(self):
        """
        Displays the current account details.
        """
        if self.__account_ID and self.__account_server and self.__account_password:
            print(f"Account Server: {self.__account_server}")
            print(f"Account ID: {self.__account_ID}")
            # Avoid printing the password for security reasons
            print("Account Password: ********")
        else:
            print("Account details are incomplete.")

    def get_basic_account_info(self):
        """
        Returns only the server, account ID, and password details.
        """
        return {
            "account_server": self.__account_server,
            "account_ID": self.__account_ID,
            "account_password": self.__account_password
        }

    def update_basic_info(self, server: str, account_id, password: str):
        """Update the basic account information (server, ID, password)."""
        self.__account_server = server
        self.__account_ID = account_id
        self.__account_password = password

    def update_closed_position_status(self,from_date=None, to_date=None):
        # Default date range (starting from 1st Jan 2000 if not provided)

        if from_date is None:
            from_date = datetime(2000, 1, 1)
        if to_date is None:
            to_date = datetime.now()

        # Fetch trading history deals
        deals = mt5.history_deals_get(from_date, to_date)

        if deals is None:
            return 0.0

        # Calculate total profit/loss
        total_profit_or_loss = 0
        for deal in deals:
            if deal.position_id and deal.price and deal.volume:
                total_profit_or_loss += deal.profit


        self.__closed_position_status = total_profit_or_loss

    def get_closed_position_status(self,from_date=None, to_date=None):
        self.update_closed_position_status(from_date,to_date)
        return self.__closed_position_status

class Notify:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:  # Ensure thread-safety
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initializes the NotificationManager, sets up logging, and configures log format.
        """
        # Set up logging with a specific format and level
        self.logger = logging.getLogger("Notify")
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s")

    def accomplish_task(self, message, level="info"):
        """
        Send a notification and log it at the specified level.

        Args:
            message (str): The notification message.
            level (str): The log level for the notification. Defaults to 'info'.
                         Valid options are 'info', 'warning', 'error'.
        """
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        else:
            self.logger.info(f"Invalid level '{level}' provided. Defaulting to 'info'.")
            self.logger.info(message)

    def send_error_notification(self, message):
        """
        Send an error notification and log it as an ERROR.

        Args:
            message (str): The error notification message.
        """
        self.accomplish_task(message, level="error")
        exit()

    def send_warning_notification(self, message):
        """
        Send a warning notification and log it as a WARNING.

        Args:
            message (str): The warning notification message.
        """
        self.accomplish_task(message, level="warning")

    def send_info_notification(self, message):
        """
        Send an informational notification and log it as INFO.

        Args:
            message (str): The info notification message.
        """
        self.accomplish_task(message, level="info")

    @staticmethod
    def send_console_message(message):
        """
        Print a message with the format 'hours:minutes:seconds - message'.

        Args:
            message (str): The message to print.
        """
        current_time = time.strftime("%H:%M:%S")
        print(f"{current_time} - {message}")

class Interpreter:
    @staticmethod
    def interpret_recommendation(recommendation):
        """
        Map a raw recommendation to a standard action: BUY, SELL, or NEUTRAL.
        """
        normalized_recommendation = recommendation.upper()
        if normalized_recommendation in {"BUY", "STRONG_BUY"}:
            return "BUY"
        elif normalized_recommendation in {"SELL", "STRONG_SELL"}:
            return "SELL"
        return None

    @staticmethod
    def process_signal_list(signal_list):
        """
        Parse a list of trading signals and standardize their recommendations.
        """
        standardized_signals = []
        for signal in signal_list:
            for symbol, details in signal.items():
                recommendation = details["RECOMMENDATION"].upper()
                action = Interpreter.interpret_recommendation(recommendation)
                standardized_signals.append({symbol: action})
        return standardized_signals

    @staticmethod
    def process_signal_pip_count(text):
        if text.endswith(("USD", "CHF", "CAD")):
            return 10.0
        elif text.endswith("GBP"):
            return 7.5
        elif text.endswith(("AUD", "NZD")):
            return  25.0
        elif text.endswith("JPY"):
            return  20.0

        return -1

class Analyst:
    def __init__(self):
        self.output = None
        self.analysis = None
        self.summary = {}
        self.notify = Notify()
    def update_symbol(self, symbol_with_exchange, parameters):
        self.output = TA_Handler(
            interval=parameters["interval"],
            exchange=parameters["exchange"],
            screener=parameters["screener"],
            symbol=symbol_with_exchange
        )
        self.analysis = self.output.get_analysis()
        self.summary = self.analysis.summary

    def get_full_summary(self):
        return self.summary

    def fetch_signals(self, currency_pairs_list, parameters):
        if not currency_pairs_list:
            self.notify.send_error_notification("No currency pairs provided for signal fetching.")

        trade_signals = []
        for symbol in currency_pairs_list:
            try:
                self.update_symbol(symbol, parameters)
                trade_signals.append({symbol: self.get_full_summary()})
            except Exception as e:
                self.notify.send_warning_notification(f"Error updating symbol {symbol}: {e}")

        return trade_signals

class Validator:
    def __init__(self):
        self.confirmed_signals = False
        self.turning_point = None
        self.data_frame = None

    def validate_signal(self, currency_pair: str, action: str) -> bool:
        action = action.lower()
        self.fetch_data(symbol=currency_pair, data_points=50, timeframe=mt5.TIMEFRAME_M5)
        self.update_turning_points()  # Update turning points after fetching data

        if self.turning_point and self.data_frame is not None:

            # Extract the most recent completed candle data
            most_recent_low, most_recent_high = self.get_most_recent_turning_point() #mrtp on bigger time frame

            self.fetch_data(symbol=currency_pair, data_points=5, timeframe=mt5.TIMEFRAME_M5)
            most_recent_candle = self.data_frame.iloc[-1]

            # Verify conditions for buy or sell signals
            if action == "buy" and most_recent_high:
                if most_recent_candle["close"] > most_recent_high['high']:
                    if most_recent_candle["open"] < most_recent_high['high']:
                        return True

            elif action == "sell" and most_recent_low:
                if most_recent_candle["close"] < most_recent_low['low']:
                    if most_recent_candle["open"] > most_recent_low['low']:
                        return True

        return False

    def get_most_recent_turning_point(self):
        """
        Returns the most recent high and low turning points from the list of turning points.

        Parameters:
            turning_points (list): A list of dictionaries representing turning points with their indices.

        Returns:
            tuple: A tuple containing the most recent low turning point and the most recent high turning point,
                   or (None, None) if there are no turning points.
        """
        recent_low = None
        recent_high = None

        # Iterate through the turning points in reverse order to find the most recent low and high turning points
        for tp in reversed(self.turning_point):
            if tp['type'] == 'low' and recent_low is None:
                recent_low = tp  # Found the most recent low turning point
            if tp['type'] == 'high' and recent_high is None:
                recent_high = tp  # Found the most recent high turning point
            # If both are found, no need to continue checking
            if recent_low and recent_high:
                break

        return recent_low, recent_high  # Return the most recent low and high turning points, or (None, None) if not found

    def update_turning_points(self):

        turning_points = []

        if self.data_frame is not None:
            for i in range(1, len(self.data_frame) - 1):
                # Check for a downtrend to uptrend (low turning point)
                if self.data_frame.iloc[i - 1]['close'] > self.data_frame.iloc[i]['close'] < self.data_frame.iloc[i + 1]['close']:
                    # For a low turning point, find the candle with the lowest low in the surrounding region
                    low_region = self.data_frame.iloc[i - 1:i + 2]  # Include the previous, current, and next candle
                    low_candle = low_region.loc[low_region['low'].idxmin()]  # Get the candle with the lowest low

                    turning_points.append({
                        **low_candle.to_dict(),
                        'index': i,
                        'type': 'low',
                        'peak': low_candle['low']  # Store the lowest low of the region
                    })

                # Check for an uptrend to downtrend (high turning point)
                if self.data_frame.iloc[i - 1]['close'] < self.data_frame.iloc[i]['close'] > self.data_frame.iloc[i + 1]['close']:
                    # For a high turning point, find the candle with the highest high in the surrounding region
                    high_region = self.data_frame.iloc[i - 1:i + 2]  # Include the previous, current, and next candle
                    high_candle = high_region.loc[high_region['high'].idxmax()]  # Get the candle with the highest high

                    turning_points.append({
                        **high_candle.to_dict(),
                        'index': i,
                        'type': 'high',
                        'trough': high_candle['high']  # Store the highest high of the region
                    })

        self.turning_point = turning_points


    def fetch_data(self, symbol: str, data_points: int = 1, timeframe: int = mt5.TIMEFRAME_M5, retries: int = 5):
        for attempt in range(retries):
            try:
                ohlc_data = mt5.copy_rates_from_pos(symbol, timeframe, 1, data_points)
                if ohlc_data is not None and len(ohlc_data) > 0:
                    data_frame = pd.DataFrame(ohlc_data)
                    data_frame['time'] = pd.to_datetime(data_frame['time'], unit='s')
                    data_frame['time'] = data_frame['time'].apply(m_dates.date2num)
                    self.data_frame = data_frame
                    return

            except (ValueError, TypeError):
                pass  # Handle known exceptions quietly

        self.data_frame = None  # Set to None if retries are exhausted

class Processor:
    def __init__(self):
        # Store the previous state and changes
        self.__previous = []
        self.__changes = []
        self.signal_validator =  Validator()

    def update_previous_and_changes(self, current_signal):
        # First-time use: if there's no previous list, set it and no changes
        if not self.__previous:
            self.__previous = current_signal
            self.__changes = []  # No changes for the first update
            return

        # Identify changes by comparing current_signal to the previous state
        for each_signal in current_signal:
            key = list(each_signal.keys())[0]  # Extract the symbol (key)
            current_value = each_signal[key]  # Current value associated with the key

            # Check if the key exists in the previous signal
            found_in_previous = next((item for item in self.__previous if key in item), None)

            if found_in_previous :
                previous_value = found_in_previous[key]  # Previous value associated with the key

                # If the value has changed, record the change
                if (current_value is not None) and \
                   (current_value != previous_value) and \
                   (self.signal_validator.validate_signal(key,current_value) ) :
                    self.__changes.append(each_signal)
                    location = self.__previous.index({key: previous_value})
                    self.__previous[location] = {key: current_value}
                elif (current_value is None) and (previous_value is not None):
                    location = self.__previous.index({key: previous_value})
                    self.__previous[location] = {key: previous_value}


            else:
                # If the symbol doesn't exist in the previous list, it's a new signal
                self.__changes.append(each_signal)


    def reset_changes(self):
        """Clears the list of recorded changes."""
        self.__changes.clear()

    def get_previous(self):  # Getter for the previous state
        return self.__previous

    def get_changes(self):  # Getter for the changes made
        return self.__changes

class Calculator:
    _instance = None  # Class-level variable to store the singleton instance

    def __new__(cls):
        if cls._instance is None:
            # Create the singleton instance if it doesn't exist
            cls._instance = super(Calculator, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def calculate_value_per_pip(current_price, currency_pair):
        # If the currency pair is a JPY pair, pip value is calculated differently

        if 'JPY' in currency_pair:
            p_v = (0.01  * 100_000)/ current_price
            value_per_pip = p_v

        elif currency_pair.endswith('USD'):
            p_v = (0.0001 * 100_000)/ current_price
            value_per_pip = p_v * current_price
        else:
            p_v = (0.0001 * 100_000)/ current_price
            value_per_pip = p_v

        return round(value_per_pip,3)


    @staticmethod
    def calculate_lot_size(account_size, risk_ratio, pips, symbol,current_price):

            amount_per_pip = Calculator.calculate_value_per_pip(current_price, symbol)
            amount_to_risk = account_size * (risk_ratio/100)

            lot_size_amount = amount_to_risk / (pips * amount_per_pip)
            return round(lot_size_amount, 2)

    @staticmethod
    def calculate_stop_loss(entry_price, stop_loss_pips, position_type, symbol):

        #NOTE CHECK THIS PART VERY VERY WELL
        if symbol.upper() == "XAUUSD":  # For gold (XAU/USD)
            pip_value = 0.01  # Gold pip value is 0.01 per ounce
            precision = 2  # Round to 2 decimal places for gold
        elif symbol.upper() == "XAGUSD":  # For silver (XAG/USD)
            pip_value = 0.01  # Silver pip value is 0.01 per ounce
            precision = 2  # Round to 2 decimal places for silver
        elif symbol.upper().endswith("JPY"):  # For JPY pairs (e.g., USDJPY)
            pip_value = 0.01  # 1 pip is 0.01 for JPY pairs
            precision = 3  # Round to 2 decimal places for JPY pairs
        else:  # For other currency pairs
            pip_value = 0.0001  # Standard pip value for currency pairs (0.0001)
            precision = 5  # Round to 4 decimal places for other currency pairs

        # Calculate stop loss in price based on the number of pips
        stop_loss_in_price = stop_loss_pips * pip_value

        # Adjust stop-loss price based on the position type
        if position_type.lower() == "buy":
            # For buy positions, stop-loss is below the entry price
            stop_loss_price = entry_price - stop_loss_in_price
        elif position_type.lower() == "sell":
            # For sell positions, stop-loss is above the entry price
            stop_loss_price = entry_price + stop_loss_in_price
        else:
            print("Position type must be 'buy' or 'sell'.")
            stop_loss_price = 0
        # Return the calculated stop-loss price, rounded according to the currency's precision
        return round(stop_loss_price, precision)

    @staticmethod
    def calculate_target_stop_loss(symbol, current_stop_loss, stop_loss_pips, trade_type):
        # Use the inverse of the trade type for the calculation
        inverse_trade_type = "sell" if trade_type.lower() == "buy" else "buy"

        # Call TradeCalculator.calculate_stop_loss with the inverse trade type
        # return Calculator.calculate_stop_loss(current_stop_loss, stop_loss_pips, inverse_trade_type, symbol)
        return Calculator.calculate_stop_loss(entry_price=current_stop_loss,
                                              stop_loss_pips=stop_loss_pips,
                                              position_type=inverse_trade_type,
                                              symbol=symbol)

class Execute:
    """
    Handles trade executions, including opening, closing, and updating orders.

    Responsibilities:
    - Open trades with specified parameters (symbol, action, risk ratio, pips).
    - Close existing trades for a given symbol.
    - Update stop-loss (SL) or take-profit (TP) for active positions.
    - Retry failed trade requests with a notification system.
    """


    # Singleton instance and lock for thread-safety
    _instance = None
    _lock = threading.Lock()

    # Ensure only one instance is created (thread-safe singleton)
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    # Initialize attributes
    def __init__(self):
        self.result = None
        self.request = dict()
        self.notify = Notify()
        self.position = None

    # Retry on exception, stop after 5 attempts, wait 0.5 seconds between retries
    @retry(retry=retry_if_exception_type(Exception), stop=stop_after_attempt(5), wait=wait_fixed(0.5))
    def process_trade_request(self,pips=None, symbol=None, action=None,position=None, target_sl=None, risk_ratio=None):
        """
        Process and send trade requests based on provided parameters.

        Args:
            pips (float, optional): The number of pips for the trade. Default is None.
            symbol (str, optional): The symbol for the trade (e.g., 'EURUSD'). Default is None.
            action (str, optional): The type of action ('buy' or 'sell'). Default is None.
            position (str, optional): The position identifier for updating SL/TP. Default is None.
            target_sl (float, optional): The target stop-loss value for the position. Default is None.
            risk_ratio (float, optional): The risk-to-reward ratio for the trade. Default is None.

        Raises:
            ValueError: If required parameters are not provided.

        Returns:
            dict: A dictionary containing the result of the trade request or error message.
        """


        # Prepare the trade request based on provided parameters
        if symbol and action and risk_ratio and pips:
            self.update_open_order_request(symbol=symbol, action=action, risk_ratio=risk_ratio, pips=pips)

        elif symbol:
            self.update_close_order_request(symbol=symbol)

        elif position and target_sl is not None:
            self.update_sl_order_request(position=position, target_sl= target_sl)

        else:
            raise ValueError("Insufficient parameters provided for trade request.")


        # If the request wasn't prepared accurately
        if self.request is None:
            raise RuntimeError("Trade request is invalid or not properly formed.")

        # Send the trade request
        self.result = mt5.order_send(self.request)

        # Return success message if successful
        if self.result and self.result.retcode == mt5.TRADE_RETCODE_DONE:
            self.notify.send_console_message(f"Symbol: {symbol} [ORDER SUCCESS]")

        else:
            error_message = f"Trade failed for symbol: {symbol}. Retcode: {'No result' if not self.result else self.result.retcode}"
            raise RuntimeError(error_message)

    def update_open_order_request(self, symbol, action, risk_ratio, pips):
        """
        Prepares a trade request for placing an open order via MetaTrader5.

        Parameters:
            symbol (str): Trading symbol (e.g., "EURUSD").
            action (str): Order type, 'buy' or 'sell'.
            risk_ratio (float): Risk-to-reward ratio for the trade.
            pips (float): Number of pips for stop-loss or target.

        Sets:
            self.request: The constructed request dictionary for execution.

        Raises:
            ValueError: If any parameter is invalid or missing.
        """

        # Ensure action is uppercase and determine the correct order type
        action = action.upper()

        # Get relevant data
        balance = mt5.account_info().balance
        symbol_tick = mt5.symbol_info_tick(symbol)

        # Determine order details, type, and validate quote currency
        order_price = symbol_tick.ask if action == "BUY" else symbol_tick.bid
        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        quote_currency = symbol[-3:]

        # Calculate lot size and handle currency assumptions for non-USD base currencies
        if  quote_currency in ["AUD","GBP","NZD","USD"]:
            lot_calculator_price =  order_price
        else:
            quote_currency = "USD" + quote_currency
            quote_currency_tick = mt5.symbol_info_tick(quote_currency)
            quote_currency_price = quote_currency_tick.ask if action == "BUY" else quote_currency_tick.bid
            lot_calculator_price = quote_currency_price


        # Calculate the lot-size using the price of the USD-based quote currency
        lot = float(Calculator.calculate_lot_size( current_price=lot_calculator_price,
                                                   account_size=balance,
                                                   risk_ratio=risk_ratio,
                                                   symbol=symbol,
                                                   pips=pips))

        # Calculate the stop loss price
        order_sl = float(Calculator.calculate_stop_loss( entry_price= order_price,
                                                         position_type= action,
                                                         stop_loss_pips= pips,
                                                         symbol= symbol ))

        # Prepare trade request
        self.request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol,
                            "price": order_price,
                            "sl": order_sl,
                            "type": order_type,
                            "volume": lot,
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                            "comment": "python_script_open"
                        }

    def update_close_order_request(self, symbol):
        """
        Prepares a request to close a single open position for the given trading symbol.

        Parameters:
            symbol (str): The trading symbol (e.g., "EURUSD").

        Sets:
            self.request: The constructed close order request for the specified symbol.
        """

        # Retrieve and validate current positions for the symbol, ensuring safe access and handling invalid ticks
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            position = positions[0]
        else:
            self.request = None
            return

        # Retrieve symbol info & tick data
        symbol_info = mt5.symbol_info(position.symbol)
        tick = mt5.symbol_info_tick(position.symbol)

        # Validate volume
        valid_volume = max(symbol_info.volume_min,round(position.volume / symbol_info.volume_step) * symbol_info.volume_step)

        # Determine opposite order type and price
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        order_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

        # Prepare trade request
        self.request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": position.ticket,
                    "symbol": position.symbol,
                    "price": order_price,
                    "type": order_type,
                    "volume": valid_volume,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                    "comment": "python_script_close",
                }

    def update_sl_order_request(self,position,target_sl):

        """
        Prepares a request to update the stop-loss for a given position.

        Parameters:
            position (dict): The position to update, containing details like symbol and entry price.
            target_sl (float): The new stop-loss value to be set.

        Sets:
            self.request: The constructed stop-loss update request.
        """

        # Get the ticket (order) for the position to update
        ticket = position["ticket"]

        # Prepare trade request
        self.request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "sl": target_sl,
            "position": ticket,
        }

class PositionModifier:
    def __init__(self):
        self.risk_amount = 0  # Set the risk amount (e.g., $20)

    def get_target_sl(self, position):

        current_profit = position.get("profit")  # Extract current profit from position dictionary
        symbol = position.get("symbol")  # Extract symbol from position
        order_type = "buy" if position.get("type") == 0 else "sell"  # Determine order type based on position type
        entry_price = position.get("entry_price")  # Extract entry price from position
        stop_loss =  position.get("stop_loss")

        risk_amount = self.risk_amount  # Set trade risk amount

        level = math.floor(current_profit / risk_amount) - 1  # Compute level based on profit and risk

        stop_loss_pips = level * Interpreter.process_signal_pip_count(symbol)  # Stop-loss pips based on level

        # Use the calculator to calculate the new stop-loss
        target_sl = Calculator.calculate_target_stop_loss(
            current_stop_loss=entry_price,
            stop_loss_pips=stop_loss_pips,
            trade_type=order_type,
            symbol=symbol
        )

        # Ensure the target stop loss is permanent
        if order_type == "buy" and target_sl < stop_loss:
            # For a buy, the target stop-loss cannot be lower than the previous stop-loss
            return stop_loss
        elif order_type == "sell" and target_sl > stop_loss:
            # For a sell, the target stop-loss cannot be higher than the previous stop-loss
            return stop_loss


        return target_sl if target_sl else None

    def set_risk_amount(self, risk_amount):
        """ Sets the risk amount for each trade (e.g., $20). """
        self.risk_amount = risk_amount

class PositionsTracker:

    @staticmethod
    def find_highest_profit_id(positions):
        """
        Finds the ID of the position with the highest profit.
        :param positions: A list of positions
        :return: The ID of the position with the highest profit, or None if no positions exist.
        """
        if not positions:
            return None  # Return None if no positions are available

        # Find the position with the highest profit
        highest_profit_position = max(positions, key=lambda pos: pos['profit'])

        return highest_profit_position

# if __name__ == "__main__":
#     import MetaTrader5 as mt5
#     from datetime import datetime, timedelta
#
#
#
#     # Assuming the connection to MetaTrader5 has already been made
#     mt5.initialize()
#
#     # Create an instance of the MT5Account class
#     account = MT5Account()
#     # Update the account details
#     _account_number_ = 88771620  # Account number
#     _server_ = "MetaQuotes-Demo"  # MT5 server name (optional, depending on broker)
#     _password_ = "My-b5wOa"  # MT5 account password
#
#     # Update account information
#     account.update_basic_info(server=_server_,
#                                    account_id=_account_number_,
#                                    password=_password_)
#
#     # Test get_account_id
#     print(f"Account ID: {account.get_account_id()}")
#
#     # Test get_report (it will internally update positions)
#     print(f"Account Report: {account.get_report()}")
#
#     # Test get_balance
#     print(f"Current Balance: {account.get_balance()}")
#
#     # Test get_positions
#     print(f"Positions: {account.get_positions()}")
#
#     # Test get_open_position_status (this will calculate the total profit/loss of open positions)
#     print(f"Open Position Status: {account.get_open_position_status()}")
#
#     # Test get_closed_position_status with a specific date range (optional)
#     from_date = datetime(2023, 1, 1)  # Start date for closed positions
#     to_date = datetime.now()  # End date for closed positions
#     print(f"Closed Position Status (from {from_date} to {to_date}): {account.get_closed_position_status(from_date, to_date)}")
#
#     # Test display_account_info (it will print account details)
#     account.display_account_info()
#
#
#     # Shutdown the MT5 connection after testing
#     mt5.shutdown()
