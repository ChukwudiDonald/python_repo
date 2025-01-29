from datetime import datetime
from tenacity import RetryError
from utils_factory import util_factory
from abc import ABC, abstractmethod
import MetaTrader5  as mt5
import time

class Manager(ABC):
    notify = util_factory.create_notify()  # Class-level variable (shared across all instances)

    @abstractmethod
    def accomplish_task(self, *args, **kwargs):
        pass

class TaskManager(Manager,ABC):
    def __init__(self):
        self.execute = util_factory.create_execute()
        self.interpreter = util_factory.create_interpreter()

    @abstractmethod
    def accomplish_task(self, *args, **kwargs):
        pass

class ConnectionManager(Manager):
    def __init__(self, mt5_account= None):
        self.connected = False
        self.logged_in = False
        self.account_server = None
        self.account_ID = None
        self.account_password = None


        # Use the account's basic info if provided; otherwise, default to None
        self.set_account_info(mt5_account)

    def check_initialized_and_logged_in(self):
        return self.logged_in and self.connected

    def accomplish_task(self):
        """Accomplishes both the connection and login tasks."""

        # If connected, proceed to log in
        if self.initialize_connection():
            self.login_mt5()


        return self.check_initialized_and_logged_in()

    def change_account(self, mt5_account):
        """
        Change the account information in the ConnectionManager.
        """
        account_info = mt5_account.get_basic_account_info()  # Get account details
        self.account_server = account_info["account_server"]
        self.account_ID = account_info["account_ID"]
        self.account_password = account_info["account_password"]
        self.notify.send_info_notification(f"Account information updated for ID: {self.account_ID}")

    def initialize_connection(self):
        """Establish the connection to the trading platform."""

        if mt5.initialize():
            self.connected = True
            # Send notifications using NotificationManager
            self.notify.send_info_notification("Successfully connected to MetaTrader 5 server.")
        else:
            self.notify.send_error_notification("Failed to connect to the MetaTrader 5 server. Please check your connection and try again.")

        return self.connected

    def initialize_symbols(self, currency_list):
        failed_initialization = []
        successful_initialization = []

        for symbol in currency_list:
            # Get information about the symbol from MetaTrader 5
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                failed_initialization.append(symbol)
                continue  # Skip to the next symbol in the list

            if not symbol_info.visible:  # Check if the symbol is visible in the Market Watch
                if not mt5.symbol_select(symbol, True):
                    failed_initialization.append(symbol)
                    continue  # Skip to the next symbol in the list

            successful_initialization.append(symbol)

        # Send notifications using NotificationManager
        if successful_initialization:
            self.notify.send_info_notification(f"Successfully initialized: {successful_initialization}")
        if failed_initialization:
            self.notify.send_warning_notification(f"Failed to initialize: {failed_initialization}")

        return successful_initialization

    def login_mt5(self):
        if mt5.login(self.account_ID,self.account_password,self.account_server):
            self.logged_in = True
            self.notify.send_info_notification("Logged in successfully")
        else:
            self.notify.send_error_notification("Failed to login")
        return self.logged_in

    def shutdown_connection(self):
        """Close the connection."""
        time.sleep(1)
        if self.connected:
            mt5.shutdown()
            self.notify.send_info_notification("MetaTrader 5 connection closed successfully.\n")

        else:
            self.notify.send_warning_notification("No active connection to close. Please establish a connection first.")

    def set_account_info(self, account):
        if account is not None:
            account_info = account.get_basic_account_info()
            self.account_server = account_info.get("account_server", None)
            self.account_ID = account_info.get("account_ID", None)
            self.account_password = account_info.get("account_password", None)


class OrderManager(TaskManager):
    def __init__(self):
        super().__init__()
        self.parameters = {}
        self.signal_to_process = []
        self.currency_pair_to_fetch = []
        self.analyst = util_factory.create_analyst()
        self.processor = util_factory.create_processor()

    def set_parameters(self, exchange, screener, interval):
        """Set the exchange, screener, and interval parameters."""
        self.parameters = {
            "exchange": exchange,
            "screener": screener,
            "interval": interval
        }

    def accomplish_task(self, currency_pairs, risk_per_trade, exchange, screener, interval):
        self.set_parameters(exchange, screener, interval)
        self.currency_pair_to_fetch = currency_pairs
        self.fetch_orders()

        # Step 2: Process recommendations
        self.processor.update_previous_and_changes(self.signal_to_process)

        self.notify.send_console_message(f"Previous List: {self.processor.get_previous()}")
        self.notify.send_console_message(f"Changes: {self.processor.get_changes()}")

        # Step 3: Place orders based on changes
        for new_signal in self.processor.get_changes():
            for curr_pair, recommendation in new_signal.items():
                pip_per_currency = self.interpreter.process_signal_pip_count(curr_pair)

                try:
                    self.execute.process_trade_request(symbol=curr_pair)
                except RetryError:
                    self.notify.send_console_message(f"Close - {curr_pair} [ORDER FAILURE]")  # Ignore the error

                try:
                    self.execute.process_trade_request(symbol=curr_pair, action=recommendation,
                                                       risk_ratio=risk_per_trade,pips=pip_per_currency)
                except RetryError:
                    self.notify.send_console_message(f"Open - {curr_pair} [ORDER FAILURE]")  # Ignore the error

        # Step 5: Reset changes after processing
        self.processor.reset_changes()

    def fetch_orders(self):
        if not self.currency_pair_to_fetch:
            self.notify("Currency pair list is empty. Cannot fetch orders.")

        signals = self.analyst.fetch_signals(currency_pairs_list=self.currency_pair_to_fetch,
                                             parameters=self.parameters)
        self.signal_to_process = self.interpreter.process_signal_list(signals)


class AccountManager(TaskManager):
    def __init__(self, mt5_account=None,start_time=None):
        # Initialize the instance variables with default values
        super().__init__()
        self.progress = 0
        self.balance = None
        self.positions = None
        self.account_ID = None
        self.total_profit = None
        # self.total_loss_per_period = None
        self.positions_tracker = util_factory.create_position_tracker()  # Tracker for position-related tasks
        self.positions_modifier = util_factory.create_position_modifier()  # Modifier to adjust position properties
        self.start_time= start_time if start_time is not None else datetime.now()

        # If an account is provided, use it to initialize the account information
        self.set_account_info(mt5_account)

    def accomplish_task(self, risk, account, max_profit_, min_loss_):
        """This method accomplishes tasks related to the account."""

        self.set_account_info(account)  # Set the account information
        risk_per_trade = (risk/100) * self.balance  # Risk amount per trade based on balance

        # Calculate profit and loss percentages
        profit_percentage = round((self.total_profit/self.balance)* 100)
        # loss_percentage = round(((self.total_loss_per_period / self.balance) * 100),2)

        # print(f"{loss_percentage}%")
        # Close all positions if profit percentage exceeds maximum threshold
        if profit_percentage >= max_profit_:
            for position in self.positions:
                symbol_name = position['symbol']  # Symbol name for the position
                self.execute.close_order(symbol_name=symbol_name)  # Close the position
                return  # Exit after closing positions

        # # # Close position if loss percentage falls below the minimum threshold
        # elif loss_percentage < min_loss_:
        #     position = self.positions_tracker.find_highest_profit_id(self.positions)  # Find highest profit position
        #     self.execute.close_order(symbol_name=position['symbol'])  # Close the identified position

        self.positions_modifier.set_risk_amount(risk_per_trade)  # Set the risk amount for future trades

        # Update stop-loss for positions with profit exceeding risk per trade
        for position in self.positions:
            if position["profit"] > risk_per_trade:  # Check if profit exceeds risk per trade
                target_sl = self.positions_modifier.get_target_sl(position)  # Get target stop-loss

                if (target_sl is not None) and (target_sl != position["stop_loss"]):  # Ensure target stop-loss is valid
                    try:
                        self.execute.process_trade_request(position=position, target_sl= target_sl)
                    except RetryError:
                        self.notify.send_console_message(f"Update - {position["symbol"]} [ORDER FAILURE]")  # Ignore the error

    def set_account_info(self, account):
        if account is not None:
            # Set balance, positions, account ID, and total profit from the provided account
            self.balance = account.get_balance()
            self.positions = account.get_positions()
            self.account_ID = account.get_account_id()
            self.total_profit = account.get_open_position_status()
            # self.total_loss_per_period = account.get_closed_position_status(from_date=self.start_time)


# if __name__ == "__main__":
#     mt5.initialize()
    # account_test = util_factory.create_account()
    # _account_number_ = 88771620  # Account number
    # _server_ = "MetaQuotes-Demo"  # MT5 server name (optional, depending on broker)
    # _password_ = "My-b5wOa"  # MT5 account password
    #
    # # Update account information
    # account_test.update_basic_info(server=_server_,
    #                                account_id=_account_number_,
    #                                password=_password_)
    #
    # account_manager = AccountManager()
    # r_p_t =  0.5
    # account_manager.accomplish_task(risk= r_p_t ,
    #                                  account=account_test,
    #                                  max_profit_=10,
    #                                  min_loss_=-4)

    # start_time_now = datetime.now()
    #
    # while True:
    #     deals = mt5.history_deals_get(start_time_now, datetime.now())
    #     for deal in deals:
    #         print(deal)
    #     time.sleep(15)
    #     print("update")


