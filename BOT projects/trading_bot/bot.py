from managers_factory import manager_factory
from utils_factory import util_factory
from Timer import  Timer
from constants import get_active_currency_pairs
class Bot:
    def __init__(self):
        """Initialize bot with account details, connection manager, and notification manager."""
        self.connection_manager = manager_factory.create_connection_manager()
        self.account_manager = manager_factory.create_account_manager()
        self.order_manager = manager_factory.create_order_manager()
        self.notify = util_factory.create_notify()
        self.currency_pairs_list = None  # List of currency pairs to trade
        self.timer = Timer()
    def start_bot(self, currency_pairs_list):
        """Starts the bot, sets account details, and initializes the connection."""

        # Log the bot startup process
        self.notify.send_info_notification("Bot is starting...")

        # Step 2: Attempt to connect and perform necessary tasks
        if self.connection_manager.initialize_connection():

            # Initialize the list of currency pairs
            self.currency_pairs_list = self.connection_manager.initialize_symbols(currency_pairs_list)

            # Shutdown the connection once tasks are completed
            self.connection_manager.shutdown_connection()

    def __do_task(self,account,exchange, interval, screener,risk_per_trade):
        """Runs a specific task if the bot is running."""

        # Step 1: Attempt to log in into account
        self.connection_manager.set_account_info(account)

        # Step 2: Attempt to connect and execute tasks
        if self.connection_manager.accomplish_task():
            # If connection is successfully established, proceed with task execution
            self.order_manager.accomplish_task(currency_pairs= self.currency_pairs_list,
                                               risk_per_trade= risk_per_trade,# Set risk per trade
                                               exchange= exchange, # Specify the exchange platform
                                               interval=interval, # Define time interval
                                               screener=screener) # Set market screener criteria

            self.account_manager.accomplish_task(risk= risk_per_trade,
                                                 account=account,
                                                 max_profit_=10,
                                                 min_loss_=-4)
            self.connection_manager.shutdown_connection()

    def end_bot(self):
        """Ends the bot and sets the running state to False."""
        pass

    def run(self, cp_list,account, exchange, interval, screener, risk_per_trade):
        """Start the bot and run the task on schedule."""
        while True:
            if self.timer.is_task_time():
                self.currency_pairs_list = get_active_currency_pairs()
                self.__do_task(account=account,
                               screener=screener,
                               risk_per_trade=risk_per_trade,
                               exchange=exchange,
                               interval=interval)
                self.timer.rest(5)



if __name__ == "__main__":

    from constants import CONFIG, currency_pairs

    # Account Information
    account_test = util_factory.create_account()
    _account_number_ = 10005351167  # Account number
    _server_ = "MetaQuotes-Demo"  # MT5 server name (optional, depending on broker)
    _password_ = "!4ViOzTo"  # MT5 account password

    account_test.update_basic_info(server=_server_,
                                   account_id=_account_number_,
                                   password=_password_)

    # Bot Initialization and Execution
    bot = Bot()
    bot.start_bot(currency_pairs_list=currency_pairs)

    # Running the Bot with parameters
    bot.run(account=account_test,
            cp_list= CONFIG["cp_by_time_zone"],
            exchange=CONFIG["exchange"],
            screener=CONFIG["screener"],
            interval=CONFIG["interval"][1],
            risk_per_trade=0.5)

    # Assuming util_factory is predefined and has a create_account method

