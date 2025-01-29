import utils as ut
import managers as mng


class ManagersFactory:
    def __init__(self):
        """Initialize the factory with empty dictionaries to store managers and utilities."""
        self.notify = ut.Notify()  # Shared NotificationManager

    def create_connection_manager(self, mt5_account= None):
        """Create and return a new ConnectionManager."""
        try:
            # Create the ConnectionManager with the account
            return mng.ConnectionManager(mt5_account)
        except Exception as e:
            self.notify.send_error_notification(f"Error creating ConnectionManager: {e}")
            return None

    def create_order_manager(self):
        """Create and return a new ConnectionManager."""
        try:
            # Create the OrderManager with the account
            return mng.OrderManager()
        except Exception as e:
            self.notify.send_error_notification(f"Error creating ConnectionManager: {e}")
            return None

    def create_account_manager(self):
        """Create and return a new ConnectionManager."""
        try:
            # Create the OrderManager with the account
            return mng.AccountManager()
        except Exception as e:
            self.notify.send_error_notification(f"Error creating AccountManager: {e}")
            return None

manager_factory = ManagersFactory()
