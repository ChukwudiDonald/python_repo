import utils as ut

class UtilsFactory:
    def __init__(self):
        """Initialize the factory with empty dictionaries to store managers and utilities."""
        self.notification_manager = ut.Notify()  # Shared NotificationManager

    def create_account(self):
        """Create and return the shared NotificationManager."""
        try:
            # Create the NotificationManager
            return ut.MT5Account()
        except Exception as e:
            self.notification_manager.send_error_notification(f"Error Creating Account: {e}")
            return None

    def create_notify(self):
        """Create and return the shared NotificationManager."""
        try:
            # Create the NotificationManager
            return ut.Notify()
        except Exception as e:
            self.notification_manager.send_error_notification(f"Error creating NotificationManager: {e}")
            return None

    def create_analyst(self):
        """Create and return the shared Analyst."""
        try:
            # Create the Analyst
            return ut.Analyst()
        except Exception as e:
            self.notification_manager.send_error_notification(f"Error creating Analyst: {e}")
            return None

    def create_interpreter(self):
        """Create and return the shared Interpreter."""
        try:
            # Create the Interpreter
            return ut.Interpreter()
        except Exception as e:
            self.notification_manager.send_error_notification(f"Error creating Interpreter: {e}")
            return None

    def create_processor(self):
        """Create and return the shared Processor."""
        try:
            # Create the Processor
            return ut.Processor()
        except Exception as e:
            self.notification_manager.send_error_notification(f"Error creating Processor: {e}")
            return None

    def create_execute(self):
        """Create and return the shared Execute."""
        try:
            # Create the Execute
            return ut.Execute()
        except Exception as e:
            self.notification_manager.send_error_notification(f"Error creating Execute: {e}")
            return None

    def create_position_tracker(self):
        """Create and return the shared Tracker."""
        try:
            # Create the Tracker
            return ut.PositionsTracker()
        except Exception as e:
            self.notification_manager.send_error_notification(f"Error creating Tracker: {e}")
            return None

    def create_position_modifier(self):
        """Create and return the shared Tracker."""
        try:
            # Create the Modifier
            return ut.PositionModifier()
        except Exception as e:
            self.notification_manager.send_error_notification(f"Error creating Modifier: {e}")
            return None


util_factory  = UtilsFactory()