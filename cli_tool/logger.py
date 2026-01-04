import logging
import os
import sys

def setup_logger():
    """Configures and returns the logger for the CLI tool."""
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "cli_chat.log")

    logger = logging.getLogger("cli_tool")
    logger.setLevel(logging.INFO)
    
    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        return logger

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console Handler (Optional, maybe we only want errors or concise info on console to not clutter chat)
    # The requirement said "chat interface", so we probably shouldn't log EVERYTHING to console, 
    # but maybe just errors or app startup info.
    # For now, I'll log INFO to console too but maybe formatted simply.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[LOG] %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    # logger.addHandler(console_handler) # Commented out console logging to avoid cluttering the chat UI

    return logger

logger = setup_logger()
