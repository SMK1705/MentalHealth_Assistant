import os
import logging
import logging.config

# Define the logs directory (using a relative path)
log_dir = "logs"

# Create the logs directory if it does not exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'DEBUG',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'level': 'INFO',
            'filename': os.path.join(log_dir, 'app.log'),
            'mode': 'a',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'DEBUG',
    },
}

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
