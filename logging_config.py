import os
import logging
import logging.config
import logging.handlers  # noqa: F401  (referenced by dictConfig class path)

# Define the logs directory (using a relative path)
log_dir = "logs"

# Create the logs directory if it does not exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Default to INFO so verbose DEBUG lines (which include patient text and
# retrieved cases) are not written to disk/console unless explicitly enabled.
# Override with LOG_LEVEL=DEBUG for local troubleshooting only.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

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
            'level': LOG_LEVEL,
            'stream': 'ext://sys.stdout',
        },
        'file': {
            # Rotating handler so the log cannot grow unbounded.
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'standard',
            'level': LOG_LEVEL,
            'filename': os.path.join(log_dir, 'app.log'),
            'maxBytes': 1_000_000,
            'backupCount': 3,
            'encoding': 'utf-8',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': LOG_LEVEL,
    },
}

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
