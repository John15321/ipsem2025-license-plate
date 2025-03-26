# Utilities Module

This module provides various helper tools and utilities for the IPSEM 2025 License Plate Recognition project.

## Logging Utilities

The logging utilities provide a flexible and configurable logging system for the entire project.

### Features

- Console and file logging support
- Customizable log formats and levels
- Support for multiple loggers
- Comprehensive error handling
- Type-safe interface

### Usage

```python
from ipsem2025_license_plate.utils.logging_utils import configure_logging, get_logger

# Configure logging globally
configure_logging(
    level="INFO",
    log_to_console=True,
    log_to_file=True,
    log_file="my_application.log",
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format="%Y-%m-%d %H:%M:%S"
)

# Get a logger for a specific module
logger = get_logger("my_module")

# Use the logger
logger.info("Application started")
logger.debug("Detailed information")
logger.warning("Warning message")
logger.error("Error occurred")
```

### API Reference

#### `configure_logging`

Configure the logging system for the package.

**Parameters:**
- `level`: The logging level (e.g., `logging.INFO`, `logging.DEBUG`, or their string equivalents)
- `log_to_console`: Whether to log to console (stdout)
- `log_to_file`: Whether to log to a file
- `log_file`: The log file path (if log_to_file is True). If None, a default path in the current directory will be used.
- `log_format`: The log message format
- `date_format`: The format for timestamps
- `loggers`: List of specific logger names to configure. If None, configures the root logger.

#### `get_logger`

Get a logger with the specified name.

**Parameters:**
- `name`: Name of the logger to get

**Returns:**
- A logger instance

### Default Values

- Default log format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- Default date format: `%Y-%m-%d %H:%M:%S`
- Default log level: `INFO`
- Default log file (when enabled): `ipsem2025_license_plate.log` in the current directory