#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : logging_utils.py
# @Time    : 2025/6/15 15:38
# @Desc    :


import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional


class LoggingUtils:
    """Utility functions for logging configuration"""

    @staticmethod
    def setup_logging(log_level: str = "INFO",
                      log_file: Optional[str] = None,
                      log_dir: str = "logs") -> logging.Logger:
        """Setup logging configuration"""

        # Create log directory if it doesn't exist
        if log_file and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            if not log_file.startswith('/'):
                log_file = os.path.join(log_dir, log_file)

        # Configure logging level
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {log_level}')

        # Create logger
        logger = logging.getLogger('spatiotemporal_generator')
        logger.setLevel(numeric_level)

        # Clear existing handlers
        logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if log_file specified
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    @staticmethod
    def log_generation_progress(logger: logging.Logger,
                                completed: int,
                                total: int,
                                start_time: datetime) -> None:
        """Log generation progress"""

        if total == 0:
            return

        progress_percent = (completed / total) * 100
        elapsed_time = (datetime.now() - start_time).total_seconds()

        if completed > 0:
            estimated_total_time = elapsed_time * total / completed
            remaining_time = estimated_total_time - elapsed_time

            logger.info(
                f"Progress: {completed}/{total} ({progress_percent:.1f}%) - "
                f"Elapsed: {elapsed_time:.1f}s - "
                f"Estimated remaining: {remaining_time:.1f}s"
            )
        else:
            logger.info(f"Starting generation of {total} trajectories...")

    @staticmethod
    def log_performance_stats(logger: logging.Logger, stats: Dict[str, Any]) -> None:
        """Log performance statistics"""

        logger.info("=== Performance Statistics ===")
        logger.info(f"Total generated: {stats.get('total_generated', 0)}")
        logger.info(f"Total time: {stats.get('total_time', 0):.2f}s")
        logger.info(f"Success rate: {stats.get('success_rate', 0):.2%}")
        logger.info(f"Average generation time: {stats.get('average_generation_time', 0):.2f}s")
        logger.info(f"Throughput: {stats.get('throughput', 0):.2f} trajectories/min")

        if 'peak_memory_usage' in stats:
            peak_memory_mb = stats['peak_memory_usage'] / 1024 / 1024
            logger.info(f"Peak memory usage: {peak_memory_mb:.1f} MB")