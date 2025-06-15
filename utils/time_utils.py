#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : time_utils.py
# @Time    : 2025/6/15 15:38
# @Desc    :

import pytz
from datetime import datetime, timedelta, time
from typing import List, Dict, Any, Optional, Tuple
import calendar


class TimeUtils:
    """Utility functions for time processing and analysis"""

    @staticmethod
    def parse_time_string(time_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
        """Parse time string to datetime object"""
        try:
            return datetime.strptime(time_str, format_str)
        except ValueError as e:
            # Try alternative formats
            alternative_formats = [
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M",
                "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y"
            ]

            for alt_format in alternative_formats:
                try:
                    return datetime.strptime(time_str, alt_format)
                except ValueError:
                    continue

            raise ValueError(f"Unable to parse time string: {time_str}")

    @staticmethod
    def format_duration(minutes: int) -> str:
        """Format duration in minutes to human-readable string"""
        if minutes < 60:
            return f"{minutes}m"
        elif minutes < 1440:  # Less than 24 hours
            hours = minutes // 60
            mins = minutes % 60
            if mins == 0:
                return f"{hours}h"
            else:
                return f"{hours}h {mins}m"
        else:  # Days
            days = minutes // 1440
            remaining_hours = (minutes % 1440) // 60
            if remaining_hours == 0:
                return f"{days}d"
            else:
                return f"{days}d {remaining_hours}h"

    @staticmethod
    def get_time_period(hour: int) -> str:
        """Classify hour into time period"""
        if 0 <= hour < 6:
            return "night"
        elif 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    @staticmethod
    def get_day_type(date: datetime) -> str:
        """Get day type (weekday/weekend)"""
        return "weekend" if date.weekday() >= 5 else "weekday"

    @staticmethod
    def get_season(date: datetime) -> str:
        """Get season for given date"""
        month = date.month
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "autumn"
        else:
            return "winter"

    @staticmethod
    def calculate_time_overlap(start1: datetime, end1: datetime,
                               start2: datetime, end2: datetime) -> int:
        """
        Calculate overlap between two time periods in minutes
        Returns 0 if no overlap
        """
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        if overlap_start < overlap_end:
            return int((overlap_end - overlap_start).total_seconds() / 60)
        else:
            return 0

    @staticmethod
    def find_time_gaps(time_periods: List[Tuple[datetime, datetime]],
                       start_bound: datetime, end_bound: datetime) -> List[Tuple[datetime, datetime]]:
        """
        Find gaps between time periods within given bounds
        """
        if not time_periods:
            return [(start_bound, end_bound)]

        # Sort periods by start time
        sorted_periods = sorted(time_periods, key=lambda x: x[0])

        gaps = []
        current_time = start_bound

        for period_start, period_end in sorted_periods:
            # Check for gap before this period
            if current_time < period_start:
                gaps.append((current_time, period_start))

            # Update current time to end of this period
            current_time = max(current_time, period_end)

        # Check for gap after last period
        if current_time < end_bound:
            gaps.append((current_time, end_bound))

        return gaps

    @staticmethod
    def generate_time_slots(start_time: datetime, end_time: datetime,
                            slot_duration: int, gap_duration: int = 0) -> List[Tuple[datetime, datetime]]:
        """
        Generate time slots between start and end time
        slot_duration and gap_duration in minutes
        """
        slots = []
        current_time = start_time

        while current_time + timedelta(minutes=slot_duration) <= end_time:
            slot_end = current_time + timedelta(minutes=slot_duration)
            slots.append((current_time, slot_end))
            current_time = slot_end + timedelta(minutes=gap_duration)

        return slots

    @staticmethod
    def calculate_business_hours_overlap(activity_start: datetime, activity_end: datetime,
                                         business_start: time = time(9, 0),
                                         business_end: time = time(17, 0)) -> int:
        """
        Calculate overlap with business hours in minutes
        """
        # Convert business hours to datetime for comparison
        business_start_dt = datetime.combine(activity_start.date(), business_start)
        business_end_dt = datetime.combine(activity_start.date(), business_end)

        return TimeUtils.calculate_time_overlap(
            activity_start, activity_end,
            business_start_dt, business_end_dt
        )

    @staticmethod
    def is_peak_hour(hour: int, day_type: str = "weekday") -> bool:
        """Check if hour is considered peak time"""
        if day_type == "weekday":
            return hour in [8, 9, 12, 13, 17, 18, 19]
        else:  # weekend
            return hour in [11, 12, 13, 19, 20, 21]

    @staticmethod
    def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
        """Convert datetime between timezones"""
        from_timezone = pytz.timezone(from_tz)
        to_timezone = pytz.timezone(to_tz)

        # Localize to source timezone if naive
        if dt.tzinfo is None:
            dt = from_timezone.localize(dt)

        # Convert to target timezone
        return dt.astimezone(to_timezone)