#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : temporal_tools.py
# @Time    : 2025/6/15 15:33
# @Desc    : Temporal Management Tools for activity scheduling and time-related operations


import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional, Tuple
from .base_tool import BaseMCPTool, MCPMessage
import json


class TemporalManagementTool(BaseMCPTool):
    """
    Temporal management tool that handles time-related queries and operations
    without hardcoded preferences - all preferences come from personal memory
    """

    def __init__(self, memory_manager=None):
        super().__init__("temporal_management", memory_manager)
        self.capabilities = [
            "time_query",
            "schedule_optimization",
            "duration_estimation",
            "temporal_conflict_detection",
            "activity_timing_suggestion",
            "schedule_feasibility_check"
        ]

    def get_capabilities(self) -> List[str]:
        return self.capabilities

    def process_query(self, query: MCPMessage) -> MCPMessage:
        """Process temporal-related queries"""

        query_type = query.data.get("query_type")

        if query_type == "time_query":
            return self._handle_time_query(query)
        elif query_type == "schedule_optimization":
            return self._handle_schedule_optimization(query)
        elif query_type == "duration_estimation":
            return self._handle_duration_estimation(query)
        elif query_type == "conflict_detection":
            return self._handle_conflict_detection(query)
        elif query_type == "timing_suggestion":
            return self._handle_timing_suggestion(query)
        elif query_type == "feasibility_check":
            return self._handle_feasibility_check(query)
        else:
            return self._create_response(
                {"error": f"Unknown query type: {query_type}"},
                status="error"
            )

    def _handle_time_query(self, query: MCPMessage) -> MCPMessage:
        """Handle basic time information queries"""

        current_time = datetime.now()

        response_data = {
            "current_time": current_time.isoformat(),
            "current_hour": current_time.hour,
            "current_day": current_time.strftime("%A"),
            "day_type": "weekend" if current_time.weekday() >= 5 else "weekday",
            "time_period": self._get_time_period(current_time.hour)
        }

        return self._create_response(response_data)

    def _handle_schedule_optimization(self, query: MCPMessage) -> MCPMessage:
        """Optimize schedule based on personal temporal preferences"""

        tasks = query.data.get("tasks", [])
        constraints = query.data.get("constraints", {})
        target_date = query.data.get("target_date", datetime.now().isoformat())

        if isinstance(target_date, str):
            target_date = datetime.fromisoformat(target_date)

        # Get personal temporal preferences from memory
        optimized_schedule = []

        for task in tasks:
            activity_type = task.get("activity_type")
            duration = task.get("duration")

            # Get personal preferences for this activity type
            preferred_hour = self._get_personal_preference(
                f"{activity_type}_preferred_hour",
                default_value=12  # Default noon
            )

            preferred_duration = self._get_personal_preference(
                f"{activity_type}_preferred_duration",
                default_value=duration if duration else 60
            )

            # Find optimal time slot
            optimal_time = self._find_optimal_time_slot(
                target_date,
                preferred_hour,
                preferred_duration,
                constraints,
                optimized_schedule
            )

            if optimal_time:
                optimized_schedule.append({
                    "activity_type": activity_type,
                    "start_time": optimal_time.isoformat(),
                    "duration": preferred_duration,
                    "end_time": (optimal_time + timedelta(minutes=preferred_duration)).isoformat(),
                    "confidence": self._calculate_timing_confidence(activity_type, optimal_time)
                })

        response_data = {
            "optimized_schedule": optimized_schedule,
            "total_scheduled_time": sum(task["duration"] for task in optimized_schedule),
            "optimization_method": "personal_preference_based"
        }

        return self._create_response(response_data)

    def _handle_duration_estimation(self, query: MCPMessage) -> MCPMessage:
        """Estimate activity duration based on personal history"""

        activity_type = query.data.get("activity_type")
        context = query.data.get("context", {})

        # Get personal duration preferences from memory
        preferred_duration = self._get_personal_preference(
            f"{activity_type}_preferred_duration",
            default_value=60
        )

        duration_variance = self._get_personal_preference(
            f"{activity_type}_duration_variance",
            default_value=15
        )

        # Add context-based adjustments
        adjusted_duration = preferred_duration
        confidence = 0.8

        if context.get("companions"):
            # Social activities typically take longer
            social_multiplier = self._get_personal_preference(
                f"{activity_type}_group_duration_multiplier",
                default_value=1.2
            )
            adjusted_duration *= social_multiplier
            confidence *= 0.9

        if context.get("day_type") == "weekend":
            # Weekend activities might be more relaxed
            weekend_multiplier = self._get_personal_preference(
                "weekend_duration_multiplier",
                default_value=1.1
            )
            adjusted_duration *= weekend_multiplier
            confidence *= 0.95

        # Calculate range based on personal variance
        min_duration = max(15, adjusted_duration - duration_variance)
        max_duration = adjusted_duration + duration_variance

        response_data = {
            "estimated_duration": int(adjusted_duration),
            "duration_range": {
                "min": int(min_duration),
                "max": int(max_duration)
            },
            "confidence": confidence,
            "basis": "personal_historical_data"
        }

        return self._create_response(response_data)

    def _handle_conflict_detection(self, query: MCPMessage) -> MCPMessage:
        """Detect temporal conflicts in proposed schedule"""

        proposed_activities = query.data.get("activities", [])
        existing_schedule = query.data.get("existing_schedule", [])

        conflicts = []
        all_activities = existing_schedule + proposed_activities

        # Sort activities by start time
        all_activities.sort(key=lambda x: datetime.fromisoformat(x["start_time"]))

        for i in range(len(all_activities) - 1):
            current = all_activities[i]
            next_activity = all_activities[i + 1]

            current_end = datetime.fromisoformat(current["start_time"]) + \
                          timedelta(minutes=current["duration"])
            next_start = datetime.fromisoformat(next_activity["start_time"])

            # Check for overlap
            if current_end > next_start:
                conflicts.append({
                    "type": "overlap",
                    "activity_1": current,
                    "activity_2": next_activity,
                    "overlap_duration": (current_end - next_start).total_seconds() / 60
                })

            # Check for insufficient transition time
            elif (next_start - current_end).total_seconds() / 60 < 15:
                # Get personal minimum transition time preference
                min_transition = self._get_personal_preference(
                    "minimum_transition_time",
                    default_value=15
                )

                if (next_start - current_end).total_seconds() / 60 < min_transition:
                    conflicts.append({
                        "type": "insufficient_transition",
                        "activity_1": current,
                        "activity_2": next_activity,
                        "transition_time": (next_start - current_end).total_seconds() / 60,
                        "required_time": min_transition
                    })

        response_data = {
            "conflicts": conflicts,
            "conflict_count": len(conflicts),
            "feasible": len(conflicts) == 0
        }

        return self._create_response(response_data)

    def _handle_timing_suggestion(self, query: MCPMessage) -> MCPMessage:
        """Suggest optimal timing for an activity based on personal preferences"""

        activity_type = query.data.get("activity_type")
        date = query.data.get("date", datetime.now().isoformat())
        constraints = query.data.get("constraints", {})

        if isinstance(date, str):
            date = datetime.fromisoformat(date)

        # Get personal timing preferences
        preferred_hour = self._get_personal_preference(
            f"{activity_type}_preferred_hour",
            default_value=self._get_default_timing_by_activity(activity_type)
        )

        hour_variance = self._get_personal_preference(
            f"{activity_type}_hour_variance",
            default_value=2
        )

        # Get peak activity hours
        peak_hours = self._get_personal_preference(
            "peak_activity_hours",
            default_value=[12, 18, 20]
        )

        # Generate suggestions
        suggestions = []

        # Primary suggestion based on preference
        primary_time = date.replace(hour=int(preferred_hour), minute=0, second=0, microsecond=0)
        suggestions.append({
            "time": primary_time.isoformat(),
            "confidence": 0.9,
            "reason": "personal_preference",
            "type": "optimal"
        })

        # Alternative suggestions within variance
        for offset in [-hour_variance, hour_variance]:
            alt_hour = int(preferred_hour + offset)
            if 6 <= alt_hour <= 23:  # Reasonable activity hours
                alt_time = date.replace(hour=alt_hour, minute=0, second=0, microsecond=0)
                suggestions.append({
                    "time": alt_time.isoformat(),
                    "confidence": 0.7,
                    "reason": "preference_variance",
                    "type": "alternative"
                })

        # Peak hours suggestions
        for peak_hour in peak_hours[:2]:  # Top 2 peak hours
            if peak_hour != int(preferred_hour):
                peak_time = date.replace(hour=peak_hour, minute=0, second=0, microsecond=0)
                suggestions.append({
                    "time": peak_time.isoformat(),
                    "confidence": 0.6,
                    "reason": "peak_activity_hour",
                    "type": "peak_based"
                })

        # Filter by constraints
        if constraints:
            suggestions = self._filter_by_constraints(suggestions, constraints)

        response_data = {
            "suggestions": suggestions[:5],  # Top 5 suggestions
            "basis": "personal_temporal_preferences"
        }

        return self._create_response(response_data)

    def _handle_feasibility_check(self, query: MCPMessage) -> MCPMessage:
        """Check if proposed schedule is feasible given personal constraints"""

        proposed_schedule = query.data.get("schedule", [])
        date = query.data.get("date", datetime.now().isoformat())

        if isinstance(date, str):
            date = datetime.fromisoformat(date)

        # Get personal constraints from memory
        energy_pattern = self._get_personal_preference("energy_pattern", "moderate")
        max_activities_per_day = self._get_personal_preference("max_activities_per_day", 6)
        preferred_break_duration = self._get_personal_preference("preferred_break_duration", 30)

        feasibility_checks = {
            "total_activities": len(proposed_schedule),
            "exceeds_activity_limit": len(proposed_schedule) > max_activities_per_day,
            "total_active_time": 0,
            "adequate_breaks": True,
            "energy_compatibility": True,
            "temporal_distribution": "balanced"
        }

        # Calculate total active time and check breaks
        if proposed_schedule:
            total_time = sum(activity.get("duration", 60) for activity in proposed_schedule)
            feasibility_checks["total_active_time"] = total_time

            # Check break adequacy between activities
            sorted_schedule = sorted(proposed_schedule,
                                     key=lambda x: datetime.fromisoformat(x["start_time"]))

            for i in range(len(sorted_schedule) - 1):
                current_end = datetime.fromisoformat(sorted_schedule[i]["start_time"]) + \
                              timedelta(minutes=sorted_schedule[i]["duration"])
                next_start = datetime.fromisoformat(sorted_schedule[i + 1]["start_time"])

                break_time = (next_start - current_end).total_seconds() / 60
                if break_time < preferred_break_duration:
                    feasibility_checks["adequate_breaks"] = False
                    break

        # Overall feasibility score
        feasibility_score = 1.0
        if feasibility_checks["exceeds_activity_limit"]:
            feasibility_score -= 0.3
        if not feasibility_checks["adequate_breaks"]:
            feasibility_score -= 0.2
        if feasibility_checks["total_active_time"] > 12 * 60:  # More than 12 hours
            feasibility_score -= 0.3

        feasibility_checks["feasibility_score"] = max(0, feasibility_score)
        feasibility_checks["overall_feasible"] = feasibility_score >= 0.6

        return self._create_response(feasibility_checks)

    def _find_optimal_time_slot(self, date: datetime, preferred_hour: float,
                                duration: int, constraints: Dict,
                                existing_schedule: List) -> Optional[datetime]:
        """Find optimal time slot for activity"""

        # Try preferred time first
        preferred_time = date.replace(
            hour=int(preferred_hour),
            minute=int((preferred_hour % 1) * 60),
            second=0, microsecond=0
        )

        if self._is_time_slot_available(preferred_time, duration, existing_schedule):
            return preferred_time

        # Try nearby slots
        for offset_minutes in [30, -30, 60, -60, 90, -90]:
            candidate_time = preferred_time + timedelta(minutes=offset_minutes)
            if (6 <= candidate_time.hour <= 23 and
                    self._is_time_slot_available(candidate_time, duration, existing_schedule)):
                return candidate_time

        return None

    def _is_time_slot_available(self, start_time: datetime, duration: int,
                                existing_schedule: List) -> bool:
        """Check if time slot is available"""

        end_time = start_time + timedelta(minutes=duration)

        for scheduled_item in existing_schedule:
            scheduled_start = datetime.fromisoformat(scheduled_item["start_time"])
            scheduled_end = scheduled_start + timedelta(minutes=scheduled_item["duration"])

            # Check for overlap
            if start_time < scheduled_end and end_time > scheduled_start:
                return False

        return True

    def _calculate_timing_confidence(self, activity_type: str, timing: datetime) -> float:
        """Calculate confidence score for activity timing"""

        preferred_hour = self._get_personal_preference(
            f"{activity_type}_preferred_hour", 12
        )

        hour_diff = abs(timing.hour - preferred_hour)
        confidence = max(0.3, 1.0 - (hour_diff / 12.0))

        return confidence

    def _get_default_timing_by_activity(self, activity_type: str) -> int:
        """Get reasonable default timing for activities when no personal preference exists"""

        defaults = {
            "work": 9,
            "breakfast": 8,
            "lunch": 12,
            "dinner": 18,
            "exercise": 19,
            "shopping": 14,
            "leisure": 20,
            "social": 19,
            "personal_care": 22
        }

        return defaults.get(activity_type, 12)

    def _get_time_period(self, hour: int) -> str:
        """Classify hour into time period"""

        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    def _filter_by_constraints(self, suggestions: List, constraints: Dict) -> List:
        """Filter timing suggestions by constraints"""

        filtered = []

        earliest = constraints.get("earliest_time")
        latest = constraints.get("latest_time")

        for suggestion in suggestions:
            suggestion_time = datetime.fromisoformat(suggestion["time"])

            valid = True

            if earliest:
                earliest_dt = datetime.fromisoformat(earliest)
                if suggestion_time < earliest_dt:
                    valid = False

            if latest:
                latest_dt = datetime.fromisoformat(latest)
                if suggestion_time > latest_dt:
                    valid = False

            if valid:
                filtered.append(suggestion)

        return filtered