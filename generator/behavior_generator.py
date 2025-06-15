#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : behavior_generator.py
# @Time    : 2025/6/15 15:35
# @Desc    : Main behavior generation engine integrating CoT reasoning with MCP tools


import json
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from cot_reasoning import ChainOfThoughtEngine, ReasoningContext
from mcp_tools import MCPToolManager, MCPMessage
from memory import PersonalMemoryManager, EventMemory


@dataclass
class PersonProfile:
    """Individual person profile for behavior generation"""
    person_id: str
    demographics: Dict[str, Any]
    preferences: Dict[str, Any]
    constraints: Dict[str, Any]
    spatial_anchors: Dict[str, Tuple[float, float]]  # home, work, etc.


@dataclass
class ActivityInstance:
    """Single activity instance in a trajectory"""
    activity_id: str
    activity_type: str
    start_time: datetime
    end_time: datetime
    duration: int  # minutes
    location: Tuple[float, float]
    location_name: str
    companions: List[str]
    transportation_mode: str
    route_info: Dict[str, Any]
    satisfaction_prediction: float
    confidence: float
    reasoning_chain: List[Dict[str, Any]]


@dataclass
class DailyTrajectory:
    """Complete daily activity trajectory for an individual"""
    person_id: str
    date: datetime
    activities: List[ActivityInstance]
    total_active_time: int  # minutes
    total_travel_time: int  # minutes
    trajectory_quality: float
    generation_metadata: Dict[str, Any]


class SpatiotemporalBehaviorGenerator:
    """
    Main behavior generation engine implementing the MCP-enhanced CoT framework
    """

    def __init__(self, model, memory_manager: PersonalMemoryManager,
                 mcp_tool_manager: MCPToolManager, config: Dict[str, Any]):
        self.model = model
        self.memory_manager = memory_manager
        self.mcp_tool_manager = mcp_tool_manager
        self.config = config

        # Initialize reasoning engine
        self.reasoning_engine = ChainOfThoughtEngine(mcp_tool_manager, memory_manager)

        # Generation parameters
        self.generation_params = {
            "temperature": config.get("temperature", 0.7),
            "max_activities_per_day": config.get("max_activities_per_day", 8),
            "min_activity_duration": config.get("min_activity_duration", 15),
            "max_activity_duration": config.get("max_activity_duration", 300),
            "daily_time_window": config.get("daily_time_window", (6, 23)),
            "reasoning_depth": config.get("reasoning_depth", "detailed")
        }

    def generate_daily_trajectory(self, person_profile: PersonProfile,
                                  target_date: datetime,
                                  generation_context: Dict[str, Any] = None) -> DailyTrajectory:
        """Generate a complete daily activity trajectory for an individual"""

        try:
            # Initialize generation context
            if generation_context is None:
                generation_context = {}

            # Step 1: Initialize daily structure
            daily_structure = self._initialize_daily_structure(person_profile, target_date)

            # Step 2: Generate activity sequence
            activity_sequence = self._generate_activity_sequence(
                person_profile, target_date, daily_structure, generation_context
            )

            # Step 3: Optimize temporal arrangement
            optimized_sequence = self._optimize_temporal_arrangement(activity_sequence, person_profile)

            # Step 4: Plan routes and transportation
            route_planned_sequence = self._plan_routes_and_transportation(
                optimized_sequence, person_profile
            )

            # Step 5: Validate and refine trajectory
            final_trajectory = self._validate_and_refine_trajectory(
                route_planned_sequence, person_profile, target_date
            )

            # Step 6: Calculate trajectory quality
            trajectory_quality = self._calculate_trajectory_quality(final_trajectory, person_profile)

            # Compile daily trajectory
            daily_trajectory = DailyTrajectory(
                person_id=person_profile.person_id,
                date=target_date,
                activities=final_trajectory,
                total_active_time=sum(activity.duration for activity in final_trajectory),
                total_travel_time=sum(
                    activity.route_info.get("travel_time", 0) for activity in final_trajectory
                ),
                trajectory_quality=trajectory_quality,
                generation_metadata={
                    "generation_timestamp": datetime.now().isoformat(),
                    "model_version": self.config.get("model_version", "unknown"),
                    "reasoning_steps": len(final_trajectory),
                    "generation_method": "mcp_enhanced_cot"
                }
            )

            # Update personal memory with generated trajectory
            self._update_memory_with_trajectory(daily_trajectory)

            return daily_trajectory

        except Exception as e:
            # Return minimal trajectory with error information
            return DailyTrajectory(
                person_id=person_profile.person_id,
                date=target_date,
                activities=[],
                total_active_time=0,
                total_travel_time=0,
                trajectory_quality=0.0,
                generation_metadata={
                    "error": str(e),
                    "generation_timestamp": datetime.now().isoformat(),
                    "generation_method": "mcp_enhanced_cot",
                    "status": "failed"
                }
            )

    def _initialize_daily_structure(self, person_profile: PersonProfile,
                                    target_date: datetime) -> Dict[str, Any]:
        """Initialize basic daily structure based on person profile"""

        day_type = "weekend" if target_date.weekday() >= 5 else "weekday"

        # Get personal schedule preferences from memory
        wake_time = self.memory_manager.get_preference("wake_time_" + day_type,
                                                       7 if day_type == "weekday" else 8)
        sleep_time = self.memory_manager.get_preference("sleep_time_" + day_type, 23)

        # Initialize anchor activities based on day type
        anchor_activities = []

        if day_type == "weekday":
            # Work schedule
            work_start = self.memory_manager.get_preference("work_start_time", 9)
            work_end = self.memory_manager.get_preference("work_end_time", 17)

            if work_start and work_end:
                anchor_activities.append({
                    "activity_type": "work",
                    "start_time": target_date.replace(hour=work_start, minute=0),
                    "end_time": target_date.replace(hour=work_end, minute=0),
                    "location": person_profile.spatial_anchors.get("work"),
                    "fixed": True,
                    "priority": 1.0
                })

        # Home-based activities
        anchor_activities.extend([
            {
                "activity_type": "personal_care",
                "start_time": target_date.replace(hour=wake_time, minute=0),
                "duration": 60,
                "location": person_profile.spatial_anchors.get("home"),
                "fixed": False,
                "priority": 0.8
            },
            {
                "activity_type": "home",
                "start_time": target_date.replace(hour=sleep_time, minute=0),
                "duration": 30,
                "location": person_profile.spatial_anchors.get("home"),
                "fixed": False,
                "priority": 0.9
            }
        ])

        return {
            "day_type": day_type,
            "time_window": (wake_time, sleep_time),
            "anchor_activities": anchor_activities,
            "available_time_slots": self._calculate_available_time_slots(
                anchor_activities, wake_time, sleep_time
            )
        }

    def _generate_activity_sequence(self, person_profile: PersonProfile,
                                    target_date: datetime,
                                    daily_structure: Dict[str, Any],
                                    generation_context: Dict[str, Any]) -> List[ActivityInstance]:
        """Generate sequence of activities using CoT reasoning"""

        activity_sequence = []

        # Start with anchor activities
        for anchor in daily_structure["anchor_activities"]:
            if anchor.get("fixed", False):
                activity_instance = self._create_activity_instance_from_anchor(
                    anchor, person_profile, target_date
                )
                activity_sequence.append(activity_instance)

        # Generate additional activities for available time slots
        available_slots = daily_structure["available_time_slots"]

        for time_slot in available_slots:
            if time_slot["duration"] >= self.generation_params["min_activity_duration"]:
                # Use CoT reasoning to decide on activity for this slot
                activity_decision = self._reason_about_activity_choice(
                    person_profile, target_date, time_slot, activity_sequence, generation_context
                )

                if activity_decision and activity_decision.get("selected_activity"):
                    activity_instance = self._create_activity_instance_from_decision(
                        activity_decision, time_slot, person_profile
                    )
                    activity_sequence.append(activity_instance)

        # Sort activities by start time
        activity_sequence.sort(key=lambda x: x.start_time)

        return activity_sequence

    def _reason_about_activity_choice(self, person_profile: PersonProfile,
                                      target_date: datetime,
                                      time_slot: Dict[str, Any],
                                      existing_activities: List[ActivityInstance],
                                      generation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Use CoT reasoning to decide on activity for time slot"""

        # Prepare reasoning context
        current_situation = {
            "current_time": time_slot["start_time"].isoformat(),
            "current_location": self._get_current_location(time_slot["start_time"], existing_activities),
            "day_type": "weekend" if target_date.weekday() >= 5 else "weekday",
            "available_time": time_slot["duration"],
            "energy_level": self._estimate_energy_level(time_slot["start_time"], existing_activities),
            "social_context": generation_context.get("social_context", {}),
            "available_companions": generation_context.get("available_companions", [])
        }

        # Get environmental context using MCP tools
        environmental_context = self._get_environmental_context(
            current_situation["current_location"],
            time_slot["start_time"]
        )

        # Get relevant memories
        memory_context = self.memory_manager.retrieve_relevant_memories({
            "hour": time_slot["start_time"].hour,
            "location": current_situation["current_location"],
            "day_type": current_situation["day_type"]
        }, k=5)

        # Create reasoning context
        reasoning_context = ReasoningContext(
            person_profile=person_profile.__dict__,
            current_situation=current_situation,
            memory_context=memory_context,
            environmental_context=environmental_context,
            constraints=person_profile.constraints
        )

        # Execute reasoning chain
        reasoning_goal = f"select_activity_for_timeslot_{time_slot['start_time'].hour}"
        reasoning_result = self.reasoning_engine.execute_reasoning_chain(
            reasoning_context, reasoning_goal
        )

        return reasoning_result.get("final_decision", {})

    def _get_environmental_context(self, location: Tuple[float, float],
                                   timestamp: datetime) -> Dict[str, Any]:
        """Get environmental context using MCP environmental tools"""

        environmental_context = {}

        if self.mcp_tool_manager and location:
            # Weather query
            weather_query = MCPMessage(
                message_type="query",
                data={
                    "query_type": "weather_query",
                    "location": location,
                    "timestamp": timestamp.isoformat()
                },
                metadata={},
                timestamp=datetime.now(),
                message_id=f"weather_{timestamp.timestamp()}"
            )

            weather_response = self.mcp_tool_manager.process_query("environmental", weather_query)
            if weather_response.metadata.get("status") == "success":
                environmental_context["weather"] = weather_response.data.get("weather", {})

            # Crowd density query
            crowd_query = MCPMessage(
                message_type="query",
                data={
                    "query_type": "crowd_density",
                    "location": location,
                    "timestamp": timestamp.isoformat()
                },
                metadata={},
                timestamp=datetime.now(),
                message_id=f"crowd_{timestamp.timestamp()}"
            )

            crowd_response = self.mcp_tool_manager.process_query("environmental", crowd_query)
            if crowd_response.metadata.get("status") == "success":
                environmental_context["crowd"] = crowd_response.data.get("crowd_density", {})

        return environmental_context

    def _optimize_temporal_arrangement(self, activity_sequence: List[ActivityInstance],
                                       person_profile: PersonProfile) -> List[ActivityInstance]:
        """Optimize temporal arrangement of activities using temporal MCP tools"""

        if not self.mcp_tool_manager:
            return activity_sequence

        # Prepare schedule data for optimization
        schedule_data = []
        for activity in activity_sequence:
            schedule_data.append({
                "activity_type": activity.activity_type,
                "start_time": activity.start_time.isoformat(),
                "duration": activity.duration,
                "priority": getattr(activity, "priority", 0.5),
                "flexibility": getattr(activity, "flexibility", 0.3)
            })

        # Use temporal MCP tool for optimization
        optimization_query = MCPMessage(
            message_type="query",
            data={
                "query_type": "schedule_optimization",
                "tasks": schedule_data,
                "constraints": person_profile.constraints.get("temporal", {}),
                "target_date": activity_sequence[
                    0].start_time.date().isoformat() if activity_sequence else datetime.now().date().isoformat()
            },
            metadata={},
            timestamp=datetime.now(),
            message_id="schedule_optimization"
        )

        optimization_response = self.mcp_tool_manager.process_query("temporal", optimization_query)

        if optimization_response.metadata.get("status") == "success":
            optimized_schedule = optimization_response.data.get("optimized_schedule", [])

            # Update activity instances with optimized timing
            for i, optimized_item in enumerate(optimized_schedule):
                if i < len(activity_sequence):
                    activity_sequence[i].start_time = datetime.fromisoformat(optimized_item["start_time"])
                    activity_sequence[i].end_time = datetime.fromisoformat(optimized_item["end_time"])
                    activity_sequence[i].duration = optimized_item["duration"]

        return activity_sequence

    def _plan_routes_and_transportation(self, activity_sequence: List[ActivityInstance],
                                        person_profile: PersonProfile) -> List[ActivityInstance]:
        """Plan routes and transportation between activities using spatial MCP tools"""

        if not self.mcp_tool_manager or len(activity_sequence) < 2:
            return activity_sequence

        for i in range(len(activity_sequence) - 1):
            current_activity = activity_sequence[i]
            next_activity = activity_sequence[i + 1]

            # Plan route between current and next activity
            route_query = MCPMessage(
                message_type="query",
                data={
                    "query_type": "route_planning",
                    "origin": current_activity.location,
                    "destination": next_activity.location,
                    "transport_modes": ["walking", "public_transit", "car", "cycling"]
                },
                metadata={},
                timestamp=datetime.now(),
                message_id=f"route_planning_{i}"
            )

            route_response = self.mcp_tool_manager.process_query("spatial", route_query)

            if route_response.metadata.get("status") == "success":
                routes = route_response.data.get("routes", [])
                recommended_route = route_response.data.get("recommended_route")

                if recommended_route:
                    # Update transportation info for next activity
                    next_activity.transportation_mode = recommended_route["transport_mode"]
                    next_activity.route_info = {
                        "origin": current_activity.location,
                        "destination": next_activity.location,
                        "travel_time": recommended_route["travel_time"],
                        "distance": recommended_route["distance"],
                        "cost": recommended_route.get("cost", 0),
                        "route_details": recommended_route
                    }

                    # Adjust start time if travel time affects schedule
                    travel_time_minutes = recommended_route["travel_time"]
                    required_departure = next_activity.start_time - timedelta(minutes=travel_time_minutes)

                    if required_departure < current_activity.end_time:
                        # Adjust timing to accommodate travel
                        time_conflict = (current_activity.end_time - required_departure).total_seconds() / 60
                        next_activity.start_time = current_activity.end_time + timedelta(minutes=travel_time_minutes)
                        next_activity.end_time = next_activity.start_time + timedelta(minutes=next_activity.duration)

        return activity_sequence

    def _validate_and_refine_trajectory(self, activity_sequence: List[ActivityInstance],
                                        person_profile: PersonProfile,
                                        target_date: datetime) -> List[ActivityInstance]:
        """Validate and refine the complete trajectory"""

        refined_sequence = []

        for activity in activity_sequence:
            # Validate activity feasibility
            if self._validate_activity_feasibility(activity, person_profile):
                # Refine activity details
                refined_activity = self._refine_activity_details(activity, person_profile)
                refined_sequence.append(refined_activity)
            else:
                # Log validation failure but continue with other activities
                print(f"Activity {activity.activity_type} at {activity.start_time} failed validation")

        # Check for temporal conflicts and resolve
        refined_sequence = self._resolve_temporal_conflicts(refined_sequence)

        # Ensure minimum breaks between activities
        refined_sequence = self._ensure_minimum_breaks(refined_sequence, person_profile)

        return refined_sequence

    def _validate_activity_feasibility(self, activity: ActivityInstance,
                                       person_profile: PersonProfile) -> bool:
        """Validate if activity is feasible given constraints"""

        # Check temporal feasibility
        if activity.duration < self.generation_params["min_activity_duration"]:
            return False

        if activity.duration > self.generation_params["max_activity_duration"]:
            return False

        # Check spatial feasibility (basic distance check)
        if activity.location and person_profile.spatial_anchors.get("home"):
            home_location = person_profile.spatial_anchors["home"]
            distance = self._calculate_distance(activity.location, home_location)
            max_distance = person_profile.constraints.get("spatial", {}).get("max_distance", 50000)  # 50km default

            if distance > max_distance:
                return False

        # Check resource feasibility
        estimated_cost = getattr(activity, "estimated_cost", 0)
        budget_limit = person_profile.constraints.get("financial", {}).get("daily_budget", 1000)

        if estimated_cost > budget_limit:
            return False

        return True

    def _refine_activity_details(self, activity: ActivityInstance,
                                 person_profile: PersonProfile) -> ActivityInstance:
        """Refine activity details using MCP evaluation tools"""

        if not self.mcp_tool_manager:
            return activity

        # Evaluate activity using MCP evaluation tools
        evaluation_query = MCPMessage(
            message_type="query",
            data={
                "query_type": "activity_evaluation",
                "activity": {
                    "type": activity.activity_type,
                    "duration": activity.duration,
                    "location": activity.location,
                    "companions": activity.companions
                },
                "criteria": ["satisfaction", "duration", "cost", "accessibility"]
            },
            metadata={},
            timestamp=datetime.now(),
            message_id=f"activity_evaluation_{activity.activity_id}"
        )

        evaluation_response = self.mcp_tool_manager.process_query("evaluation", evaluation_query)

        if evaluation_response.metadata.get("status") == "success":
            evaluation_data = evaluation_response.data.get("activity_evaluation", {})

            # Update activity with evaluation results
            activity.satisfaction_prediction = evaluation_data.get("overall_score", 0.5)
            activity.confidence = evaluation_data.get("confidence", 0.5)

            # Apply recommendations if available
            recommendations = evaluation_response.data.get("recommendations", [])
            for rec in recommendations:
                if rec.get("type") == "duration_adjustment":
                    activity.duration = max(self.generation_params["min_activity_duration"],
                                            min(self.generation_params["max_activity_duration"],
                                                rec.get("suggested_duration", activity.duration)))

        return activity

    def _calculate_trajectory_quality(self, activity_sequence: List[ActivityInstance],
                                      person_profile: PersonProfile) -> float:
        """Calculate overall quality score for the trajectory"""

        if not activity_sequence:
            return 0.0

        quality_factors = {
            "satisfaction": self._calculate_satisfaction_score(activity_sequence),
            "feasibility": self._calculate_feasibility_score(activity_sequence, person_profile),
            "diversity": self._calculate_diversity_score(activity_sequence),
            "efficiency": self._calculate_efficiency_score(activity_sequence),
            "preference_alignment": self._calculate_preference_alignment_score(activity_sequence, person_profile)
        }

        # Weighted combination
        weights = {
            "satisfaction": 0.3,
            "feasibility": 0.25,
            "diversity": 0.15,
            "efficiency": 0.15,
            "preference_alignment": 0.15
        }

        overall_quality = sum(
            quality_factors[factor] * weights[factor]
            for factor in quality_factors
        )

        return min(1.0, max(0.0, overall_quality))

    def _calculate_satisfaction_score(self, activity_sequence: List[ActivityInstance]) -> float:
        """Calculate predicted satisfaction score"""

        if not activity_sequence:
            return 0.0

        satisfaction_scores = [activity.satisfaction_prediction for activity in activity_sequence]
        return sum(satisfaction_scores) / len(satisfaction_scores)

    def _calculate_feasibility_score(self, activity_sequence: List[ActivityInstance],
                                     person_profile: PersonProfile) -> float:
        """Calculate feasibility score based on constraints"""

        feasibility_score = 1.0

        # Check temporal feasibility
        total_time = sum(activity.duration for activity in activity_sequence)
        available_time = 16 * 60  # 16 hours available time

        if total_time > available_time:
            feasibility_score *= 0.5

        # Check travel time reasonableness
        total_travel_time = sum(
            activity.route_info.get("travel_time", 0) for activity in activity_sequence
        )

        if total_travel_time > total_time * 0.3:  # More than 30% travel time
            feasibility_score *= 0.8

        return feasibility_score

    def _calculate_diversity_score(self, activity_sequence: List[ActivityInstance]) -> float:
        """Calculate diversity score based on activity variety"""

        if not activity_sequence:
            return 0.0

        activity_types = set(activity.activity_type for activity in activity_sequence)
        diversity_score = len(activity_types) / len(activity_sequence)

        return min(1.0, diversity_score * 2)  # Scale to favor diversity

    def _calculate_efficiency_score(self, activity_sequence: List[ActivityInstance]) -> float:
        """Calculate efficiency score based on time utilization"""

        if not activity_sequence:
            return 0.0

        total_time = sum(activity.duration for activity in activity_sequence)
        total_travel_time = sum(
            activity.route_info.get("travel_time", 0) for activity in activity_sequence
        )

        if total_time + total_travel_time == 0:
            return 0.0

        efficiency = total_time / (total_time + total_travel_time)
        return efficiency

    def _calculate_preference_alignment_score(self, activity_sequence: List[ActivityInstance],
                                              person_profile: PersonProfile) -> float:
        """Calculate preference alignment score"""

        if not activity_sequence or not self.memory_manager:
            return 0.5

        alignment_scores = []

        for activity in activity_sequence:
            # Check activity frequency preference
            freq_pref = self.memory_manager.get_preference(
                f"{activity.activity_type}_frequency_preference", 0.5
            )

            # Check timing preference
            preferred_hour = self.memory_manager.get_preference(
                f"{activity.activity_type}_preferred_hour", 12
            )
            hour_diff = abs(activity.start_time.hour - preferred_hour)
            timing_alignment = max(0, 1.0 - hour_diff / 12)

            # Combine alignment factors
            activity_alignment = (freq_pref * 0.6 + timing_alignment * 0.4)
            alignment_scores.append(activity_alignment)

        return sum(alignment_scores) / len(alignment_scores)

    def _update_memory_with_trajectory(self, trajectory: DailyTrajectory) -> None:
        """Update personal memory with generated trajectory"""

        for activity in trajectory.activities:
            # Create event memory from activity
            event_memory = EventMemory(
                timestamp=activity.start_time,
                location=activity.location,
                activity_type=activity.activity_type,
                conditions={
                    "day_type": "weekend" if trajectory.date.weekday() >= 5 else "weekday",
                    "hour": activity.start_time.hour,
                    "companions": activity.companions,
                    "transportation": activity.transportation_mode
                },
                emotion=0.0,  # Neutral for generated activities
                duration=activity.duration,
                companions=activity.companions,
                satisfaction=activity.satisfaction_prediction,
                memory_id=activity.activity_id
            )

            # Add to memory (marked as synthetic)
            event_memory.conditions["synthetic"] = True
            self.memory_manager.add_event_memory(event_memory)

    # Helper methods

    def _calculate_available_time_slots(self, anchor_activities: List[Dict[str, Any]],
                                        wake_time: int, sleep_time: int) -> List[Dict[str, Any]]:
        """Calculate available time slots between anchor activities"""

        time_slots = []

        # Sort anchor activities by start time
        sorted_anchors = sorted([a for a in anchor_activities if a.get("start_time")],
                                key=lambda x: x["start_time"])

        current_time = datetime.now().replace(hour=wake_time, minute=0, second=0, microsecond=0)
        end_time = datetime.now().replace(hour=sleep_time, minute=0, second=0, microsecond=0)

        for anchor in sorted_anchors:
            anchor_start = anchor["start_time"]

            # Check if there's a gap before this anchor
            if current_time < anchor_start:
                gap_duration = (anchor_start - current_time).total_seconds() / 60
                if gap_duration >= self.generation_params["min_activity_duration"]:
                    time_slots.append({
                        "start_time": current_time,
                        "end_time": anchor_start,
                        "duration": int(gap_duration)
                    })

            # Update current time to end of anchor activity
            anchor_end = anchor.get("end_time")
            if not anchor_end and anchor.get("duration"):
                anchor_end = anchor_start + timedelta(minutes=anchor["duration"])

            if anchor_end:
                current_time = max(current_time, anchor_end)

        # Check for gap after last anchor
        if current_time < end_time:
            gap_duration = (end_time - current_time).total_seconds() / 60
            if gap_duration >= self.generation_params["min_activity_duration"]:
                time_slots.append({
                    "start_time": current_time,
                    "end_time": end_time,
                    "duration": int(gap_duration)
                })

        return time_slots

    def _get_current_location(self, timestamp: datetime,
                              existing_activities: List[ActivityInstance]) -> Tuple[float, float]:
        """Estimate current location at given timestamp"""

        # Find the most recent activity before this timestamp
        relevant_activities = [a for a in existing_activities if a.end_time <= timestamp]

        if relevant_activities:
            # Return location of most recent activity
            recent_activity = max(relevant_activities, key=lambda x: x.end_time)
            return recent_activity.location
        else:
            # Default to home location if available
            if hasattr(self, 'person_profile') and self.person_profile.spatial_anchors.get("home"):
                return self.person_profile.spatial_anchors["home"]
            else:
                # Default Lujiazui coordinates
                return (31.240, 121.505)

    def _estimate_energy_level(self, timestamp: datetime,
                               existing_activities: List[ActivityInstance]) -> float:
        """Estimate energy level at given timestamp"""

        # Simple energy model based on time of day and recent activities
        hour = timestamp.hour

        # Base energy pattern (higher in morning, lower in evening)
        if 6 <= hour <= 10:
            base_energy = 0.8
        elif 10 <= hour <= 14:
            base_energy = 0.9
        elif 14 <= hour <= 18:
            base_energy = 0.7
        elif 18 <= hour <= 22:
            base_energy = 0.6
        else:
            base_energy = 0.4

        # Adjust based on recent activities
        recent_activities = [a for a in existing_activities
                             if a.end_time <= timestamp and
                             (timestamp - a.end_time).total_seconds() <= 3600]  # Within 1 hour

        energy_drain = sum(0.1 for _ in recent_activities)  # Each activity drains 0.1 energy

        final_energy = max(0.1, base_energy - energy_drain)
        return final_energy

    def _create_activity_instance_from_anchor(self, anchor: Dict[str, Any],
                                              person_profile: PersonProfile,
                                              target_date: datetime) -> ActivityInstance:
        """Create activity instance from anchor activity"""

        activity_id = f"{person_profile.person_id}_{anchor['activity_type']}_{anchor['start_time'].hour}"

        end_time = anchor.get("end_time")
        if not end_time and anchor.get("duration"):
            end_time = anchor["start_time"] + timedelta(minutes=anchor["duration"])

        return ActivityInstance(
            activity_id=activity_id,
            activity_type=anchor["activity_type"],
            start_time=anchor["start_time"],
            end_time=end_time or anchor["start_time"] + timedelta(hours=1),
            duration=anchor.get("duration", 60),
            location=anchor.get("location", person_profile.spatial_anchors.get("home", (31.240, 121.505))),
            location_name=anchor.get("location_name", "Unknown"),
            companions=[],
            transportation_mode="walking",
            route_info={},
            satisfaction_prediction=0.7,
            confidence=0.8,
            reasoning_chain=[]
        )

    def _create_activity_instance_from_decision(self, decision: Dict[str, Any],
                                                time_slot: Dict[str, Any],
                                                person_profile: PersonProfile) -> ActivityInstance:
        """Create activity instance from CoT reasoning decision"""

        selected_activity = decision.get("selected_activity", {})

        activity_id = f"{person_profile.person_id}_{selected_activity.get('activity_type', 'unknown')}_{time_slot['start_time'].hour}"

        # Extract activity details from decision
        activity_type = selected_activity.get("activity_type", "leisure")
        duration = min(selected_activity.get("duration", 60), time_slot["duration"])
        location = selected_activity.get("location", person_profile.spatial_anchors.get("home", (31.240, 121.505)))
        companions = selected_activity.get("companions", [])

        end_time = time_slot["start_time"] + timedelta(minutes=duration)

        return ActivityInstance(
            activity_id=activity_id,
            activity_type=activity_type,
            start_time=time_slot["start_time"],
            end_time=end_time,
            duration=duration,
            location=location,
            location_name=selected_activity.get("location_name", "Selected Location"),
            companions=companions,
            transportation_mode="walking",
            route_info={},
            satisfaction_prediction=decision.get("expected_satisfaction", 0.6),
            confidence=decision.get("decision_confidence", 0.6),
            reasoning_chain=decision.get("reasoning_chain", [])
        )

    def _calculate_distance(self, loc1: Tuple[float, float],
                            loc2: Tuple[float, float]) -> float:
        """Calculate distance between two locations in meters"""

        import math

        lat1, lon1 = loc1
        lat2, lon2 = loc2

        R = 6371000  # Earth's radius in meters
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _resolve_temporal_conflicts(self, activity_sequence: List[ActivityInstance]) -> List[ActivityInstance]:
        """Resolve temporal conflicts in activity sequence"""

        if len(activity_sequence) <= 1:
            return activity_sequence

        resolved_sequence = []

        for i, activity in enumerate(activity_sequence):
            if i == 0:
                resolved_sequence.append(activity)
                continue

            previous_activity = resolved_sequence[-1]

            # Check for overlap
            if activity.start_time < previous_activity.end_time:
                # Adjust start time to avoid overlap
                activity.start_time = previous_activity.end_time + timedelta(minutes=5)
                activity.end_time = activity.start_time + timedelta(minutes=activity.duration)

            resolved_sequence.append(activity)

        return resolved_sequence

    def _ensure_minimum_breaks(self, activity_sequence: List[ActivityInstance],
                               person_profile: PersonProfile) -> List[ActivityInstance]:
        """Ensure minimum breaks between activities"""

        min_break = self.memory_manager.get_preference("minimum_break_duration", 15) if self.memory_manager else 15

        adjusted_sequence = []

        for i, activity in enumerate(activity_sequence):
            if i == 0:
                adjusted_sequence.append(activity)
                continue

            previous_activity = adjusted_sequence[-1]
            time_gap = (activity.start_time - previous_activity.end_time).total_seconds() / 60

            if time_gap < min_break:
                # Add minimum break
                required_adjustment = min_break - time_gap
                activity.start_time += timedelta(minutes=required_adjustment)
                activity.end_time += timedelta(minutes=required_adjustment)

            adjusted_sequence.append(activity)

        return adjusted_sequence

