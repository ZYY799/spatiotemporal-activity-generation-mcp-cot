#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : memory_manager.py
# @Time    : 2025/6/15 15:31
# @Desc    : Core memory management system implementing the three-tier memory architecture（Event Memory, Pattern Memory, and Summary Memory）


import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import pickle
import os


@dataclass
class EventMemory:
    """Event-level memory storing specific activity experiences"""
    timestamp: datetime
    location: Tuple[float, float]  # (lat, lon)
    activity_type: str
    conditions: Dict[str, Any]  # Environmental and contextual conditions
    emotion: float  # Emotional feedback [-1, 1]
    duration: float  # Activity duration in minutes
    companions: List[str]  # Social companions
    satisfaction: float  # Post-activity satisfaction [0, 1]
    memory_id: str

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventMemory':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class PatternMemory:
    """Pattern-level memory storing behavioral regularities"""
    pattern_id: str
    condition_signature: str  # Hash of conditions that trigger this pattern
    activity_sequence: List[str]
    probability: float
    frequency: int
    last_updated: datetime
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['last_updated'] = self.last_updated.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternMemory':
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)


@dataclass
class SummaryMemory:
    """Summary-level memory storing abstract preferences and long-term habits"""
    summary_id: str
    category: str  # 'temporal_preference', 'spatial_preference', 'activity_preference', etc.
    summary_data: Dict[str, Any]
    strength: float  # Memory strength [0, 1]
    last_reinforced: datetime

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['last_reinforced'] = self.last_reinforced.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SummaryMemory':
        data['last_reinforced'] = datetime.fromisoformat(data['last_reinforced'])
        return cls(**data)


class PersonalMemoryManager:
    """
    Manages individual memory system with hierarchical architecture
    and dynamic transfer between short-term and long-term memory
    """

    def __init__(self, person_id: str, memory_dir: str = "data/memory"):
        self.person_id = person_id
        self.memory_dir = memory_dir
        self.memory_file = os.path.join(memory_dir, f"{person_id}_memory.pkl")

        # Three-tier memory architecture
        self.event_memories: List[EventMemory] = []
        self.pattern_memories: List[PatternMemory] = []
        self.summary_memories: List[SummaryMemory] = []

        # Short-term and long-term memory
        self.short_term_memory: List[EventMemory] = []
        self.long_term_memory: List[EventMemory] = []

        # Memory parameters
        self.short_term_capacity = 50
        self.transfer_threshold = 0.7
        self.forgetting_rate = 0.01
        self.pattern_extraction_threshold = 3

        # Load existing memory if available
        self.load_memory()

    def add_event_memory(self, event: EventMemory) -> None:
        """Add new event to short-term memory"""
        self.short_term_memory.append(event)

        # Manage short-term memory capacity
        if len(self.short_term_memory) > self.short_term_capacity:
            self._transfer_to_long_term()

        # Update patterns and summaries
        self._update_patterns()
        self._update_summaries()

    def _transfer_to_long_term(self) -> None:
        """Transfer important memories from short-term to long-term"""
        # Calculate importance for each memory
        importance_scores = []
        for memory in self.short_term_memory:
            importance = self._calculate_importance(memory)
            importance_scores.append((memory, importance))

        # Sort by importance and transfer high-importance memories
        importance_scores.sort(key=lambda x: x[1], reverse=True)

        for memory, importance in importance_scores:
            if importance > self.transfer_threshold:
                self.long_term_memory.append(memory)
                self.event_memories.append(memory)

        # Keep only recent memories in short-term
        self.short_term_memory = self.short_term_memory[-self.short_term_capacity // 2:]

    def _calculate_importance(self, memory: EventMemory) -> float:
        """Calculate memory importance based on frequency, recency, and emotional salience"""
        # Frequency component (Hebbian learning)
        frequency = self._get_activity_frequency(memory.activity_type)
        frequency_score = min(frequency / 10.0, 1.0)

        # Recency component (Ebbinghaus forgetting curve)
        hours_ago = (datetime.now() - memory.timestamp).total_seconds() / 3600
        recency_score = np.exp(-self.forgetting_rate * hours_ago)

        # Emotional salience component
        emotional_salience = abs(memory.emotion) * memory.satisfaction

        # Weighted combination
        importance = (0.3 * frequency_score +
                      0.4 * recency_score +
                      0.3 * emotional_salience)

        return importance

    def _get_activity_frequency(self, activity_type: str) -> int:
        """Get frequency of specific activity type"""
        count = 0
        for memory in self.event_memories:
            if memory.activity_type == activity_type:
                count += 1
        return count

    def _update_patterns(self) -> None:
        """Extract and update behavioral patterns"""
        # Group events by similar conditions
        condition_groups = defaultdict(list)

        for memory in self.event_memories[-100:]:  # Consider recent memories
            condition_signature = self._create_condition_signature(memory.conditions)
            condition_groups[condition_signature].append(memory)

        # Extract patterns from frequent condition groups
        for signature, memories in condition_groups.items():
            if len(memories) >= self.pattern_extraction_threshold:
                self._extract_pattern(signature, memories)

    def _create_condition_signature(self, conditions: Dict[str, Any]) -> str:
        """Create a signature string for similar conditions"""
        # Simplify conditions for pattern matching
        simplified = {
            'hour': conditions.get('hour', 0) // 3 * 3,  # 3-hour buckets
            'day_type': conditions.get('day_type', 'weekday'),
            'weather': conditions.get('weather', 'clear'),
            'location_type': conditions.get('location_type', 'unknown')
        }
        return str(sorted(simplified.items()))

    def _extract_pattern(self, signature: str, memories: List[EventMemory]) -> None:
        """Extract behavioral pattern from grouped memories"""
        # Calculate activity sequence probabilities
        activity_sequences = []
        for memory in memories:
            activity_sequences.append(memory.activity_type)

        # Find most common activity for this condition
        activity_counts = defaultdict(int)
        for activity in activity_sequences:
            activity_counts[activity] += 1

        if activity_counts:
            most_common_activity = max(activity_counts, key=activity_counts.get)
            probability = activity_counts[most_common_activity] / len(activity_sequences)

            # Create or update pattern
            pattern_id = f"pattern_{hash(signature)}"
            existing_pattern = self._find_pattern(pattern_id)

            if existing_pattern:
                existing_pattern.frequency += 1
                existing_pattern.probability = (existing_pattern.probability + probability) / 2
                existing_pattern.last_updated = datetime.now()
            else:
                new_pattern = PatternMemory(
                    pattern_id=pattern_id,
                    condition_signature=signature,
                    activity_sequence=[most_common_activity],
                    probability=probability,
                    frequency=len(memories),
                    last_updated=datetime.now(),
                    confidence=min(probability * len(memories) / 10, 1.0)
                )
                self.pattern_memories.append(new_pattern)

    def _find_pattern(self, pattern_id: str) -> Optional[PatternMemory]:
        """Find existing pattern by ID"""
        for pattern in self.pattern_memories:
            if pattern.pattern_id == pattern_id:
                return pattern
        return None

    def _update_summaries(self) -> None:
        """Update summary memories from patterns and events"""
        # Update temporal preferences
        self._update_temporal_preferences()

        # Update spatial preferences
        self._update_spatial_preferences()

        # Update activity preferences
        self._update_activity_preferences()

        # Update social preferences
        self._update_social_preferences()

    def _update_temporal_preferences(self) -> None:
        """Extract temporal preferences from memory"""
        temporal_data = {
            'preferred_start_times': defaultdict(list),
            'preferred_durations': defaultdict(list),
            'peak_activity_hours': defaultdict(int)
        }

        for memory in self.event_memories:
            hour = memory.timestamp.hour
            activity = memory.activity_type

            temporal_data['preferred_start_times'][activity].append(hour)
            temporal_data['preferred_durations'][activity].append(memory.duration)
            temporal_data['peak_activity_hours'][hour] += 1

        # Calculate preferences
        preferences = {}
        for activity, hours in temporal_data['preferred_start_times'].items():
            if len(hours) >= 3:  # Minimum threshold for pattern
                preferences[f"{activity}_preferred_hour"] = np.mean(hours)
                preferences[f"{activity}_hour_variance"] = np.var(hours)

        for activity, durations in temporal_data['preferred_durations'].items():
            if len(durations) >= 3:
                preferences[f"{activity}_preferred_duration"] = np.mean(durations)
                preferences[f"{activity}_duration_variance"] = np.var(durations)

        # Most active hours
        if temporal_data['peak_activity_hours']:
            peak_hours = sorted(temporal_data['peak_activity_hours'].items(),
                                key=lambda x: x[1], reverse=True)[:3]
            preferences['peak_activity_hours'] = [hour for hour, _ in peak_hours]

        self._update_summary_memory('temporal_preference', preferences)

    def _update_spatial_preferences(self) -> None:
        """Extract spatial preferences from memory"""
        spatial_data = {
            'activity_locations': defaultdict(list),
            'preferred_distances': defaultdict(list),
            'location_satisfaction': defaultdict(list)
        }

        for memory in self.event_memories:
            activity = memory.activity_type
            location = memory.location
            satisfaction = memory.satisfaction

            spatial_data['activity_locations'][activity].append(location)
            spatial_data['location_satisfaction'][activity].append(satisfaction)

        # Calculate spatial preferences
        preferences = {}
        for activity, locations in spatial_data['activity_locations'].items():
            if len(locations) >= 3:
                # Calculate centroid of preferred locations
                lats = [loc[0] for loc in locations]
                lons = [loc[1] for loc in locations]
                preferences[f"{activity}_preferred_lat"] = np.mean(lats)
                preferences[f"{activity}_preferred_lon"] = np.mean(lons)
                preferences[f"{activity}_location_variance"] = np.var(lats) + np.var(lons)

        self._update_summary_memory('spatial_preference', preferences)

    def _update_activity_preferences(self) -> None:
        """Extract activity preferences from memory"""
        activity_data = {
            'activity_satisfaction': defaultdict(list),
            'activity_frequency': defaultdict(int),
            'activity_companions': defaultdict(list)
        }

        for memory in self.event_memories:
            activity = memory.activity_type
            activity_data['activity_satisfaction'][activity].append(memory.satisfaction)
            activity_data['activity_frequency'][activity] += 1
            activity_data['activity_companions'][activity].extend(memory.companions)

        # Calculate activity preferences
        preferences = {}
        for activity, satisfactions in activity_data['activity_satisfaction'].items():
            if len(satisfactions) >= 2:
                preferences[f"{activity}_avg_satisfaction"] = np.mean(satisfactions)
                preferences[f"{activity}_satisfaction_stability"] = 1.0 / (np.var(satisfactions) + 0.1)

        # Activity frequency preferences
        total_activities = sum(activity_data['activity_frequency'].values())
        for activity, freq in activity_data['activity_frequency'].items():
            preferences[f"{activity}_frequency_preference"] = freq / total_activities

        self._update_summary_memory('activity_preference', preferences)

    def _update_social_preferences(self) -> None:
        """Extract social preferences from memory"""
        social_data = {
            'companion_frequency': defaultdict(int),
            'solo_vs_group': {'solo': 0, 'group': 0},
            'activity_social_preference': defaultdict(lambda: {'solo': 0, 'group': 0})
        }

        for memory in self.event_memories:
            activity = memory.activity_type
            if memory.companions:
                social_data['solo_vs_group']['group'] += 1
                social_data['activity_social_preference'][activity]['group'] += 1
                for companion in memory.companions:
                    social_data['companion_frequency'][companion] += 1
            else:
                social_data['solo_vs_group']['solo'] += 1
                social_data['activity_social_preference'][activity]['solo'] += 1

        # Calculate social preferences
        preferences = {}
        total_activities = sum(social_data['solo_vs_group'].values())
        if total_activities > 0:
            preferences['solo_preference'] = social_data['solo_vs_group']['solo'] / total_activities
            preferences['group_preference'] = social_data['solo_vs_group']['group'] / total_activities

        # Activity-specific social preferences
        for activity, counts in social_data['activity_social_preference'].items():
            total = counts['solo'] + counts['group']
            if total >= 3:
                preferences[f"{activity}_solo_preference"] = counts['solo'] / total
                preferences[f"{activity}_group_preference"] = counts['group'] / total

        self._update_summary_memory('social_preference', preferences)

    def _update_summary_memory(self, category: str, preferences: Dict[str, Any]) -> None:
        """Update summary memory for a specific category"""
        summary_id = f"{self.person_id}_{category}"

        # Find existing summary or create new one
        existing_summary = None
        for summary in self.summary_memories:
            if summary.summary_id == summary_id:
                existing_summary = summary
                break

        if existing_summary:
            # Update existing summary with exponential moving average
            alpha = 0.3  # Learning rate
            for key, value in preferences.items():
                if key in existing_summary.summary_data:
                    if isinstance(value, (int, float)):
                        existing_summary.summary_data[key] = (
                                alpha * value + (1 - alpha) * existing_summary.summary_data[key]
                        )
                    else:
                        existing_summary.summary_data[key] = value
                else:
                    existing_summary.summary_data[key] = value
            existing_summary.last_reinforced = datetime.now()
            existing_summary.strength = min(existing_summary.strength + 0.1, 1.0)
        else:
            # Create new summary
            new_summary = SummaryMemory(
                summary_id=summary_id,
                category=category,
                summary_data=preferences,
                strength=0.5,
                last_reinforced=datetime.now()
            )
            self.summary_memories.append(new_summary)

    def retrieve_relevant_memories(self, context: Dict[str, Any], k: int = 5) -> Dict[str, List]:
        """Context-sensitive memory retrieval"""
        # Calculate relevance scores for all memories
        event_scores = []
        for memory in self.event_memories:
            score = self._calculate_memory_relevance(memory, context)
            event_scores.append((memory, score))

        pattern_scores = []
        for pattern in self.pattern_memories:
            score = self._calculate_pattern_relevance(pattern, context)
            pattern_scores.append((pattern, score))

        summary_scores = []
        for summary in self.summary_memories:
            score = self._calculate_summary_relevance(summary, context)
            summary_scores.append((summary, score))

        # Sort by relevance and return top-k
        event_scores.sort(key=lambda x: x[1], reverse=True)
        pattern_scores.sort(key=lambda x: x[1], reverse=True)
        summary_scores.sort(key=lambda x: x[1], reverse=True)

        return {
            'events': [memory for memory, _ in event_scores[:k]],
            'patterns': [pattern for pattern, _ in pattern_scores[:k]],
            'summaries': [summary for summary, _ in summary_scores[:k]]
        }

    def _calculate_memory_relevance(self, memory: EventMemory, context: Dict[str, Any]) -> float:
        """Calculate relevance score for event memory"""
        score = 0.0

        # Temporal similarity
        if 'hour' in context:
            hour_diff = abs(memory.timestamp.hour - context['hour'])
            temporal_score = 1.0 - (hour_diff / 12.0)  # Normalize to [0, 1]
            score += 0.3 * temporal_score

        # Spatial similarity
        if 'location' in context and memory.location:
            distance = self._calculate_distance(memory.location, context['location'])
            spatial_score = np.exp(-distance / 5000)  # 5km characteristic distance
            score += 0.3 * spatial_score

        # Activity similarity
        if 'activity_type' in context:
            if memory.activity_type == context['activity_type']:
                score += 0.4

        # Recency boost
        hours_ago = (datetime.now() - memory.timestamp).total_seconds() / 3600
        recency_score = np.exp(-hours_ago / 168)  # One week characteristic time
        score += 0.1 * recency_score

        return score

    def _calculate_pattern_relevance(self, pattern: PatternMemory, context: Dict[str, Any]) -> float:
        """Calculate relevance score for pattern memory"""
        condition_signature = self._create_condition_signature(context)

        # Direct condition match
        if pattern.condition_signature == condition_signature:
            return pattern.confidence * pattern.probability

        # Partial condition match (simplified)
        return 0.1 * pattern.confidence

    def _calculate_summary_relevance(self, summary: SummaryMemory, context: Dict[str, Any]) -> float:
        """Calculate relevance score for summary memory"""
        # Always include summaries as they contain general preferences
        return summary.strength * 0.8

    def _calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate distance between two locations in meters"""
        lat1, lon1 = loc1
        lat2, lon2 = loc2

        # Haversine formula
        R = 6371000  # Earth's radius in meters
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c

        return distance

    def get_preference(self, preference_key: str, default_value: Any = None) -> Any:
        """Get specific preference from summary memories"""
        for summary in self.summary_memories:
            if preference_key in summary.summary_data:
                return summary.summary_data[preference_key]
        return default_value

    def save_memory(self) -> None:
        """Save memory to persistent storage"""
        os.makedirs(self.memory_dir, exist_ok=True)

        memory_data = {
            'person_id': self.person_id,
            'event_memories': [memory.to_dict() for memory in self.event_memories],
            'pattern_memories': [pattern.to_dict() for pattern in self.pattern_memories],
            'summary_memories': [summary.to_dict() for summary in self.summary_memories],
            'short_term_memory': [memory.to_dict() for memory in self.short_term_memory],
            'long_term_memory': [memory.to_dict() for memory in self.long_term_memory],
        }

        with open(self.memory_file, 'wb') as f:
            pickle.dump(memory_data, f)

    def load_memory(self) -> None:
        """Load memory from persistent storage"""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'rb') as f:
                memory_data = pickle.load(f)

            self.event_memories = [EventMemory.from_dict(data) for data in memory_data.get('event_memories', [])]
            self.pattern_memories = [PatternMemory.from_dict(data) for data in memory_data.get('pattern_memories', [])]
            self.summary_memories = [SummaryMemory.from_dict(data) for data in memory_data.get('summary_memories', [])]
            self.short_term_memory = [EventMemory.from_dict(data) for data in memory_data.get('short_term_memory', [])]
            self.long_term_memory = [EventMemory.from_dict(data) for data in memory_data.get('long_term_memory', [])]