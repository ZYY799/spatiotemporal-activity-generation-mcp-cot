#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : activity_chain_builder.py
# @Time    : 2025/6/15 15:36
# @Desc    : Activity chain builder for constructing realistic activity sequences


from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import random
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, Counter


@dataclass
class ActivityChainCandidate:
    """Activity chain candidate solution"""
    activity_sequence: List[str]
    durations: List[int]
    confidence_scores: List[float]
    reasoning_chain: List[Dict[str, Any]]
    total_satisfaction_prediction: float
    feasibility_score: float


class MemoryBasedActivityChainBuilder:
    """
    Memory-based activity chain builder
    Learns and generates from personal historical experiences instead of fixed patterns
    """

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.reasoning_temperature = 0.7  # Control randomness in reasoning

    def build_activity_chain(self, person_profile: Dict[str, Any],
                             day_type: str, available_time: int,
                             current_context: Dict[str, Any] = None,
                             constraints: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Build activity chain based on personal memory"""

        if current_context is None:
            current_context = {}

        # Step 1: Analyze personal historical patterns (learn from memory)
        personal_patterns = self._analyze_personal_patterns(day_type, current_context)

        # Step 2: Identify similar historical experiences for current context
        similar_experiences = self._find_similar_experiences(current_context, available_time)

        # Step 3: Generate activity candidates based on memory
        activity_candidates = self._generate_memory_based_candidates(
            personal_patterns, similar_experiences, available_time, current_context
        )

        # Step 4: Evaluate and select the best activity chain
        best_chain = self._select_best_chain(activity_candidates, constraints)

        # Step 5: Personalize the chain
        personalized_chain = self._personalize_chain(best_chain, person_profile)

        return personalized_chain

    def _analyze_personal_patterns(self, day_type: str,
                                   current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze behavioral patterns from personal memory"""

        # Get relevant historical memories
        context_for_retrieval = {
            "day_type": day_type,
            **current_context
        }
        relevant_memories = self.memory_manager.retrieve_relevant_memories(
            context_for_retrieval, k=20
        )

        patterns = {
            "activity_transitions": defaultdict(lambda: defaultdict(int)),
            "time_preferences": defaultdict(list),
            "duration_preferences": defaultdict(list),
            "satisfaction_history": defaultdict(list),
            "social_patterns": defaultdict(list),
            "location_patterns": defaultdict(list)
        }

        # Analyze event memories
        events = relevant_memories.get('events', [])
        for i, event in enumerate(events):
            # Activity transition patterns
            if i < len(events) - 1:
                next_event = events[i + 1]
                time_gap = (next_event.timestamp - event.timestamp).total_seconds() / 3600
                if time_gap < 8:  # Transitions within the same day
                    patterns["activity_transitions"][event.activity_type][next_event.activity_type] += 1

            # Time preferences
            patterns["time_preferences"][event.activity_type].append(event.timestamp.hour)

            # Duration preferences
            patterns["duration_preferences"][event.activity_type].append(event.duration)

            # Satisfaction history
            patterns["satisfaction_history"][event.activity_type].append(event.satisfaction)

            # Social patterns
            social_level = len(event.companions) if event.companions else 0
            patterns["social_patterns"][event.activity_type].append(social_level)

            # Location patterns
            patterns["location_patterns"][event.activity_type].append(event.location)

        # Analyze pattern memories
        pattern_memories = relevant_memories.get('patterns', [])
        learned_sequences = {}
        for pattern in pattern_memories:
            learned_sequences[pattern.condition_signature] = {
                "sequence": pattern.activity_sequence,
                "confidence": pattern.confidence,
                "frequency": pattern.frequency
            }

        return {
            "transition_patterns": dict(patterns["activity_transitions"]),
            "time_preferences": dict(patterns["time_preferences"]),
            "duration_preferences": dict(patterns["duration_preferences"]),
            "satisfaction_history": dict(patterns["satisfaction_history"]),
            "social_patterns": dict(patterns["social_patterns"]),
            "location_patterns": dict(patterns["location_patterns"]),
            "learned_sequences": learned_sequences
        }

    def _find_similar_experiences(self, current_context: Dict[str, Any],
                                  available_time: int) -> List[Dict[str, Any]]:
        """Find historical experiences in similar contexts"""

        # Build similarity search context
        search_context = {
            "hour": current_context.get("current_hour", 12),
            "day_type": current_context.get("day_type", "weekday"),
            "weather": current_context.get("weather", "clear"),
            "available_time": available_time
        }

        similar_memories = self.memory_manager.retrieve_relevant_memories(
            search_context, k=15
        )

        experiences = []
        for event in similar_memories.get('events', []):
            # Calculate context similarity
            similarity_score = self._calculate_context_similarity(
                current_context, {
                    "hour": event.timestamp.hour,
                    "day_type": "weekend" if event.timestamp.weekday() >= 5 else "weekday",
                    "weather": event.conditions.get("weather", "clear"),
                    "duration": event.duration
                }
            )

            if similarity_score > 0.3:  # Only consider sufficiently similar experiences
                experiences.append({
                    "event": event,
                    "similarity": similarity_score,
                    "success_score": event.satisfaction * (1 + event.emotion)  # Combined satisfaction and emotion
                })

        # Sort by similarity and success rate
        experiences.sort(key=lambda x: x["similarity"] * x["success_score"], reverse=True)
        return experiences[:10]  # Return top 10 most relevant experiences

    def _calculate_context_similarity(self, context1: Dict[str, Any],
                                      context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts"""

        similarity = 0.0
        total_weight = 0.0

        # Time similarity
        if "hour" in context1 and "hour" in context2:
            hour_diff = abs(context1["hour"] - context2["hour"])
            hour_similarity = 1.0 - min(hour_diff, 12) / 12.0
            similarity += hour_similarity * 0.3
            total_weight += 0.3

        # Day type similarity
        if "day_type" in context1 and "day_type" in context2:
            day_similarity = 1.0 if context1["day_type"] == context2["day_type"] else 0.3
            similarity += day_similarity * 0.4
            total_weight += 0.4

        # Weather similarity
        if "weather" in context1 and "weather" in context2:
            weather_similarity = 1.0 if context1["weather"] == context2["weather"] else 0.5
            similarity += weather_similarity * 0.2
            total_weight += 0.2

        # Time duration similarity
        if "available_time" in context1 and "duration" in context2:
            time_ratio = min(context1["available_time"], context2["duration"]) / max(context1["available_time"],
                                                                                     context2["duration"])
            similarity += time_ratio * 0.1
            total_weight += 0.1

        return similarity / total_weight if total_weight > 0 else 0.0

    def _generate_memory_based_candidates(self, personal_patterns: Dict[str, Any],
                                          similar_experiences: List[Dict[str, Any]],
                                          available_time: int,
                                          current_context: Dict[str, Any]) -> List[ActivityChainCandidate]:
        """Generate activity chain candidates based on memory"""

        candidates = []

        # Strategy 1: Based on successful historical experiences
        candidates.extend(self._generate_from_successful_experiences(
            similar_experiences, available_time, current_context
        ))

        # Strategy 2: Based on learned transition patterns
        candidates.extend(self._generate_from_transition_patterns(
            personal_patterns, available_time, current_context
        ))

        # Strategy 3: Based on innovative combinations of personal preferences
        candidates.extend(self._generate_innovative_combinations(
            personal_patterns, available_time, current_context
        ))

        # Strategy 4: Based on underexplored activities
        candidates.extend(self._generate_exploration_candidates(
            personal_patterns, available_time, current_context
        ))

        return candidates

    def _generate_from_successful_experiences(self, similar_experiences: List[Dict[str, Any]],
                                              available_time: int,
                                              current_context: Dict[str, Any]) -> List[ActivityChainCandidate]:
        """Generate candidates based on successful historical experiences"""

        candidates = []

        for exp in similar_experiences[:5]:  # Take top 5 most similar successful experiences
            event = exp["event"]

            if event.satisfaction > 0.6:  # Only consider experiences with high satisfaction
                # Create activity chain based on this successful experience
                activity_sequence = [event.activity_type]
                durations = [min(event.duration, available_time)]
                confidence_scores = [exp["similarity"] * event.satisfaction]

                reasoning_chain = [{
                    "reasoning_type": "successful_experience_replication",
                    "source_experience": {
                        "timestamp": event.timestamp.isoformat(),
                        "satisfaction": event.satisfaction,
                        "similarity": exp["similarity"]
                    },
                    "confidence": confidence_scores[0]
                }]

                # If there's still time, try to add related activities
                remaining_time = available_time - durations[0]
                if remaining_time > 30:  # At least 30 minutes remaining
                    next_activities = self._predict_next_activities(
                        event.activity_type, remaining_time, current_context
                    )

                    for next_activity, duration, confidence in next_activities:
                        if duration <= remaining_time:
                            activity_sequence.append(next_activity)
                            durations.append(duration)
                            confidence_scores.append(confidence)
                            remaining_time -= duration

                            reasoning_chain.append({
                                "reasoning_type": "pattern_based_extension",
                                "activity": next_activity,
                                "confidence": confidence
                            })

                candidate = ActivityChainCandidate(
                    activity_sequence=activity_sequence,
                    durations=durations,
                    confidence_scores=confidence_scores,
                    reasoning_chain=reasoning_chain,
                    total_satisfaction_prediction=np.mean(confidence_scores),
                    feasibility_score=self._calculate_feasibility(activity_sequence, durations)
                )

                candidates.append(candidate)

        return candidates

    def _generate_from_transition_patterns(self, personal_patterns: Dict[str, Any],
                                           available_time: int,
                                           current_context: Dict[str, Any]) -> List[ActivityChainCandidate]:
        """Generate candidates based on learned transition patterns"""

        candidates = []
        transition_patterns = personal_patterns.get("transition_patterns", {})

        # Select possible starting activities
        current_hour = current_context.get("current_hour", 12)
        possible_starts = self._get_preferred_activities_for_time(
            personal_patterns, current_hour
        )

        for start_activity in possible_starts[:3]:  # Consider top 3 most likely starting activities
            chain = self._build_chain_from_transitions(
                start_activity, transition_patterns, available_time,
                personal_patterns, current_context
            )

            if chain and len(chain["activities"]) > 0:
                candidate = ActivityChainCandidate(
                    activity_sequence=chain["activities"],
                    durations=chain["durations"],
                    confidence_scores=chain["confidences"],
                    reasoning_chain=chain["reasoning"],
                    total_satisfaction_prediction=np.mean(chain["confidences"]),
                    feasibility_score=self._calculate_feasibility(
                        chain["activities"], chain["durations"]
                    )
                )
                candidates.append(candidate)

        return candidates

    def _get_preferred_activities_for_time(self, personal_patterns: Dict[str, Any],
                                           hour: int) -> List[str]:
        """Get preferred activities for specific time"""

        time_preferences = personal_patterns.get("time_preferences", {})
        activity_scores = []

        for activity, hours in time_preferences.items():
            if hours:
                # Calculate preference score for this activity at current time
                hour_distances = [abs(h - hour) for h in hours]
                avg_distance = np.mean(hour_distances)
                preference_score = 1.0 / (1.0 + avg_distance / 6.0)  # 6 hours as characteristic distance
                activity_scores.append((activity, preference_score))

        # Sort by preference score
        activity_scores.sort(key=lambda x: x[1], reverse=True)
        return [activity for activity, _ in activity_scores]

    def _build_chain_from_transitions(self, start_activity: str,
                                      transition_patterns: Dict[str, Dict[str, int]],
                                      available_time: int,
                                      personal_patterns: Dict[str, Any],
                                      current_context: Dict[str, Any]) -> Dict[str, List]:
        """Build activity chain from transition patterns"""

        activities = [start_activity]
        durations = []
        confidences = []
        reasoning = []

        current_activity = start_activity
        remaining_time = available_time

        # Get preferred duration for starting activity
        duration_prefs = personal_patterns.get("duration_preferences", {})
        if current_activity in duration_prefs and duration_prefs[current_activity]:
            preferred_duration = int(np.mean(duration_prefs[current_activity]))
        else:
            preferred_duration = 60  # Default 1 hour

        actual_duration = min(preferred_duration, remaining_time)
        durations.append(actual_duration)

        # Calculate confidence score based on historical satisfaction
        satisfaction_history = personal_patterns.get("satisfaction_history", {})
        if current_activity in satisfaction_history and satisfaction_history[current_activity]:
            confidence = np.mean(satisfaction_history[current_activity])
        else:
            confidence = 0.5
        confidences.append(confidence)

        reasoning.append({
            "reasoning_type": "time_preference_based_start",
            "activity": current_activity,
            "confidence": confidence
        })

        remaining_time -= actual_duration

        # Build subsequent activity chain
        max_chain_length = 5  # Limit chain length
        chain_length = 1

        while remaining_time > 30 and chain_length < max_chain_length:
            # Find common follow-up activities for current activity
            if current_activity in transition_patterns:
                transitions = transition_patterns[current_activity]
                if transitions:
                    # Select most common transition, but add some randomness
                    next_activities = list(transitions.keys())
                    transition_counts = list(transitions.values())

                    # Calculate probability distribution
                    total_count = sum(transition_counts)
                    probabilities = [count / total_count for count in transition_counts]

                    # Use temperature parameter to adjust selection randomness
                    adjusted_probs = np.array(probabilities) ** (1.0 / self.reasoning_temperature)
                    adjusted_probs = adjusted_probs / np.sum(adjusted_probs)

                    # Select next activity
                    next_activity = np.random.choice(next_activities, p=adjusted_probs)

                    # Calculate confidence for this transition
                    transition_confidence = transitions[next_activity] / total_count

                    # Get activity duration
                    if next_activity in duration_prefs and duration_prefs[next_activity]:
                        preferred_duration = int(np.mean(duration_prefs[next_activity]))
                    else:
                        preferred_duration = 60

                    actual_duration = min(preferred_duration, remaining_time)

                    # Calculate overall confidence score
                    activity_satisfaction = 0.5
                    if next_activity in satisfaction_history and satisfaction_history[next_activity]:
                        activity_satisfaction = np.mean(satisfaction_history[next_activity])

                    overall_confidence = (transition_confidence + activity_satisfaction) / 2

                    activities.append(next_activity)
                    durations.append(actual_duration)
                    confidences.append(overall_confidence)

                    reasoning.append({
                        "reasoning_type": "transition_pattern_based",
                        "from_activity": current_activity,
                        "to_activity": next_activity,
                        "transition_frequency": transitions[next_activity],
                        "confidence": overall_confidence
                    })

                    current_activity = next_activity
                    remaining_time -= actual_duration
                    chain_length += 1
                else:
                    break
            else:
                break

        return {
            "activities": activities,
            "durations": durations,
            "confidences": confidences,
            "reasoning": reasoning
        }

    def _generate_innovative_combinations(self, personal_patterns: Dict[str, Any],
                                          available_time: int,
                                          current_context: Dict[str, Any]) -> List[ActivityChainCandidate]:
        """Generate innovative activity combinations based on personal preferences"""

        candidates = []

        # Get activities with high personal satisfaction
        satisfaction_history = personal_patterns.get("satisfaction_history", {})
        high_satisfaction_activities = []

        for activity, satisfactions in satisfaction_history.items():
            if satisfactions and np.mean(satisfactions) > 0.7:
                high_satisfaction_activities.append((activity, np.mean(satisfactions)))

        # Sort by satisfaction
        high_satisfaction_activities.sort(key=lambda x: x[1], reverse=True)

        # Generate innovative combinations
        if len(high_satisfaction_activities) >= 2:
            # Combine top high-satisfaction activities
            for i in range(min(3, len(high_satisfaction_activities))):
                for j in range(i + 1, min(i + 3, len(high_satisfaction_activities))):
                    activity1, satisfaction1 = high_satisfaction_activities[i]
                    activity2, satisfaction2 = high_satisfaction_activities[j]

                    # Check if this combination is feasible
                    duration_prefs = personal_patterns.get("duration_preferences", {})

                    duration1 = 60  # Default value
                    if activity1 in duration_prefs and duration_prefs[activity1]:
                        duration1 = int(np.mean(duration_prefs[activity1]))

                    duration2 = 60
                    if activity2 in duration_prefs and duration_prefs[activity2]:
                        duration2 = int(np.mean(duration_prefs[activity2]))

                    total_duration = duration1 + duration2

                    if total_duration <= available_time:
                        candidate = ActivityChainCandidate(
                            activity_sequence=[activity1, activity2],
                            durations=[duration1, duration2],
                            confidence_scores=[satisfaction1, satisfaction2],
                            reasoning_chain=[
                                {
                                    "reasoning_type": "innovative_combination",
                                    "strategy": "combine_high_satisfaction_activities",
                                    "activity1": activity1,
                                    "activity2": activity2,
                                    "expected_satisfaction": (satisfaction1 + satisfaction2) / 2
                                }
                            ],
                            total_satisfaction_prediction=(satisfaction1 + satisfaction2) / 2,
                            feasibility_score=0.8  # Slightly lower feasibility due to innovation
                        )
                        candidates.append(candidate)

        return candidates

    def _generate_exploration_candidates(self, personal_patterns: Dict[str, Any],
                                         available_time: int,
                                         current_context: Dict[str, Any]) -> List[ActivityChainCandidate]:
        """Generate exploratory activity candidates"""

        candidates = []

        # Get personal preferred activity types
        satisfaction_history = personal_patterns.get("satisfaction_history", {})
        tried_activities = set(satisfaction_history.keys())

        # Define all possible activity types
        all_activities = {
            "work", "home", "shopping", "dining", "leisure", "exercise",
            "social", "travel", "healthcare", "education", "personal_care",
            "reading", "cooking", "gaming", "music", "art", "volunteering",
            "meditation", "photography", "gardening"
        }

        # Find untried or rarely tried activities
        unexplored_activities = all_activities - tried_activities
        underexplored_activities = []

        for activity in tried_activities:
            if len(satisfaction_history[activity]) < 3:  # Tried fewer than 3 times
                underexplored_activities.append(activity)

        exploration_targets = list(unexplored_activities) + underexplored_activities

        if exploration_targets and available_time >= 60:
            # Randomly select an exploration target
            exploration_activity = random.choice(exploration_targets)
            exploration_duration = min(90, available_time)  # Give more time to exploration activities

            candidate = ActivityChainCandidate(
                activity_sequence=[exploration_activity],
                durations=[exploration_duration],
                confidence_scores=[0.6],  # Medium confidence due to exploration
                reasoning_chain=[
                    {
                        "reasoning_type": "exploration",
                        "strategy": "try_new_activity",
                        "activity": exploration_activity,
                        "rationale": "Expanding personal experience and discovering new preferences"
                    }
                ],
                total_satisfaction_prediction=0.6,
                feasibility_score=0.7
            )
            candidates.append(candidate)

        return candidates

    def _predict_next_activities(self, current_activity: str, remaining_time: int,
                                 current_context: Dict[str, Any]) -> List[Tuple[str, int, float]]:
        """Predict possible activities after current activity"""

        # Get relevant transition patterns
        context_for_retrieval = {
            "activity_type": current_activity,
            **current_context
        }

        relevant_memories = self.memory_manager.retrieve_relevant_memories(
            context_for_retrieval, k=10
        )

        next_activities = defaultdict(list)

        # Analyze historical transitions
        events = relevant_memories.get('events', [])
        for i, event in enumerate(events[:-1]):
            if event.activity_type == current_activity:
                next_event = events[i + 1]
                time_gap = (next_event.timestamp - event.timestamp).total_seconds() / 60

                if time_gap <= remaining_time and time_gap >= 15:  # Reasonable time interval
                    next_activities[next_event.activity_type].append({
                        'duration': int(time_gap),
                        'satisfaction': next_event.satisfaction
                    })

        # Build prediction results
        predictions = []
        for activity, instances in next_activities.items():
            if len(instances) >= 2:  # Appeared at least twice
                avg_duration = int(np.mean([inst['duration'] for inst in instances]))
                avg_satisfaction = np.mean([inst['satisfaction'] for inst in instances])
                confidence = min(len(instances) / 5.0, 1.0)  # Frequency-based confidence

                predictions.append((activity, avg_duration, confidence * avg_satisfaction))

        # Sort by confidence
        predictions.sort(key=lambda x: x[2], reverse=True)
        return predictions[:3]  # Return top 3 predictions

    def _select_best_chain(self, candidates: List[ActivityChainCandidate],
                           constraints: Dict[str, Any] = None) -> ActivityChainCandidate:
        """Select the best activity chain candidate"""

        if not candidates:
            return self._create_fallback_chain()

        # Calculate overall score for each candidate
        scored_candidates = []

        for candidate in candidates:
            score = self._calculate_candidate_score(candidate, constraints)
            scored_candidates.append((candidate, score))

        # Select candidate with highest score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0]

    def _calculate_candidate_score(self, candidate: ActivityChainCandidate,
                                   constraints: Dict[str, Any] = None) -> float:
        """Calculate overall score for candidate"""

        # Basic score components
        satisfaction_score = candidate.total_satisfaction_prediction
        feasibility_score = candidate.feasibility_score
        confidence_score = np.mean(candidate.confidence_scores) if candidate.confidence_scores else 0.5

        # Diversity score (encourage activity diversity)
        diversity_score = len(set(candidate.activity_sequence)) / len(candidate.activity_sequence)

        # Apply constraint penalties
        constraint_penalty = 0.0
        if constraints:
            constraint_penalty = self._calculate_constraint_penalty(candidate, constraints)

        # Overall score
        total_score = (
                0.4 * satisfaction_score +
                0.3 * feasibility_score +
                0.2 * confidence_score +
                0.1 * diversity_score -
                constraint_penalty
        )

        return max(0.0, total_score)

    def _calculate_constraint_penalty(self, candidate: ActivityChainCandidate,
                                      constraints: Dict[str, Any]) -> float:
        """Calculate penalty for constraint violations"""

        penalty = 0.0

        # Time constraints
        total_time = sum(candidate.durations)
        if 'max_time' in constraints and total_time > constraints['max_time']:
            penalty += 0.5

        # Activity type constraints
        if 'forbidden_activities' in constraints:
            forbidden = set(constraints['forbidden_activities'])
            if any(activity in forbidden for activity in candidate.activity_sequence):
                penalty += 0.8

        # Budget constraints (simplified version)
        if 'max_budget' in constraints:
            estimated_cost = self._estimate_chain_cost(candidate)
            if estimated_cost > constraints['max_budget']:
                penalty += 0.3

        return penalty

    def _estimate_chain_cost(self, candidate: ActivityChainCandidate) -> float:
        """Estimate cost of activity chain"""

        cost_estimates = {
            "dining": 50, "entertainment": 80, "shopping": 100,
            "exercise": 30, "social": 40, "leisure": 20,
            "travel": 25, "personal_care": 10,
            "work": 0, "home": 0, "education": 15
        }

        total_cost = 0.0
        for i, activity in enumerate(candidate.activity_sequence):
            base_cost = cost_estimates.get(activity, 20)
            duration_hours = candidate.durations[i] / 60.0
            total_cost += base_cost * min(duration_hours, 2.0)  # Cost doesn't grow linearly too much

        return total_cost

    def _calculate_feasibility(self, activities: List[str], durations: List[int]) -> float:
        """Calculate feasibility score of activity chain"""

        if not activities or not durations:
            return 0.0

        feasibility = 1.0

        # Check reasonableness of activity durations
        for i, (activity, duration) in enumerate(zip(activities, durations)):
            if duration < 15:  # Too short
                feasibility *= 0.7
            elif duration > 300:  # Too long (over 5 hours)
                feasibility *= 0.8

        # Check reasonableness of activity transitions
        for i in range(len(activities) - 1):
            current_activity = activities[i]
            next_activity = activities[i + 1]

            # Some activity transitions are less reasonable
            if (current_activity == "exercise" and next_activity == "dining" and
                    durations[i] > 120):  # Eating immediately after long exercise
                feasibility *= 0.9

            if current_activity == "work" and next_activity == "exercise":
                feasibility *= 1.1  # Exercise after work is good

        return feasibility

    def _create_fallback_chain(self) -> ActivityChainCandidate:
        """Create fallback basic activity chain"""

        return ActivityChainCandidate(
            activity_sequence=["leisure"],
            durations=[90],
            confidence_scores=[0.5],
            reasoning_chain=[{
                "reasoning_type": "fallback",
                "rationale": "No suitable candidates found, using safe default activity"
            }],
            total_satisfaction_prediction=0.5,
            feasibility_score=0.8
        )

    def _personalize_chain(self, chain: ActivityChainCandidate,
                           person_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Personalize the selected activity chain"""

        personalized_activities = []

        for i, activity_type in enumerate(chain.activity_sequence):
            duration = chain.durations[i]
            confidence = chain.confidence_scores[i] if i < len(chain.confidence_scores) else 0.5

            # Get personal preference for social settings
            social_preference = self._get_social_preference(activity_type)
            companions = self._suggest_companions(activity_type, social_preference)

            # Get preferred location
            preferred_location = self._get_preferred_location(activity_type)

            # Adjust duration based on personal preferences
            adjusted_duration = self._adjust_duration_for_person(activity_type, duration)

            personalized_activity = {
                "activity_type": activity_type,
                "duration": adjusted_duration,
                "social_setting": "group" if companions else "solo",
                "companions": companions,
                "preferred_location_type": preferred_location,
                "confidence": confidence,
                "flexibility": 0.3,  # Default flexibility
                "priority": self._get_activity_priority(activity_type),
                "reasoning": chain.reasoning_chain[i] if i < len(chain.reasoning_chain) else {}
            }

            personalized_activities.append(personalized_activity)

        return personalized_activities

    def _get_social_preference(self, activity_type: str) -> float:
        """Get social preference for specific activity"""

        social_pref = self.memory_manager.get_preference(
            f"{activity_type}_social_preference", 0.5
        )
        return social_pref

    def _suggest_companions(self, activity_type: str, social_preference: float) -> List[str]:
        """Suggest companions based on social preference"""

        if social_preference < 0.3:
            return []

        # Get common companions based on historical memory
        relevant_memories = self.memory_manager.retrieve_relevant_memories(
            {"activity_type": activity_type}, k=10
        )

        companion_counts = defaultdict(int)
        for event in relevant_memories.get('events', []):
            if event.activity_type == activity_type:
                for companion in event.companions:
                    companion_counts[companion] += 1

        if companion_counts:
            # Return most common companion
            most_common = max(companion_counts, key=companion_counts.get)
            return [most_common]

        # Default companion suggestions
        if activity_type in ["social", "dining"]:
            return ["friend"] if social_preference > 0.6 else []
        elif activity_type == "exercise":
            return ["workout_partner"] if social_preference > 0.7 else []

        return []

    def _get_preferred_location(self, activity_type: str) -> str:
        """Get preferred location type for activity"""

        # Get location preference from memory
        location_mapping = {
            "work": "office",
            "home": "residential",
            "shopping": "commercial",
            "dining": "restaurant",
            "leisure": "entertainment",
            "exercise": "gym",
            "social": "entertainment",
            "travel": "transport",
            "healthcare": "medical",
            "education": "educational",
            "personal_care": "residential"
        }

        return location_mapping.get(activity_type, "general")

    def _adjust_duration_for_person(self, activity_type: str, default_duration: int) -> int:
        """Adjust activity duration based on personal preferences"""

        preferred_duration = self.memory_manager.get_preference(
            f"{activity_type}_preferred_duration", default_duration
        )

        # Balance between preferred duration and default duration
        adjusted = int(0.7 * preferred_duration + 0.3 * default_duration)

        # Ensure within reasonable range
        return max(15, min(300, adjusted))

    def _get_activity_priority(self, activity_type: str) -> float:
        """Get activity priority"""

        priority_map = {
            "work": 1.0, "personal_care": 0.9, "healthcare": 0.9,
            "dining": 0.8, "exercise": 0.7, "education": 0.7,
            "shopping": 0.6, "social": 0.5, "leisure": 0.4,
            "entertainment": 0.3
        }

        base_priority = priority_map.get(activity_type, 0.5)

        # Adjust priority based on personal historical satisfaction
        personal_satisfaction = self.memory_manager.get_preference(
            f"{activity_type}_avg_satisfaction", 0.5
        )

        # Slightly increase priority for activities with high satisfaction
        adjusted_priority = base_priority * (0.8 + 0.4 * personal_satisfaction)

        return min(1.0, adjusted_priority)