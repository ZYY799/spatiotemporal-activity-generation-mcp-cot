#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : memory_generator.py
# @Time    : 2025/6/15 15:31
# @Desc    : Generate initial personal memory data for individuals with data-driven approach

"""
Personal Memory Generator

This module generates realistic initial memory data for individuals based on real mobility patterns
and activity distributions.

PRIVACY NOTICE:
Original experiments used real survey data for demographic characteristics, behavioral preferences,
and personal attributes. To protect data privacy and security, this open-source version uses
randomized generation for all personal attributes as demonstration examples. Only aggregated
population-level location and activity patterns are used from anonymized datasets.
"""

import json
import random
import string
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from .memory_manager import PersonalMemoryManager, EventMemory
import uuid
import os


class PersonalMemoryGenerator:
    """Generate realistic initial memory data for individuals"""

    def __init__(self, data_dir: str = "data/memory", use_data_files: bool = True):
        self.data_dir = data_dir
        self.use_data_files = use_data_files

        self.activity_types = [
            "work", "home", "shopping", "dining", "leisure", "exercise",
            "social", "travel", "healthcare", "education", "personal_care"
        ]

        self.location_types = [
            "office", "residential", "commercial", "restaurant", "park",
            "gym", "entertainment", "transport", "medical", "educational"
        ]

        self.weather_conditions = ["sunny", "cloudy", "rainy", "snowy"]

        # Lujiazui area bounds (fallback)
        self.lujiazui_bounds = {
            'lat_min': 31.235, 'lat_max': 31.250,
            'lon_min': 121.495, 'lon_max': 121.520
        }

        # Population type definitions
        self.population_types = [
            "employment_workers",  # Workers employed in Lujiazui
            "local_residents",  # Local residents of Lujiazui
            "tourists",  # Tourist visitors
            "business_visitors",  # Business travelers
            "local_consumers",  # Local entertainment/shopping consumers
            "others"  # Other categories
        ]

        self._load_data_files()

    def _load_data_files(self):
        """Load CSV data files"""
        try:
            if self.use_data_files:
                # Load population distribution data (work/home coordinates)
                population_file = os.path.join(self.data_dir, "population_distribution.csv")
                if os.path.exists(population_file):
                    self.population_data = pd.read_csv(population_file)
                    print(f"Loaded population distribution data: {len(self.population_data)} records")
                else:
                    print(f"Population file not found: {population_file}, using random generation")
                    self.population_data = None

                # Load activity time probability density data
                activity_time_file = os.path.join(self.data_dir, "activity_time_distribution.csv")
                if os.path.exists(activity_time_file):
                    self.activity_time_data = pd.read_csv(activity_time_file)
                    print(f"Loaded activity time distribution data: {self.activity_time_data.shape}")
                else:
                    print(f"Activity time file not found: {activity_time_file}, using random generation")
                    self.activity_time_data = None

                # Load activity duration distribution data
                duration_file = os.path.join(self.data_dir, "activity_duration_distribution.csv")
                if os.path.exists(duration_file):
                    self.duration_data = pd.read_csv(duration_file)
                    print(f"Loaded activity duration distribution data: {self.duration_data.shape}")
                else:
                    print(f"Duration file not found: {duration_file}, using random generation")
                    self.duration_data = None
            else:
                print("Configured to not use data files, using random generation")
                self.population_data = None
                self.activity_time_data = None
                self.duration_data = None

        except Exception as e:
            print(f"Error loading data files: {e}, falling back to random generation")
            self.population_data = None
            self.activity_time_data = None
            self.duration_data = None

    def _get_population_type_distribution(self) -> Dict[str, float]:
        """Get population type distribution ratios"""
        if self.population_data is not None and 'population_type' in self.population_data.columns:
            type_counts = self.population_data['population_type'].value_counts()
            total = type_counts.sum()
            return {ptype: count / total for ptype, count in type_counts.items()}
        else:
            # Default distribution (example data for demonstration)
            return {
                "employment_workers": 0.45,
                "local_residents": 0.25,
                "tourists": 0.10,
                "business_visitors": 0.10,
                "local_consumers": 0.08,
                "others": 0.02
            }

    def _select_population_type(self) -> str:
        """Select population type based on distribution probabilities"""
        distribution = self._get_population_type_distribution()
        return np.random.choice(
            list(distribution.keys()),
            p=list(distribution.values())
        )

    def _get_location_from_data(self, location_type: str, population_type: str) -> Optional[Tuple[float, float]]:
        """Get location coordinates from data"""
        if self.population_data is None:
            return None

        filtered_data = self.population_data[
            self.population_data['population_type'] == population_type
            ]

        if filtered_data.empty:
            return None

        if location_type == 'work' and 'work_lat' in filtered_data.columns:
            sample = filtered_data.sample(n=1)
            return (sample['work_lat'].iloc[0], sample['work_lon'].iloc[0])
        elif location_type == 'home' and 'home_lat' in filtered_data.columns:
            sample = filtered_data.sample(n=1)
            return (sample['home_lat'].iloc[0], sample['home_lon'].iloc[0])

        return None

    def _get_activity_probability_at_time(self, time_slot: int, activity_type: str) -> float:
        """Get activity probability at specific time slot"""
        if self.activity_time_data is None:
            return self._default_activity_probability(time_slot, activity_type)

        time_data = self.activity_time_data[
            (self.activity_time_data['time_slot'] == time_slot) &
            (self.activity_time_data['activity_type'] == activity_type)
            ]

        if not time_data.empty:
            return time_data['probability'].iloc[0]
        else:
            return self._default_activity_probability(time_slot, activity_type)

    def _default_activity_probability(self, time_slot: int, activity_type: str) -> float:
        """Default activity probability distribution when no data available"""
        hour = time_slot // 4  # Convert 15-min slots to hours

        if activity_type == "work":
            return 0.8 if 9 <= hour <= 17 else 0.1
        elif activity_type == "home":
            return 0.9 if hour < 8 or hour > 19 else 0.3
        elif activity_type == "dining":
            return 0.7 if hour in [12, 13, 18, 19] else 0.1
        elif activity_type == "leisure":
            if 19 <= hour <= 22:
                return 0.6
            elif 10 <= hour <= 16:
                return 0.4
            else:
                return 0.1
        else:
            return 0.2

    def _get_activity_duration_from_data(self, activity_type: str) -> int:
        """Get activity duration from data"""
        if self.duration_data is None:
            return self._default_activity_duration(activity_type)

        duration_info = self.duration_data[
            self.duration_data['activity_type'] == activity_type
            ]

        if not duration_info.empty:
            mean_duration = duration_info['mean_duration'].iloc[0]
            std_duration = duration_info['std_duration'].iloc[0]

            duration = np.random.normal(mean_duration, std_duration)
            return max(15, int(duration))  # Minimum 15 minutes
        else:
            return self._default_activity_duration(activity_type)

    def _default_activity_duration(self, activity_type: str) -> int:
        """Default activity duration in minutes"""
        default_durations = {
            "work": random.randint(480, 540),  # 8-9 hours
            "home": random.randint(600, 720),  # 10-12 hours
            "shopping": random.randint(60, 180),  # 1-3 hours
            "dining": random.randint(45, 120),  # 45min-2 hours
            "leisure": random.randint(90, 240),  # 1.5-4 hours
            "exercise": random.randint(45, 120),  # 45min-2 hours
            "social": random.randint(120, 300),  # 2-5 hours
            "travel": random.randint(30, 90),  # 30min-1.5 hours
            "healthcare": random.randint(60, 180),  # 1-3 hours
            "education": random.randint(120, 240),  # 2-4 hours
            "personal_care": random.randint(30, 90)  # 30min-1.5 hours
        }
        return default_durations.get(activity_type, random.randint(60, 180))

    def generate_person_profile(self, person_id: str, population_type: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive person profile

        Note: Demographics, preferences, and behavioral characteristics are randomly generated
        for privacy protection. Original research used real survey data.
        """

        if population_type is None:
            population_type = self._select_population_type()

        # Demographics (randomized for privacy)
        age = random.randint(22, 65)
        gender = random.choice(['male', 'female'])
        occupation = random.choice([
            'finance_professional', 'office_worker', 'manager', 'consultant',
            'teacher', 'engineer', 'designer', 'sales', 'service_worker'
        ])

        household_type = random.choice([
            'single', 'couple_no_children', 'family_with_children', 'shared_housing'
        ])

        # Work characteristics (randomized for privacy)
        work_flexibility = random.choice(['fixed', 'flexible', 'remote_hybrid'])
        work_hours = {
            'start': random.randint(8, 10),
            'end': random.randint(17, 19)
        }

        # Transport preferences (randomized for privacy)
        transport_modes = {
            'public_transit': random.uniform(0.3, 0.8),
            'walking': random.uniform(0.4, 0.9),
            'cycling': random.uniform(0.1, 0.6),
            'car': random.uniform(0.1, 0.7) if age >= 25 else random.uniform(0.05, 0.3),
            'taxi_rideshare': random.uniform(0.2, 0.6)
        }

        # Activity preferences (randomized for privacy)
        activity_preferences = {}
        for activity in self.activity_types:
            activity_preferences[activity] = {
                'frequency': random.uniform(0.1, 1.0),
                'preferred_duration': self._get_activity_duration_from_data(activity),
                'time_flexibility': random.uniform(0.2, 0.8),
                'social_preference': random.uniform(0.0, 1.0)
            }

        # Temporal preferences (randomized for privacy)
        temporal_preferences = {
            'morning_person': random.uniform(0.0, 1.0),
            'evening_person': random.uniform(0.0, 1.0),
            'weekend_late_riser': random.uniform(0.0, 1.0),
            'planning_horizon': random.choice(['spontaneous', 'short_term', 'long_term'])
        }

        # Spatial preferences (based on data or random)
        home_location = self._get_location_from_data('home', population_type) or \
                        self._generate_random_location('residential')
        work_location = self._get_location_from_data('work', population_type) or \
                        self._generate_random_location('office')

        spatial_preferences = {
            'home_location': home_location,
            'work_location': work_location,
            'exploration_tendency': random.uniform(0.2, 0.8),
            'distance_tolerance': random.uniform(0.3, 1.0)
        }

        return {
            'person_id': person_id,
            'population_type': population_type,
            'demographics': {
                'age': age,
                'gender': gender,
                'occupation': occupation,
                'household_type': household_type
            },
            'work_characteristics': {
                'flexibility': work_flexibility,
                'hours': work_hours
            },
            'transport_preferences': transport_modes,
            'activity_preferences': activity_preferences,
            'temporal_preferences': temporal_preferences,
            'spatial_preferences': spatial_preferences
        }

    def _generate_daily_activities_with_data(self, profile: Dict[str, Any],
                                             date: datetime, day_type: str) -> List[Dict[str, Any]]:
        """Generate daily activity sequence based on data"""
        activities = []
        current_time = date.replace(hour=6, minute=0, second=0, microsecond=0)

        # Generate activities by 15-minute time slots
        for time_slot in range(72):  # 24 hours * 4 (15-min slots)
            slot_time = current_time + timedelta(minutes=time_slot * 15)

            # Calculate probabilities for each activity type
            activity_probs = {}
            for activity_type in self.activity_types:
                activity_probs[activity_type] = self._get_activity_probability_at_time(
                    time_slot, activity_type
                )

            # Select most likely activity
            if random.random() < 0.8:  # 80% of time slots have activities
                selected_activity = max(activity_probs, key=activity_probs.get)

                duration = self._get_activity_duration_from_data(selected_activity)

                activities.append({
                    'timestamp': slot_time,
                    'activity_type': selected_activity,
                    'location_type': self._get_location_type_for_activity(selected_activity),
                    'duration': duration,
                    'conditions': {
                        'day_type': day_type,
                        'weather': random.choice(self.weather_conditions)
                    }
                })

                # Skip corresponding time slots
                skip_slots = duration // 15
                time_slot += skip_slots

        return activities

    def generate_initial_memories(self, profile: Dict[str, Any],
                                  num_days: int = 30) -> PersonalMemoryManager:
        """Generate initial memory data based on profile"""

        person_id = profile['person_id']
        memory_manager = PersonalMemoryManager(person_id)

        start_date = datetime.now() - timedelta(days=num_days)

        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            day_type = 'weekend' if current_date.weekday() >= 5 else 'weekday'

            if self.activity_time_data is not None:
                daily_activities = self._generate_daily_activities_with_data(
                    profile, current_date, day_type
                )
            else:
                daily_activities = self._generate_daily_activities_fallback(
                    profile, current_date, day_type
                )

            for activity_data in daily_activities:
                event_memory = self._create_event_memory(activity_data, profile)
                memory_manager.add_event_memory(event_memory)

        return memory_manager

    def _generate_daily_activities_fallback(self, profile: Dict[str, Any],
                                            date: datetime, day_type: str) -> List[Dict[str, Any]]:
        """Fallback activity generation method"""
        activities = []
        current_time = date.replace(hour=6, minute=0, second=0, microsecond=0)

        # Morning routine
        activities.append({
            'timestamp': current_time + timedelta(minutes=random.randint(0, 60)),
            'activity_type': 'personal_care',
            'location_type': 'residential',
            'duration': random.randint(30, 90),
            'conditions': {'day_type': day_type, 'weather': random.choice(self.weather_conditions)}
        })

        if day_type == 'weekday':
            work_start = profile['work_characteristics']['hours']['start']
            work_end = profile['work_characteristics']['hours']['end']

            # Commute
            activities.append({
                'timestamp': current_time + timedelta(hours=work_start - 1),
                'activity_type': 'travel',
                'location_type': 'transport',
                'duration': random.randint(20, 60),
                'conditions': {'day_type': day_type, 'weather': random.choice(self.weather_conditions)}
            })

            # Work
            activities.append({
                'timestamp': current_time + timedelta(hours=work_start),
                'activity_type': 'work',
                'location_type': 'office',
                'duration': (work_end - work_start) * 60,
                'conditions': {'day_type': day_type, 'weather': random.choice(self.weather_conditions)}
            })

            # Lunch
            if random.random() < 0.8:
                lunch_time = current_time + timedelta(hours=12)
                activities.append({
                    'timestamp': lunch_time,
                    'activity_type': 'dining',
                    'location_type': 'restaurant',
                    'duration': random.randint(30, 90),
                    'conditions': {'day_type': day_type, 'weather': random.choice(self.weather_conditions)}
                })

            # Evening activities
            evening_time = current_time + timedelta(hours=work_end + 1)
            evening_activities = ['shopping', 'dining', 'exercise', 'social', 'leisure']
            num_evening = random.randint(1, 3)

            for _ in range(num_evening):
                if random.random() < 0.7:
                    activity_type = random.choice(evening_activities)
                    duration = profile['activity_preferences'][activity_type]['preferred_duration']
                    duration += random.randint(-30, 30)

                    activities.append({
                        'timestamp': evening_time,
                        'activity_type': activity_type,
                        'location_type': self._get_location_type_for_activity(activity_type),
                        'duration': max(30, duration),
                        'conditions': {'day_type': day_type, 'weather': random.choice(self.weather_conditions)}
                    })

                    evening_time += timedelta(minutes=duration + random.randint(15, 45))

        else:
            # Weekend pattern
            weekend_activities = ['leisure', 'social', 'shopping', 'exercise', 'dining', 'travel']
            num_activities = random.randint(2, 5)
            activity_time = current_time + timedelta(hours=random.randint(9, 11))

            for _ in range(num_activities):
                activity_type = random.choice(weekend_activities)
                duration = profile['activity_preferences'][activity_type]['preferred_duration']
                duration += random.randint(-30, 60)

                activities.append({
                    'timestamp': activity_time,
                    'activity_type': activity_type,
                    'location_type': self._get_location_type_for_activity(activity_type),
                    'duration': max(30, duration),
                    'conditions': {'day_type': day_type, 'weather': random.choice(self.weather_conditions)}
                })

                activity_time += timedelta(minutes=duration + random.randint(30, 120))

        return activities

    def _get_location_type_for_activity(self, activity_type: str) -> str:
        """Map activity type to appropriate location type"""
        mapping = {
            'work': 'office',
            'home': 'residential',
            'shopping': 'commercial',
            'dining': 'restaurant',
            'leisure': 'entertainment',
            'exercise': 'gym',
            'social': 'entertainment',
            'travel': 'transport',
            'healthcare': 'medical',
            'education': 'educational',
            'personal_care': 'residential'
        }
        return mapping.get(activity_type, 'commercial')

    def _generate_random_location(self, location_type: str) -> Tuple[float, float]:
        """Generate random location within Lujiazui area"""
        if location_type == 'office':
            lat = random.uniform(31.238, 31.245)
            lon = random.uniform(121.498, 121.508)
        elif location_type == 'residential':
            lat = random.uniform(31.235, 31.250)
            lon = random.uniform(121.495, 121.520)
        elif location_type == 'commercial':
            lat = random.uniform(31.236, 31.248)
            lon = random.uniform(121.497, 121.515)
        else:
            lat = random.uniform(self.lujiazui_bounds['lat_min'],
                                 self.lujiazui_bounds['lat_max'])
            lon = random.uniform(self.lujiazui_bounds['lon_min'],
                                 self.lujiazui_bounds['lon_max'])
        return (lat, lon)

    def _create_event_memory(self, activity_data: Dict[str, Any],
                             profile: Dict[str, Any]) -> EventMemory:
        """Create EventMemory instance from activity data"""

        if activity_data['activity_type'] == 'work':
            location = profile['spatial_preferences']['work_location']
        elif activity_data['activity_type'] == 'home':
            location = profile['spatial_preferences']['home_location']
        else:
            location = self._generate_random_location(activity_data['location_type'])

        companions = self._generate_companions(activity_data['activity_type'], profile)

        activity_pref = profile['activity_preferences'][activity_data['activity_type']]
        base_satisfaction = activity_pref['frequency']

        satisfaction = np.clip(base_satisfaction + random.gauss(0, 0.2), 0, 1)
        emotion = random.gauss(satisfaction - 0.5, 0.3)
        emotion = np.clip(emotion, -1, 1)

        return EventMemory(
            timestamp=activity_data['timestamp'],
            location=location,
            activity_type=activity_data['activity_type'],
            conditions=activity_data['conditions'],
            emotion=emotion,
            duration=activity_data['duration'],
            companions=companions,
            satisfaction=satisfaction,
            memory_id=str(uuid.uuid4())
        )

    def _generate_companions(self, activity_type: str, profile: Dict[str, Any]) -> List[str]:
        """Generate companions based on activity type and social preferences"""
        activity_pref = profile['activity_preferences'][activity_type]
        social_pref = activity_pref['social_preference']

        companions = []

        if random.random() < social_pref:
            if activity_type in ['social', 'dining']:
                num_companions = random.randint(1, 4)
            elif activity_type in ['exercise', 'leisure']:
                num_companions = random.randint(1, 2)
            else:
                num_companions = 1 if random.random() < 0.3 else 0

            companion_pool = ['friend_1', 'friend_2', 'colleague_1', 'family_member', 'partner']
            companions = random.sample(companion_pool, min(num_companions, len(companion_pool)))

        return companions

    def _generate_person_id(self, index: int) -> str:
        """Generate unique person ID with population type prefix and random letters"""
        # Select population type first
        population_type = self._select_population_type()

        # Create type prefix mapping
        type_prefix_map = {
            "employment_workers": "EMP",
            "local_residents": "LOC",
            "tourists": "TOU",
            "business_visitors": "BUS",
            "local_consumers": "CON",
            "others": "OTH"
        }

        type_prefix = type_prefix_map.get(population_type, "UNK")

        # Generate random letters (mix of upper and lower case)
        random_letters = ''.join(random.choices(
            string.ascii_letters, k=3  # 3 random letters
        ))

        # Format: PREFIX_XXXX_ABC (e.g., EMP_0001_AbC)
        person_id = f"{type_prefix}_{index:04d}_{random_letters}"

        return person_id, population_type

    def generate_population_memories(self, num_people: int,
                                     output_file: str = None) -> Dict[str, PersonalMemoryManager]:
        """Generate memory data for entire population"""

        population_memories = {}
        profiles = []

        for i in range(num_people):
            person_id, population_type = self._generate_person_id(i)

            profile = self.generate_person_profile(person_id, population_type)
            profiles.append(profile)

            memory_manager = self.generate_initial_memories(profile)
            memory_manager.save_memory()

            population_memories[person_id] = memory_manager

            if (i + 1) % 10 == 0:
                print(f"Generated memories for {i + 1}/{num_people} people")

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(profiles, f, indent=2, ensure_ascii=False, default=str)

        print(f"Data usage summary:")
        print(f"- Population distribution: {'Used' if self.population_data is not None else 'Random generation'}")
        print(f"- Activity time patterns: {'Used' if self.activity_time_data is not None else 'Random generation'}")
        print(f"- Activity durations: {'Used' if self.duration_data is not None else 'Random generation'}")

        # Show sample person IDs generated
        sample_ids = [profile['person_id'] for profile in profiles[:5]]
        print(f"Sample person IDs: {sample_ids}")

        return population_memories


# Usage example
if __name__ == "__main__":
    # Generate with data files
    generator_with_data = PersonalMemoryGenerator(
        data_dir="data/memory",
        use_data_files=True
    )

    # Or generate with random data (no data files required)
    generator_random = PersonalMemoryGenerator(
        data_dir="data/memory",
        use_data_files=False
    )

    # Generate population memory data
    # Person IDs will be in format: EMP_0001_AbC, LOC_0002_XyZ, etc.
    memories = generator_with_data.generate_population_memories(
        num_people=100,
        output_file="population_profiles.json"
    )