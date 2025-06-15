#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : data_utils.py
# @Time    : 2025/6/15 15:38
# @Desc    :


import json
import pickle
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
import os
from pathlib import Path


class DataUtils:
    """Utility functions for data processing and manipulation"""

    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """Load JSON file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")

    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
        """Save data to JSON file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)

    @staticmethod
    def load_pickle(file_path: str) -> Any:
        """Load pickle file with error handling"""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Pickle file not found: {file_path}")
        except pickle.UnpicklingError as e:
            raise ValueError(f"Invalid pickle format in {file_path}: {e}")

    @staticmethod
    def save_pickle(data: Any, file_path: str) -> None:
        """Save data to pickle file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def normalize_coordinates(lat: float, lon: float) -> Tuple[float, float]:
        """Normalize coordinates to valid ranges"""
        # Clamp latitude to [-90, 90]
        lat = max(-90.0, min(90.0, lat))

        # Normalize longitude to [-180, 180]
        while lon > 180:
            lon -= 360
        while lon < -180:
            lon += 360

        return (lat, lon)

    @staticmethod
    def validate_trajectory_data(trajectory_data: Dict[str, Any]) -> bool:
        """Validate trajectory data structure"""
        required_fields = ['person_id', 'date', 'activities']

        # Check required fields
        for field in required_fields:
            if field not in trajectory_data:
                return False

        # Validate activities structure
        activities = trajectory_data.get('activities', [])
        if not isinstance(activities, list):
            return False

        for activity in activities:
            if not isinstance(activity, dict):
                return False

            activity_required = ['activity_type', 'start_time', 'duration', 'location']
            for field in activity_required:
                if field not in activity:
                    return False

        return True

    @staticmethod
    def clean_activity_data(activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize activity data"""
        cleaned = activity_data.copy()

        # Standardize activity type
        if 'activity_type' in cleaned:
            cleaned['activity_type'] = str(cleaned['activity_type']).lower().strip()

        # Ensure numeric fields are properly typed
        numeric_fields = ['duration', 'satisfaction_prediction', 'confidence']
        for field in numeric_fields:
            if field in cleaned and cleaned[field] is not None:
                try:
                    cleaned[field] = float(cleaned[field])
                except (ValueError, TypeError):
                    cleaned[field] = 0.0

        # Clean location data
        if 'location' in cleaned and isinstance(cleaned['location'], (list, tuple)) and len(cleaned['location']) >= 2:
            lat, lon = cleaned['location'][:2]
            cleaned['location'] = DataUtils.normalize_coordinates(float(lat), float(lon))

        # Ensure companions is a list
        if 'companions' in cleaned:
            if not isinstance(cleaned['companions'], list):
                cleaned['companions'] = []

        return cleaned

    @staticmethod
    def aggregate_activity_statistics(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate statistics from trajectory data"""
        if not trajectories:
            return {}

        all_activities = []
        for trajectory in trajectories:
            all_activities.extend(trajectory.get('activities', []))

        if not all_activities:
            return {}

        # Activity type distribution
        activity_types = [a.get('activity_type') for a in all_activities if a.get('activity_type')]
        activity_distribution = pd.Series(activity_types).value_counts().to_dict()

        # Duration statistics
        durations = [a.get('duration', 0) for a in all_activities if a.get('duration')]
        duration_stats = {
            'mean': np.mean(durations) if durations else 0,
            'median': np.median(durations) if durations else 0,
            'std': np.std(durations) if durations else 0,
            'min': np.min(durations) if durations else 0,
            'max': np.max(durations) if durations else 0
        }

        # Temporal distribution
        start_hours = []
        for activity in all_activities:
            start_time = activity.get('start_time')
            if start_time:
                try:
                    if isinstance(start_time, str):
                        from utils.time_utils import TimeUtils
                        dt = TimeUtils.parse_time_string(start_time)
                        start_hours.append(dt.hour)
                    elif hasattr(start_time, 'hour'):
                        start_hours.append(start_time.hour)
                except:
                    continue

        temporal_distribution = pd.Series(start_hours).value_counts().sort_index().to_dict()

        return {
            'total_activities': len(all_activities),
            'total_trajectories': len(trajectories),
            'activity_distribution': activity_distribution,
            'duration_statistics': duration_stats,
            'temporal_distribution': temporal_distribution,
            'average_activities_per_day': len(all_activities) / len(trajectories) if trajectories else 0
        }

    @staticmethod
    def export_trajectories_csv(trajectories: List[Dict[str, Any]], output_path: str) -> None:
        """Export trajectories to CSV format"""
        records = []

        for trajectory in trajectories:
            person_id = trajectory.get('person_id', 'unknown')
            date = trajectory.get('date', 'unknown')

            for i, activity in enumerate(trajectory.get('activities', [])):
                record = {
                    'person_id': person_id,
                    'date': date,
                    'activity_sequence': i + 1,
                    'activity_type': activity.get('activity_type', ''),
                    'start_time': activity.get('start_time', ''),
                    'duration': activity.get('duration', 0),
                    'location_lat': activity.get('location', [0, 0])[0] if activity.get('location') else 0,
                    'location_lon': activity.get('location', [0, 0])[1] if activity.get('location') else 0,
                    'location_name': activity.get('location_name', ''),
                    'companions': ','.join(activity.get('companions', [])),
                    'transportation_mode': activity.get('transportation_mode', ''),
                    'satisfaction_prediction': activity.get('satisfaction_prediction', 0),
                    'confidence': activity.get('confidence', 0)
                }
                records.append(record)

        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')

    @staticmethod
    def create_sample_config() -> Dict[str, Any]:
        """Create sample configuration for testing"""
        return {
            "model_settings": {
                "temperature": 0.7,
                "max_tokens": 9216
            },
            "generation": {
                "max_activities_per_day": 6,
                "min_activity_duration": 15,
                "max_activity_duration": 240,
                "daily_time_window": [6, 23]
            },
            "parallel_processing": {
                "max_workers": 4,
                "batch_size": 5,
                "memory_limit_gb": 4
            },
            "evaluation": {
                "quality_threshold": 0.6,
                "diversity_threshold": 0.5
            }
        }
