#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : geo_utils.py
# @Time    : 2025/6/15 15:38
# @Desc    :


# utils/geo_utils.py
"""
Geographic utilities for spatiotemporal processing
"""

import math
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from geopy.distance import geodesic
import json


class GeoUtils:
    """Utility functions for geographic calculations"""

    @staticmethod
    def calculate_distance(coord1: Tuple[float, float],
                           coord2: Tuple[float, float]) -> float:
        """
        Calculate distance between two coordinates using Haversine formula
        Returns distance in meters
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2

        R = 6371000  # Earth's radius in meters

        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        # Haversine formula
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c
        return distance

    @staticmethod
    def calculate_bearing(coord1: Tuple[float, float],
                          coord2: Tuple[float, float]) -> float:
        """
        Calculate bearing from coord1 to coord2
        Returns bearing in degrees (0-360)
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)

        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))

        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)

        # Normalize to 0-360
        bearing_deg = (bearing_deg + 360) % 360

        return bearing_deg

    @staticmethod
    def point_in_polygon(point: Tuple[float, float],
                         polygon: List[Tuple[float, float]]) -> bool:
        """
        Check if point is inside polygon using ray casting algorithm
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    @staticmethod
    def generate_random_point_in_bounds(bounds: Dict[str, float]) -> Tuple[float, float]:
        """
        Generate random point within geographic bounds
        """
        lat = np.random.uniform(bounds['lat_min'], bounds['lat_max'])
        lon = np.random.uniform(bounds['lon_min'], bounds['lon_max'])
        return (lat, lon)

    @staticmethod
    def calculate_centroid(coordinates: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Calculate geographic centroid of coordinate list
        """
        if not coordinates:
            return (0.0, 0.0)

        lat_sum = sum(coord[0] for coord in coordinates)
        lon_sum = sum(coord[1] for coord in coordinates)

        return (lat_sum / len(coordinates), lon_sum / len(coordinates))

    @staticmethod
    def create_bounding_box(center: Tuple[float, float],
                            radius_meters: float) -> Dict[str, float]:
        """
        Create bounding box around center point with given radius
        """
        lat, lon = center

        # Rough conversion: 1 degree latitude ≈ 111,000 meters
        # 1 degree longitude ≈ 111,000 * cos(latitude) meters
        lat_offset = radius_meters / 111000
        lon_offset = radius_meters / (111000 * math.cos(math.radians(lat)))

        return {
            'lat_min': lat - lat_offset,
            'lat_max': lat + lat_offset,
            'lon_min': lon - lon_offset,
            'lon_max': lon + lon_offset
        }

    @staticmethod
    def cluster_points_by_distance(points: List[Tuple[float, float]],
                                   max_distance: float) -> List[List[Tuple[float, float]]]:
        """
        Cluster points by maximum distance threshold
        """
        if not points:
            return []

        clusters = []
        unassigned = points.copy()

        while unassigned:
            # Start new cluster with first unassigned point
            current_cluster = [unassigned.pop(0)]

            # Find all points within distance threshold
            i = 0
            while i < len(unassigned):
                point = unassigned[i]

                # Check distance to any point in current cluster
                min_distance = min(
                    GeoUtils.calculate_distance(point, cluster_point)
                    for cluster_point in current_cluster
                )

                if min_distance <= max_distance:
                    current_cluster.append(unassigned.pop(i))
                else:
                    i += 1

            clusters.append(current_cluster)

        return clusters