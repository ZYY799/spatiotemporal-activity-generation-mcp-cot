#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : spatial_tools.py
# @Time    : 2025/6/15 15:33
# @Desc    : Spatial Navigation Tools for location-based queries and route planning

import pandas as pd
import numpy as np
import math
import requests
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from .base_tool import BaseMCPTool, MCPMessage


class SpatialNavigationTool(BaseMCPTool):
    """Spatial navigation tool for POI search, route planning, and spatial analysis"""

    def __init__(self, memory_manager=None, amap_key: str = None):
        super().__init__("spatial_navigation", memory_manager)
        self.capabilities = [
            "poi_search",
            "route_planning",
            "distance_calculation",
            "location_recommendation",
            "accessibility_analysis",
            "spatial_clustering"
        ]

        # Amap API configuration
        self.amap_key = amap_key or self._load_amap_key()
        self.amap_base_url = "https://restapi.amap.com"

        # POI data file path
        self.poi_csv_path = "data/poi/ljz_demo_poi.csv"
        self.poi_data = None

        # Config storage file path
        self.config_file = "data/spatial_config.json"

        # Load POI data and config
        self._load_poi_data()
        self._load_config()

    def get_capabilities(self) -> List[str]:
        return self.capabilities

    def process_query(self, query: MCPMessage) -> MCPMessage:
        """Process spatial-related queries"""
        query_type = query.data.get("query_type")

        if query_type == "poi_search":
            return self._handle_poi_search(query)
        elif query_type == "route_planning":
            return self._handle_route_planning(query)
        elif query_type == "distance_calculation":
            return self._handle_distance_calculation(query)
        elif query_type == "location_recommendation":
            return self._handle_location_recommendation(query)
        elif query_type == "accessibility_analysis":
            return self._handle_accessibility_analysis(query)
        elif query_type == "spatial_clustering":
            return self._handle_spatial_clustering(query)
        else:
            return self._create_response(
                {"error": f"Unknown query type: {query_type}"},
                status="error"
            )

    def _load_poi_data(self):
        """Load POI data from CSV file"""
        try:
            if os.path.exists(self.poi_csv_path):
                self.poi_data = pd.read_csv(self.poi_csv_path)
                # Ensure correct data types
                if 'lng' in self.poi_data.columns:
                    self.poi_data['lng'] = pd.to_numeric(self.poi_data['lng'], errors='coerce')
                if 'lat' in self.poi_data.columns:
                    self.poi_data['lat'] = pd.to_numeric(self.poi_data['lat'], errors='coerce')
                if '评分' in self.poi_data.columns:
                    self.poi_data['评分'] = pd.to_numeric(self.poi_data['评分'], errors='coerce')

                # Remove records with missing coordinates
                self.poi_data = self.poi_data.dropna(subset=['lng', 'lat'])
                print(f"Loaded {len(self.poi_data)} POI records")
            else:
                print(f"POI data file not found: {self.poi_csv_path}")
                self.poi_data = pd.DataFrame()
        except Exception as e:
            print(f"Failed to load POI data: {e}")
            self.poi_data = pd.DataFrame()

    def _load_amap_key(self) -> Optional[str]:
        """Load Amap API key from config file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('amap_key')
        except:
            pass
        return None

    def _load_config(self):
        """Load configuration file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                self.config = {}
        except:
            self.config = {}

    def _save_config(self):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def _handle_poi_search(self, query: MCPMessage) -> MCPMessage:
        """Multi-field POI search"""
        if self.poi_data is None or self.poi_data.empty:
            return self._create_response(
                {"error": "POI data not loaded or empty"},
                status="error"
            )

        # Search parameters
        location = query.data.get("location")  # (lat, lon) or None
        keyword = query.data.get("keyword", "")  # Keyword search
        category = query.data.get("category", "")  # Category search (一类/二类/三类)
        area = query.data.get("area", "")  # Area search
        business_district = query.data.get("business_district", "")  # Business district search
        radius = query.data.get("radius", 5000)  # Search radius in meters
        limit = query.data.get("limit", 20)  # Result limit
        min_rating = query.data.get("min_rating", 0)  # Minimum rating
        price_range = query.data.get("price_range", [])  # Price range

        # Start filtering
        filtered_data = self.poi_data.copy()

        # Keyword search in name, tags, dishes
        if keyword:
            keyword_mask = (
                    filtered_data['店名'].str.contains(keyword, case=False, na=False) |
                    filtered_data['tags'].str.contains(keyword, case=False, na=False) |
                    filtered_data['菜品'].str.contains(keyword, case=False, na=False)
            )
            filtered_data = filtered_data[keyword_mask]

        # Category search
        if category:
            category_mask = (
                    filtered_data['一类'].str.contains(category, case=False, na=False) |
                    filtered_data['二类'].str.contains(category, case=False, na=False) |
                    filtered_data['三类'].str.contains(category, case=False, na=False)
            )
            filtered_data = filtered_data[category_mask]

        # Area search
        if area:
            filtered_data = filtered_data[
                filtered_data['area'].str.contains(area, case=False, na=False)
            ]

        # Business district search
        if business_district:
            filtered_data = filtered_data[
                filtered_data['商圈'].str.contains(business_district, case=False, na=False)
            ]

        # Rating filter
        if min_rating > 0:
            filtered_data = filtered_data[filtered_data['评分'] >= min_rating]

        # Geographic location filter
        if location and len(location) == 2:
            lat, lon = location
            # Calculate distance and filter
            filtered_data['distance'] = filtered_data.apply(
                lambda row: self._calculate_distance(
                    (lat, lon), (row['lat'], row['lng'])
                ), axis=1
            )
            filtered_data = filtered_data[filtered_data['distance'] <= radius]
            # Sort by distance
            filtered_data = filtered_data.sort_values('distance')
        else:
            # Sort by rating if no location provided
            filtered_data = filtered_data.sort_values('评分', ascending=False, na_position='last')

        # Limit results
        results = filtered_data.head(limit)

        # Format results
        poi_list = []
        for _, row in results.iterrows():
            poi_info = {
                "id": row.get('id', ''),
                "name": row.get('店名', ''),
                "category": {
                    "primary": row.get('一类', ''),
                    "secondary": row.get('二类', ''),
                    "tertiary": row.get('三类', '')
                },
                "location": {
                    "lat": row.get('lat'),
                    "lng": row.get('lng'),
                    "address": row.get('地址', '')
                },
                "rating": row.get('评分', 0),
                "price": row.get('价格', ''),
                "phone": row.get('电话', ''),
                "business_hours": row.get('营业时间', ''),
                "area": row.get('area', ''),
                "business_district": row.get('商圈', ''),
                "tags": row.get('tags', ''),
                "dishes": row.get('菜品', ''),
                "comment_count": row.get('评论数', 0),
                "distance": row.get('distance', 0) if 'distance' in row else None
            }
            poi_list.append(poi_info)

        response_data = {
            "results": poi_list,
            "total_found": len(results),
            "search_params": {
                "keyword": keyword,
                "category": category,
                "area": area,
                "business_district": business_district,
                "radius": radius,
                "location": location
            }
        }

        return self._create_response(response_data)

    def _handle_route_planning(self, query: MCPMessage) -> MCPMessage:
        """Route planning using Amap API"""
        if not self.amap_key:
            return self._create_response(
                {"error": "Amap API key not configured"},
                status="error"
            )

        origin = query.data.get("origin")  # [lng, lat] or address string
        destination = query.data.get("destination")  # [lng, lat] or address string
        route_type = query.data.get("route_type", "driving")  # driving, walking, bicycling, transit, electrobike
        strategy = query.data.get("strategy", 0)  # Route strategy
        waypoints = query.data.get("waypoints", [])  # Waypoints
        extensions = query.data.get("extensions", "all")  # Result detail level

        # Format coordinates
        origin_str = self._format_location(origin)
        destination_str = self._format_location(destination)

        if not origin_str or not destination_str:
            return self._create_response(
                {"error": "Invalid origin or destination format"},
                status="error"
            )

        # Route type to API version mapping
        route_apis = {
            "driving": ("v3", "/direction/driving"),
            "walking": ("v3", "/direction/walking"),
            "bicycling": ("v4", "/direction/bicycling"),
            "transit": ("v3", "/direction/transit/integrated"),
            "electrobike": ("v5", "/direction/electrobike")
        }

        if route_type not in route_apis:
            return self._create_response(
                {"error": f"Unsupported route type: {route_type}"},
                status="error"
            )

        version, endpoint = route_apis[route_type]

        # Build request parameters
        params = {
            "key": self.amap_key,
            "origin": origin_str,
            "destination": destination_str,
            "extensions": extensions,
            "output": "json"
        }

        # Add route-specific parameters
        if route_type == "driving":
            params["strategy"] = strategy
        elif route_type == "transit":
            # Transit requires city parameter
            city = query.data.get("city", "010")  # Default to Beijing
            params["city"] = city
            params["strategy"] = strategy

        # Add waypoints if provided
        if waypoints and route_type in ["driving"]:
            waypoints_str = "|".join([self._format_location(wp) for wp in waypoints])
            params["waypoints"] = waypoints_str

        try:
            # Call Amap route planning API
            url = f"{self.amap_base_url}/{version}{endpoint}"
            response = requests.get(url, params=params, timeout=10)
            result = response.json()

            if result.get("status") == "1" and result.get("route"):
                route_data = result["route"]
                paths = route_data.get("paths", [])

                if paths:
                    best_path = paths[0]  # Take first route

                    # Parse route information
                    route_info = {
                        "distance": int(best_path.get("distance", 0)),  # meters
                        "duration": int(best_path.get("duration", 0)),  # seconds
                        "route_type": route_type,
                        "strategy": strategy,
                        "tolls": int(best_path.get("tolls", 0)),  # toll fees in yuan
                        "toll_distance": int(best_path.get("toll_distance", 0)),  # toll road distance
                        "steps": [],
                        "polyline": "",
                        "origin": origin,
                        "destination": destination
                    }

                    # Parse step information
                    for step in best_path.get("steps", []):
                        step_info = {
                            "instruction": step.get("instruction", ""),
                            "road": step.get("road", ""),
                            "distance": int(step.get("distance", 0)),
                            "duration": int(step.get("duration", 0)),
                            "polyline": step.get("polyline", "")
                        }
                        route_info["steps"].append(step_info)

                    # Get route polyline
                    if best_path.get("polyline"):
                        route_info["polyline"] = best_path["polyline"]

                    # Save route to config
                    self._save_route_to_config(route_info)

                    response_data = {
                        "route": route_info,
                        "alternative_routes": len(paths),
                        "status": "success"
                    }

                    return self._create_response(response_data)
                else:
                    return self._create_response(
                        {"error": "No valid route found"},
                        status="error"
                    )
            else:
                error_msg = result.get("info", "Route planning failed")
                return self._create_response(
                    {"error": f"Amap API error: {error_msg}"},
                    status="error"
                )

        except Exception as e:
            return self._create_response(
                {"error": f"Route planning request failed: {str(e)}"},
                status="error"
            )

    def _format_location(self, location: Union[List, str]) -> Optional[str]:
        """Format location for Amap API"""
        if isinstance(location, list) and len(location) == 2:
            # [lng, lat] format
            return f"{location[0]},{location[1]}"
        elif isinstance(location, str):
            # Address string
            return location
        else:
            return None

    def _save_route_to_config(self, route_info: Dict):
        """Save route info to config file"""
        if "recent_routes" not in self.config:
            self.config["recent_routes"] = []

        # Add timestamp
        import time
        route_info["timestamp"] = time.time()

        # Keep latest 10 routes
        self.config["recent_routes"].insert(0, route_info)
        self.config["recent_routes"] = self.config["recent_routes"][:10]

        self._save_config()

    def _handle_distance_calculation(self, query: MCPMessage) -> MCPMessage:
        """Calculate distances between locations"""
        locations = query.data.get("locations", [])

        if len(locations) < 2:
            return self._create_response(
                {"error": "At least 2 locations required"},
                status="error"
            )

        # Calculate pairwise distances
        distances = {}
        for i, loc1 in enumerate(locations):
            for j, loc2 in enumerate(locations[i + 1:], i + 1):
                if len(loc1) >= 2 and len(loc2) >= 2:
                    distance = self._calculate_distance(
                        (loc1[1], loc1[0]), (loc2[1], loc2[0])  # (lat, lng)
                    )
                    distances[f"point_{i}_to_{j}"] = {
                        "distance": distance,
                        "from": loc1,
                        "to": loc2
                    }

        # Calculate centroid
        if locations:
            centroid = self._calculate_centroid([
                (loc[1], loc[0]) for loc in locations if len(loc) >= 2
            ])
        else:
            centroid = None

        response_data = {
            "pairwise_distances": distances,
            "centroid": centroid,
            "total_locations": len(locations)
        }

        return self._create_response(response_data)

    def _handle_location_recommendation(self, query: MCPMessage) -> MCPMessage:
        """Location recommendation based on preferences and history"""
        if self.poi_data is None or self.poi_data.empty:
            return self._create_response(
                {"error": "POI data not loaded"},
                status="error"
            )

        activity_type = query.data.get("activity_type", "")
        current_location = query.data.get("current_location")  # [lng, lat]
        user_preferences = query.data.get("preferences", {})
        limit = query.data.get("limit", 10)

        # Get personal preferences
        exploration_tendency = self._get_personal_preference(
            "exploration_tendency", default_value=0.5
        )

        # Filter POI by activity type
        filtered_data = self.poi_data.copy()

        if activity_type:
            # Simple activity type mapping
            activity_mapping = {
                "dining": ["美食"],
                "shopping": ["购物"],
                "entertainment": ["休闲娱乐"],
                "beauty": ["丽人"],
                "healthcare": ["医疗健康"]
            }

            target_categories = activity_mapping.get(activity_type, [activity_type])
            category_mask = False
            for category in target_categories:
                category_mask |= filtered_data['一类'].str.contains(category, case=False, na=False)

            if category_mask.any():
                filtered_data = filtered_data[category_mask]

        # Calculate distance if current location provided
        if current_location and len(current_location) == 2:
            lng, lat = current_location
            filtered_data['distance'] = filtered_data.apply(
                lambda row: self._calculate_distance(
                    (lat, lng), (row['lat'], row['lng'])
                ), axis=1
            )

            # Distance preference score (closer = higher score)
            max_distance = filtered_data['distance'].max() if not filtered_data.empty else 1
            filtered_data['distance_score'] = 1 - (filtered_data['distance'] / max_distance)
        else:
            filtered_data['distance_score'] = 0.5

        # Normalize rating score
        if '评分' in filtered_data.columns:
            max_rating = filtered_data['评分'].max()
            if max_rating > 0:
                filtered_data['rating_score'] = filtered_data['评分'] / max_rating
            else:
                filtered_data['rating_score'] = 0.5
        else:
            filtered_data['rating_score'] = 0.5

        # Calculate composite score
        filtered_data['total_score'] = (
                filtered_data['distance_score'] * 0.4 +
                filtered_data['rating_score'] * 0.6
        )

        # Sort and return results
        recommendations = filtered_data.sort_values('total_score', ascending=False).head(limit)

        results = []
        for _, row in recommendations.iterrows():
            result = {
                "poi": {
                    "id": row.get('id', ''),
                    "name": row.get('店名', ''),
                    "category": row.get('一类', ''),
                    "rating": row.get('评分', 0),
                    "location": {
                        "lat": row.get('lat'),
                        "lng": row.get('lng'),
                        "address": row.get('地址', '')
                    }
                },
                "score": row.get('total_score', 0),
                "distance": row.get('distance', 0),
                "recommendation_reason": f"Rating: {row.get('评分', 0)}, Distance: {row.get('distance', 0):.0f}m"
            }
            results.append(result)

        response_data = {
            "recommendations": results,
            "activity_type": activity_type,
            "total_candidates": len(filtered_data)
        }

        return self._create_response(response_data)

    def _handle_accessibility_analysis(self, query: MCPMessage) -> MCPMessage:
        """Analyze location accessibility"""
        location = query.data.get("location")  # [lng, lat]
        transport_modes = query.data.get("transport_modes", ["walking", "public_transit"])
        max_time = query.data.get("max_time", 30)  # max travel time in minutes

        if not location or len(location) != 2:
            return self._create_response(
                {"error": "Valid location coordinates required"},
                status="error"
            )

        # Simple accessibility analysis (based on straight-line distance estimation)
        analysis_result = {}

        for mode in transport_modes:
            # Average speeds for different transport modes (km/h)
            mode_speeds = {
                "walking": 5,
                "cycling": 15,
                "public_transit": 20,
                "driving": 30
            }

            speed = mode_speeds.get(mode, 5)
            max_distance = (speed * 1000) * (max_time / 60)  # meters

            # Find accessible POIs
            if self.poi_data is not None and not self.poi_data.empty:
                poi_copy = self.poi_data.copy()
                lng, lat = location

                poi_copy['distance'] = poi_copy.apply(
                    lambda row: self._calculate_distance(
                        (lat, lng), (row['lat'], row['lng'])
                    ), axis=1
                )

                accessible_pois = poi_copy[poi_copy['distance'] <= max_distance]

                analysis_result[mode] = {
                    "max_distance": max_distance,
                    "accessible_count": len(accessible_pois),
                    "coverage_categories": accessible_pois['一类'].value_counts().to_dict(),
                    "sample_locations": accessible_pois.head(5)[['店名', '一类', 'distance']].to_dict('records')
                }

        response_data = {
            "accessibility_analysis": analysis_result,
            "analysis_center": location,
            "max_travel_time": max_time
        }

        return self._create_response(response_data)

    def _handle_spatial_clustering(self, query: MCPMessage) -> MCPMessage:
        """Spatial clustering analysis"""
        locations = query.data.get("locations", [])
        cluster_radius = query.data.get("cluster_radius", 500)  # clustering radius in meters

        if len(locations) < 3:
            return self._create_response(
                {"error": "At least 3 locations required for clustering"},
                status="error"
            )

        # Simple distance-based clustering
        clusters = []
        used_indices = set()

        for i, loc1 in enumerate(locations):
            if i in used_indices or len(loc1) < 2:
                continue

            cluster = {
                "id": len(clusters),
                "center": loc1,
                "locations": [{"index": i, "location": loc1}],
                "size": 1
            }
            used_indices.add(i)

            # Find nearby points
            for j, loc2 in enumerate(locations):
                if j in used_indices or len(loc2) < 2:
                    continue

                distance = self._calculate_distance(
                    (loc1[1], loc1[0]), (loc2[1], loc2[0])  # (lat, lng)
                )

                if distance <= cluster_radius:
                    cluster["locations"].append({"index": j, "location": loc2})
                    cluster["size"] += 1
                    used_indices.add(j)

            clusters.append(cluster)

        response_data = {
            "clusters": clusters,
            "total_clusters": len(clusters),
            "cluster_radius": cluster_radius,
            "clustered_points": len(used_indices),
            "total_points": len(locations)
        }

        return self._create_response(response_data)

    def _calculate_distance(self, loc1: Tuple[float, float],
                            loc2: Tuple[float, float]) -> float:
        """Calculate distance between two points using Haversine formula (meters)"""
        lat1, lon1 = loc1
        lat2, lon2 = loc2

        R = 6371000  # Earth radius in meters
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _calculate_centroid(self, locations: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate geographic centroid"""
        if not locations:
            return (0, 0)

        lat_sum = sum(loc[0] for loc in locations)
        lon_sum = sum(loc[1] for loc in locations)

        return (lat_sum / len(locations), lon_sum / len(locations))

    def set_amap_key(self, api_key: str):
        """Set Amap API key"""
        self.amap_key = api_key
        self.config["amap_key"] = api_key
        self._save_config()

    def reload_poi_data(self):
        """Reload POI data"""
        self._load_poi_data()

    def get_recent_routes(self) -> List[Dict]:
        """Get recent route records"""
        return self.config.get("recent_routes", [])