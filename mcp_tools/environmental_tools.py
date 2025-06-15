#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : environmental_tools.py
# @Time    : 2025/6/15 15:33
# @Desc    : Environmental Perception Tools for environmental condition queries


from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from .base_tool import BaseMCPTool, MCPMessage
import random
import math


class EnvironmentalPerceptionTool(BaseMCPTool):
    """
    Environmental perception tool for weather, events, and environmental conditions.

    SIMPLIFIED LOCAL IMPLEMENTATION:
    - Uses local simulation/computation for rapid prototyping and testing
    - All _simulate_* methods generate mock data locally
    - Ready for API integration: replace simulation methods with real API calls
    - Placeholder methods provide basic functionality, can be enhanced later

    Future API Integration Points:
    - Weather: OpenWeatherMap, AccuWeather API
    - Events: Eventbrite, local event APIs
    - Crowd: Google Places, traffic APIs
    - Air Quality: EPA, PurpleAir APIs
    """

    def __init__(self, memory_manager=None):
        super().__init__("environmental_perception", memory_manager)
        self.capabilities = [
            "weather_query",
            "event_detection",
            "crowd_density_estimation",
            "environmental_suitability",
            "seasonal_analysis",
            "air_quality_check"
        ]

    def get_capabilities(self) -> List[str]:
        return self.capabilities

    def process_query(self, query: MCPMessage) -> MCPMessage:
        """Route queries to appropriate handlers"""
        query_type = query.data.get("query_type")

        handlers = {
            "weather_query": self._handle_weather_query,
            "event_detection": self._handle_event_detection,
            "crowd_density": self._handle_crowd_density,
            "suitability_check": self._handle_suitability_check,
            "seasonal_analysis": self._handle_seasonal_analysis,
            "air_quality": self._handle_air_quality
        }

        handler = handlers.get(query_type)
        if handler:
            return handler(query)
        else:
            return self._create_response(
                {"error": f"Unknown query type: {query_type}"},
                status="error"
            )

    def _handle_weather_query(self, query: MCPMessage) -> MCPMessage:
        """Process weather queries with personal impact assessment"""
        location = query.data.get("location")
        timestamp = query.data.get("timestamp", datetime.now().isoformat())

        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        # Get user preferences from memory
        weather_sensitivity = self._get_personal_preference("weather_sensitivity", 0.5)
        rain_tolerance = self._get_personal_preference("rain_tolerance", 0.3)
        temperature_preference = self._get_personal_preference("temperature_preference", 22)

        # Generate weather data (replace with API call)
        weather_data = self._simulate_weather(timestamp)

        # Calculate personal impact based on preferences
        personal_impact = self._calculate_weather_impact(
            weather_data, weather_sensitivity, rain_tolerance, temperature_preference
        )

        response_data = {
            "weather": weather_data,
            "personal_impact": personal_impact,
            "activity_recommendations": self._generate_weather_activity_recommendations(
                weather_data, personal_impact
            ),
            "timestamp": timestamp.isoformat(),
            "location": location
        }

        return self._create_response(response_data)

    def _handle_event_detection(self, query: MCPMessage) -> MCPMessage:
        """Detect local events and assess personal impact"""
        location = query.data.get("location")
        date = query.data.get("date", datetime.now().isoformat())

        if isinstance(date, str):
            date = datetime.fromisoformat(date)

        # Simulate event detection (replace with API call)
        events = self._detect_local_events(location, date)

        # Get user event preferences
        event_tolerance = self._get_personal_preference("crowd_tolerance", 0.5)
        event_interest = self._get_personal_preference("event_interest", 0.5)

        # Assess impact on user activities
        event_impacts = []
        for event in events:
            impact = self._calculate_event_impact(event, event_tolerance, event_interest)
            event_impacts.append({
                "event": event,
                "personal_impact": impact,
                "recommendation": self._generate_event_recommendation(event, impact)
            })

        response_data = {
            "events": event_impacts,
            "total_events": len(events),
            "date": date.isoformat(),
            "location": location
        }

        return self._create_response(response_data)

    def _handle_crowd_density(self, query: MCPMessage) -> MCPMessage:
        """Estimate crowd density and personal comfort level"""
        location = query.data.get("location")
        timestamp = query.data.get("timestamp", datetime.now().isoformat())

        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        # Get user crowd preferences
        crowd_tolerance = self._get_personal_preference("crowd_tolerance", 0.5)
        preferred_crowd_level = self._get_personal_preference("preferred_crowd_level", "moderate")

        # Estimate crowd density (replace with API call)
        crowd_data = self._estimate_crowd_density(location, timestamp)

        # Calculate comfort level
        comfort_level = self._calculate_crowd_comfort(crowd_data, crowd_tolerance, preferred_crowd_level)

        response_data = {
            "crowd_density": crowd_data,
            "personal_comfort": comfort_level,
            "recommendation": self._generate_crowd_recommendation(crowd_data, comfort_level),
            "timestamp": timestamp.isoformat(),
            "location": location
        }

        return self._create_response(response_data)

    def _handle_suitability_check(self, query: MCPMessage) -> MCPMessage:
        """Check overall environmental suitability for planned activity"""
        activity_type = query.data.get("activity_type")
        location = query.data.get("location")
        timestamp = query.data.get("timestamp", datetime.now().isoformat())

        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        # Gather environmental data
        weather = self._simulate_weather(timestamp)
        crowd = self._estimate_crowd_density(location, timestamp)
        events = self._detect_local_events(location, timestamp)

        # Get activity-specific preferences
        activity_weather_sensitivity = self._get_personal_preference(
            f"{activity_type}_weather_sensitivity", 0.5
        )
        activity_crowd_preference = self._get_personal_preference(
            f"{activity_type}_crowd_preference", "moderate"
        )

        # Calculate suitability scores
        weather_suitability = self._calculate_activity_weather_suitability(
            activity_type, weather, activity_weather_sensitivity
        )

        crowd_suitability = self._calculate_activity_crowd_suitability(
            activity_type, crowd, activity_crowd_preference
        )

        event_impact = sum(
            self._calculate_event_activity_impact(event, activity_type)
            for event in events
        ) / max(1, len(events))

        # Weighted overall score
        overall_suitability = (
                weather_suitability * 0.4 +
                crowd_suitability * 0.3 +
                (1 - abs(event_impact)) * 0.3
        )

        response_data = {
            "overall_suitability": overall_suitability,
            "weather_suitability": weather_suitability,
            "crowd_suitability": crowd_suitability,
            "event_impact": event_impact,
            "recommendation": "proceed" if overall_suitability > 0.6 else "consider_alternative",
            "environmental_conditions": {
                "weather": weather,
                "crowd": crowd,
                "events": events
            }
        }

        return self._create_response(response_data)

    def _handle_seasonal_analysis(self, query: MCPMessage) -> MCPMessage:
        """Analyze seasonal patterns and user preferences"""
        activity_type = query.data.get("activity_type")
        date = query.data.get("date", datetime.now().isoformat())

        if isinstance(date, str):
            date = datetime.fromisoformat(date)

        season = self._get_season(date)

        # Get seasonal preferences
        seasonal_preferences = {}
        for s in ["spring", "summer", "autumn", "winter"]:
            seasonal_preferences[s] = self._get_personal_preference(
                f"{s}_activity_preference", 0.5
            )

        seasonal_activity_preferences = {}
        for s in ["spring", "summer", "autumn", "winter"]:
            seasonal_activity_preferences[s] = self._get_personal_preference(
                f"{activity_type}_{s}_preference", 0.5
            )

        current_seasonal_preference = seasonal_preferences.get(season, 0.5)
        current_activity_seasonal_preference = seasonal_activity_preferences.get(season, 0.5)

        response_data = {
            "current_season": season,
            "seasonal_preference": current_seasonal_preference,
            "activity_seasonal_preference": current_activity_seasonal_preference,
            "seasonal_recommendations": self._generate_seasonal_recommendations(
                season, activity_type, seasonal_preferences
            ),
            "optimal_seasons": sorted(
                seasonal_activity_preferences.items(),
                key=lambda x: x[1], reverse=True
            )[:2]
        }

        return self._create_response(response_data)

    def _handle_air_quality(self, query: MCPMessage) -> MCPMessage:
        """Check air quality and impact on outdoor activities"""
        location = query.data.get("location")
        timestamp = query.data.get("timestamp", datetime.now().isoformat())

        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        # Simulate air quality data (replace with API call)
        air_quality = self._simulate_air_quality(timestamp)

        # Get user sensitivity
        air_sensitivity = self._get_personal_preference("air_quality_sensitivity", 0.5)
        outdoor_activity_preference = self._get_personal_preference("outdoor_activity_preference", 0.7)

        # Calculate impact
        impact = self._calculate_air_quality_impact(air_quality, air_sensitivity)

        response_data = {
            "air_quality": air_quality,
            "personal_impact": impact,
            "outdoor_activity_recommendation": self._generate_air_quality_recommendation(
                air_quality, impact, outdoor_activity_preference
            ),
            "timestamp": timestamp.isoformat(),
            "location": location
        }

        return self._create_response(response_data)

    # === Simulation Methods (Replace with API calls) ===

    def _simulate_weather(self, timestamp: datetime) -> Dict[str, Any]:
        """Simulate weather data - replace with weather API"""
        month = timestamp.month
        hour = timestamp.hour

        # Seasonal temperature base
        base_temp = {1: 5, 2: 8, 3: 12, 4: 18, 5: 23, 6: 28,
                     7: 31, 8: 30, 9: 26, 10: 20, 11: 14, 12: 8}[month]

        # Daily temperature variation
        daily_variation = 6 * math.sin((hour - 6) * math.pi / 12)
        temperature = base_temp + daily_variation + random.gauss(0, 2)

        # Weather conditions with seasonal weights
        conditions = ["sunny", "partly_cloudy", "cloudy", "rainy", "foggy"]
        weather_weights = [0.4, 0.3, 0.15, 0.1, 0.05]

        if 6 <= month <= 8:  # Summer
            weather_weights = [0.6, 0.25, 0.1, 0.04, 0.01]
        elif month in [12, 1, 2]:  # Winter
            weather_weights = [0.2, 0.3, 0.3, 0.15, 0.05]

        condition = random.choices(conditions, weights=weather_weights)[0]

        precipitation = 0
        if condition == "rainy":
            precipitation = random.uniform(0.5, 10)

        return {
            "temperature": round(temperature, 1),
            "condition": condition,
            "precipitation": precipitation,
            "humidity": random.randint(40, 90),
            "wind_speed": random.uniform(0, 15),
            "visibility": 10 if condition != "foggy" else random.uniform(1, 5)
        }

    def _detect_local_events(self, location: str, date: datetime) -> List[Dict[str, Any]]:
        """Simulate event detection - replace with events API"""
        events = []

        if random.random() < 0.3:  # 30% chance of events
            event_types = ["concert", "festival", "exhibition", "sports", "conference"]
            num_events = random.randint(1, 3)

            for _ in range(num_events):
                event = {
                    "type": random.choice(event_types),
                    "name": f"Local {random.choice(event_types).title()} Event",
                    "start_time": date + timedelta(hours=random.randint(-2, 8)),
                    "duration": random.randint(60, 300),
                    "expected_attendance": random.randint(100, 5000),
                    "impact_radius": random.randint(200, 1000)
                }
                events.append(event)

        return events

    def _estimate_crowd_density(self, location: str, timestamp: datetime) -> Dict[str, Any]:
        """Simulate crowd density - replace with crowd analytics API"""
        hour = timestamp.hour
        day_type = "weekend" if timestamp.weekday() >= 5 else "weekday"

        # Time-based crowd patterns
        if day_type == "weekday":
            if 8 <= hour <= 9 or 17 <= hour <= 19:
                base_density = "high"
            elif 12 <= hour <= 14:
                base_density = "medium"
            else:
                base_density = "low"
        else:  # weekend
            if 11 <= hour <= 16:
                base_density = "medium"
            elif 19 <= hour <= 22:
                base_density = "medium"
            else:
                base_density = "low"

        density_levels = {"low": 0.2, "medium": 0.5, "high": 0.8}

        return {
            "density_level": base_density,
            "density_score": density_levels[base_density],
            "estimated_people_per_100m2": density_levels[base_density] * 50,
            "peak_hours": ["8-9", "12-14", "17-19"] if day_type == "weekday" else ["11-16", "19-22"]
        }

    def _simulate_air_quality(self, timestamp: datetime) -> Dict[str, Any]:
        """Simulate air quality data - replace with air quality API"""
        base_aqi = random.randint(50, 150)

        # Time-based variations
        hour = timestamp.hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            base_aqi += random.randint(10, 30)

        pollution_level = "good" if base_aqi <= 50 else "moderate" if base_aqi <= 100 else "unhealthy"

        return {
            "aqi": base_aqi,
            "pollution_level": pollution_level,
            "pm25": base_aqi * 0.5,
            "pm10": base_aqi * 0.8,
            "recommendation": "safe" if base_aqi <= 100 else "limit_outdoor_activities"
        }

    # === Helper Methods ===

    def _calculate_weather_impact(self, weather: Dict[str, Any],
                                  sensitivity: float, rain_tolerance: float,
                                  temp_preference: float) -> Dict[str, Any]:
        """Calculate personalized weather impact"""
        temp_impact = 1 - abs(weather["temperature"] - temp_preference) / 20
        temp_impact = max(0, temp_impact)

        condition_impact = {
            "sunny": 1.0,
            "partly_cloudy": 0.8,
            "cloudy": 0.6,
            "rainy": rain_tolerance,
            "foggy": 0.4
        }.get(weather["condition"], 0.5)

        overall_impact = (temp_impact * 0.6 + condition_impact * 0.4) * (2 - sensitivity)

        return {
            "temperature_impact": temp_impact,
            "condition_impact": condition_impact,
            "overall_impact": min(1.0, overall_impact),
            "activity_recommendation": "favorable" if overall_impact > 0.7 else "unfavorable"
        }

    def _get_season(self, date: datetime) -> str:
        """Determine season from date"""
        month = date.month
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "autumn"
        else:
            return "winter"

    def _calculate_air_quality_impact(self, air_quality: Dict[str, Any],
                                      sensitivity: float) -> Dict[str, Any]:
        """Calculate air quality impact on user"""
        aqi = air_quality["aqi"]

        # Impact based on AQI levels
        if aqi <= 50:
            base_impact = 0.1
        elif aqi <= 100:
            base_impact = 0.3
        elif aqi <= 150:
            base_impact = 0.6
        else:
            base_impact = 0.9

        personal_impact = base_impact * sensitivity

        return {
            "health_impact": personal_impact,
            "outdoor_suitability": 1 - personal_impact,
            "recommendation": "proceed" if personal_impact < 0.5 else "stay_indoors"
        }

    # === Simplified Implementation Methods (Replace with detailed logic) ===

    def _generate_weather_activity_recommendations(self, weather, impact):
        """Generate basic weather-based activity suggestions"""
        recommendations = []
        if weather["condition"] == "rainy":
            recommendations.append("Consider indoor activities")
        elif weather["temperature"] < 5:
            recommendations.append("Dress warmly for outdoor activities")
        elif weather["temperature"] > 30:
            recommendations.append("Stay hydrated and seek shade")
        else:
            recommendations.append("Good conditions for outdoor activities")
        return recommendations

    def _calculate_event_impact(self, event, tolerance, interest):
        """Simplified event impact calculation"""
        # Higher attendance = higher impact for low tolerance users
        attendance_factor = min(event["expected_attendance"] / 1000, 1.0)
        impact = attendance_factor * (1 - tolerance) + interest * 0.3
        return min(1.0, impact)

    def _generate_event_recommendation(self, event, impact):
        """Generate basic event-related recommendations"""
        if impact > 0.7:
            return "High impact event - plan accordingly"
        elif impact > 0.4:
            return "Moderate event activity - consider timing"
        else:
            return "Minimal event impact on plans"

    def _calculate_crowd_comfort(self, crowd_data, tolerance, preferred_level):
        """Simplified crowd comfort calculation"""
        density_score = crowd_data["density_score"]
        level_map = {"low": 0.2, "moderate": 0.5, "high": 0.8}
        preferred_score = level_map.get(preferred_level, 0.5)

        # Comfort decreases with distance from preferred level
        comfort = 1 - abs(density_score - preferred_score) * (1 - tolerance)
        return max(0, comfort)

    def _generate_crowd_recommendation(self, crowd_data, comfort):
        """Generate crowd-based recommendations"""
        if comfort > 0.7:
            return "Comfortable crowd levels"
        elif comfort > 0.4:
            return "Moderate crowds - plan for delays"
        else:
            return "High crowds - consider alternative times"

    def _calculate_activity_weather_suitability(self, activity, weather, sensitivity):
        """Simplified activity-weather suitability"""
        base_suitability = 0.8

        # Adjust for weather conditions
        if weather["condition"] == "rainy":
            base_suitability *= 0.3 if "outdoor" in activity.lower() else 0.9
        elif weather["condition"] == "sunny":
            base_suitability *= 1.0
        elif weather["condition"] == "cloudy":
            base_suitability *= 0.8

        # Temperature impact
        temp = weather["temperature"]
        if temp < 0 or temp > 35:
            base_suitability *= 0.5
        elif 5 <= temp <= 25:
            base_suitability *= 1.0
        else:
            base_suitability *= 0.8

        # Apply sensitivity
        return base_suitability * (2 - sensitivity) / 2

    def _calculate_activity_crowd_suitability(self, activity, crowd, preference):
        """Simplified activity-crowd suitability"""
        density = crowd["density_score"]
        level_map = {"low": 0.2, "moderate": 0.5, "high": 0.8}
        preferred = level_map.get(preference, 0.5)

        # Some activities benefit from crowds, others don't
        if any(keyword in activity.lower() for keyword in ["social", "festival", "concert"]):
            # Social activities prefer more crowds
            return 1 - abs(density - min(preferred + 0.3, 1.0))
        else:
            # Most activities prefer less crowds
            return 1 - abs(density - max(preferred - 0.2, 0.0))

    def _calculate_event_activity_impact(self, event, activity):
        """Simplified event-activity impact"""
        # Events generally create positive or negative impact
        if event["type"] in ["festival", "concert"] and "social" in activity.lower():
            return 0.3  # Positive impact for social activities
        elif event["type"] in ["sports", "conference"]:
            return -0.2  # Slight negative impact due to crowds/traffic
        else:
            return 0.1  # Neutral slight impact

    def _generate_seasonal_recommendations(self, season, activity, preferences):
        """Generate season-specific recommendations"""
        recommendations = []

        season_tips = {
            "spring": ["Good for outdoor activities", "Watch for allergies", "Variable weather"],
            "summer": ["Stay hydrated", "Seek shade during peak hours", "Early morning activities recommended"],
            "autumn": ["Comfortable temperatures", "Great for hiking", "Layer clothing"],
            "winter": ["Dress warmly", "Consider indoor alternatives", "Shorter daylight hours"]
        }

        base_tips = season_tips.get(season, ["Consider seasonal conditions"])

        # Add activity-specific advice
        if "outdoor" in activity.lower():
            if season == "summer":
                recommendations.append("Avoid midday sun exposure")
            elif season == "winter":
                recommendations.append("Check for weather warnings")

        return base_tips + recommendations

    def _generate_air_quality_recommendation(self, air_quality, impact, preference):
        """Generate air quality-based recommendations"""
        aqi = air_quality["aqi"]

        if aqi <= 50:
            return "Excellent air quality - all activities recommended"
        elif aqi <= 100:
            return "Good air quality - normal activities okay"
        elif aqi <= 150:
            return "Moderate air quality - sensitive individuals should limit outdoor exposure"
        else:
            return "Poor air quality - limit outdoor activities, especially exercise"