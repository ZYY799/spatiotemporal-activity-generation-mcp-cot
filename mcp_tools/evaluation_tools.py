#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : evaluation_tools.py
# @Time    : 2025/6/15 15:33
# @Desc    : Experience Evaluation Tools for activity and location assessment


from datetime import datetime
from typing import Dict, List, Any, Optional
from .base_tool import BaseMCPTool, MCPMessage
import statistics


class ExperienceEvaluationTool(BaseMCPTool):
    """
    Experience evaluation tool for assessing activities, locations, and routes
    """

    def __init__(self, memory_manager=None):
        super().__init__("experience_evaluation", memory_manager)
        self.capabilities = [
            "activity_evaluation",
            "location_assessment",
            "route_evaluation",
            "satisfaction_analysis",
            "recommendation_scoring",
            "experience_comparison"
        ]

    def get_capabilities(self) -> List[str]:
        return self.capabilities

    def process_query(self, query: MCPMessage) -> MCPMessage:
        """Process evaluation-related queries"""

        query_type = query.data.get("query_type")

        if query_type == "activity_evaluation":
            return self._handle_activity_evaluation(query)
        elif query_type == "location_assessment":
            return self._handle_location_assessment(query)
        elif query_type == "route_evaluation":
            return self._handle_route_evaluation(query)
        elif query_type == "satisfaction_analysis":
            return self._handle_satisfaction_analysis(query)
        elif query_type == "recommendation_scoring":
            return self._handle_recommendation_scoring(query)
        elif query_type == "experience_comparison":
            return self._handle_experience_comparison(query)
        else:
            return self._create_response(
                {"error": f"Unknown query type: {query_type}"},
                status="error"
            )

    def _handle_activity_evaluation(self, query: MCPMessage) -> MCPMessage:
        """Evaluate activity experience and satisfaction"""

        activity_data = query.data.get("activity")
        evaluation_criteria = query.data.get("criteria", ["satisfaction", "duration", "cost", "accessibility"])

        activity_type = activity_data.get("type")
        duration = activity_data.get("duration", 60)
        location = activity_data.get("location")
        cost = activity_data.get("cost", 0)
        companions = activity_data.get("companions", [])

        # Get personal evaluation standards from memory
        evaluation_weights = self._get_evaluation_weights(activity_type)
        personal_standards = self._get_personal_standards(activity_type)

        # Evaluate each criterion
        criterion_scores = {}

        if "satisfaction" in evaluation_criteria:
            # Expected satisfaction based on personal preferences
            expected_satisfaction = self._calculate_expected_satisfaction(
                activity_type, location, companions, personal_standards
            )
            criterion_scores["satisfaction"] = expected_satisfaction

        if "duration" in evaluation_criteria:
            # Duration appropriateness
            preferred_duration = self._get_personal_preference(
                f"{activity_type}_preferred_duration", 60
            )
            duration_score = self._evaluate_duration_appropriateness(duration, preferred_duration)
            criterion_scores["duration"] = duration_score

        if "cost" in evaluation_criteria:
            # Cost value assessment
            cost_sensitivity = self._get_personal_preference("cost_sensitivity", 0.5)
            budget_range = self._get_personal_preference(f"{activity_type}_budget_range", (0, 100))
            cost_score = self._evaluate_cost_value(cost, budget_range, cost_sensitivity)
            criterion_scores["cost"] = cost_score

        if "accessibility" in evaluation_criteria:
            # Location accessibility
            accessibility_score = self._evaluate_accessibility(location, activity_type)
            criterion_scores["accessibility"] = accessibility_score

        # Calculate overall score
        overall_score = sum(
            criterion_scores[criterion] * evaluation_weights.get(criterion, 0.25)
            for criterion in criterion_scores
        )

        # Generate recommendations
        recommendations = self._generate_activity_recommendations(
            criterion_scores, personal_standards
        )

        response_data = {
            "activity_evaluation": {
                "overall_score": overall_score,
                "criterion_scores": criterion_scores,
                "evaluation_basis": "personal_preferences_and_history"
            },
            "recommendations": recommendations,
            "improvement_suggestions": self._generate_improvement_suggestions(
                criterion_scores, evaluation_weights
            )
        }

        return self._create_response(response_data)

    def _handle_location_assessment(self, query: MCPMessage) -> MCPMessage:
        """Assess location quality for specific activities"""

        location = query.data.get("location")
        activity_types = query.data.get("activity_types", [])
        assessment_factors = query.data.get("factors", ["convenience", "ambiance", "facilities", "safety"])

        location_assessments = {}

        for activity_type in activity_types:
            # Get location history from memory
            location_history = self._get_location_history(location, activity_type)
            personal_location_preferences = self._get_location_preferences(activity_type)

            factor_scores = {}

            if "convenience" in assessment_factors:
                convenience_score = self._assess_location_convenience(
                    location, activity_type, personal_location_preferences
                )
                factor_scores["convenience"] = convenience_score

            if "ambiance" in assessment_factors:
                ambiance_score = self._assess_location_ambiance(
                    location, activity_type, personal_location_preferences
                )
                factor_scores["ambiance"] = ambiance_score

            if "facilities" in assessment_factors:
                facilities_score = self._assess_location_facilities(
                    location, activity_type, personal_location_preferences
                )
                factor_scores["facilities"] = facilities_score

            if "safety" in assessment_factors:
                safety_score = self._assess_location_safety(
                    location, personal_location_preferences
                )
                factor_scores["safety"] = safety_score

            # Calculate overall location score for this activity
            overall_location_score = statistics.mean(factor_scores.values()) if factor_scores else 0.5

            location_assessments[activity_type] = {
                "overall_score": overall_location_score,
                "factor_scores": factor_scores,
                "visit_history": location_history,
                "suitability": "high" if overall_location_score > 0.7 else "medium" if overall_location_score > 0.4 else "low"
            }

        response_data = {
            "location": location,
            "assessments": location_assessments,
            "overall_location_rating": statistics.mean(
                assessment["overall_score"] for assessment in location_assessments.values()
            ) if location_assessments else 0.5
        }

        return self._create_response(response_data)

    def _handle_route_evaluation(self, query: MCPMessage) -> MCPMessage:
        """Evaluate route quality and experience"""

        route_data = query.data.get("route")
        evaluation_aspects = query.data.get("aspects", ["efficiency", "comfort", "cost", "scenery"])

        origin = route_data.get("origin")
        destination = route_data.get("destination")
        transport_mode = route_data.get("transport_mode")
        duration = route_data.get("duration", 30)
        distance = route_data.get("distance", 2000)
        cost = route_data.get("cost", 0)

        # Get personal route preferences
        route_preferences = self._get_route_preferences(transport_mode)

        aspect_scores = {}

        if "efficiency" in evaluation_aspects:
            efficiency_score = self._evaluate_route_efficiency(
                duration, distance, transport_mode, route_preferences
            )
            aspect_scores["efficiency"] = efficiency_score

        if "comfort" in evaluation_aspects:
            comfort_score = self._evaluate_route_comfort(
                transport_mode, duration, route_preferences
            )
            aspect_scores["comfort"] = comfort_score

        if "cost" in evaluation_aspects:
            cost_score = self._evaluate_route_cost(
                cost, distance, transport_mode, route_preferences
            )
            aspect_scores["cost"] = cost_score

        if "scenery" in evaluation_aspects:
            scenery_score = self._evaluate_route_scenery(
                origin, destination, route_preferences
            )
            aspect_scores["scenery"] = scenery_score

        # Calculate overall route score
        route_weights = self._get_route_evaluation_weights()
        overall_route_score = sum(
            aspect_scores[aspect] * route_weights.get(aspect, 0.25)
            for aspect in aspect_scores
        )

        response_data = {
            "route_evaluation": {
                "overall_score": overall_route_score,
                "aspect_scores": aspect_scores,
                "transport_mode": transport_mode
            },
            "route_recommendation": "highly_recommended" if overall_route_score > 0.8 else
            "recommended" if overall_route_score > 0.6 else "acceptable",
            "alternative_suggestions": self._generate_route_alternatives(
                aspect_scores, route_preferences
            )
        }

        return self._create_response(response_data)

    def _handle_satisfaction_analysis(self, query: MCPMessage) -> MCPMessage:
        """Analyze satisfaction patterns across activities"""

        time_period = query.data.get("time_period", "past_month")
        activity_types = query.data.get("activity_types", [])

        # Get satisfaction history from memory
        satisfaction_data = self._get_satisfaction_history(time_period, activity_types)

        # Analyze patterns
        satisfaction_analysis = {
            "average_satisfaction": statistics.mean(satisfaction_data.values()) if satisfaction_data else 0.5,
            "satisfaction_by_activity": satisfaction_data,
            "highest_satisfaction": max(satisfaction_data.items(), key=lambda x: x[1]) if satisfaction_data else None,
            "lowest_satisfaction": min(satisfaction_data.items(), key=lambda x: x[1]) if satisfaction_data else None,
            "satisfaction_trend": self._calculate_satisfaction_trend(satisfaction_data),
            "satisfaction_variance": statistics.variance(satisfaction_data.values()) if len(
                satisfaction_data) > 1 else 0
        }

        # Generate insights
        insights = self._generate_satisfaction_insights(satisfaction_analysis)

        response_data = {
            "satisfaction_analysis": satisfaction_analysis,
            "insights": insights,
            "recommendations": self._generate_satisfaction_recommendations(satisfaction_analysis)
        }

        return self._create_response(response_data)

    def _handle_recommendation_scoring(self, query: MCPMessage) -> MCPMessage:
        """Score and rank activity/location recommendations"""

        recommendations = query.data.get("recommendations", [])
        scoring_criteria = query.data.get("criteria", ["personal_fit", "feasibility", "novelty"])

        scored_recommendations = []

        for rec in recommendations:
            scores = {}

            if "personal_fit" in scoring_criteria:
                fit_score = self._score_personal_fit(rec)
                scores["personal_fit"] = fit_score

            if "feasibility" in scoring_criteria:
                feasibility_score = self._score_feasibility(rec)
                scores["feasibility"] = feasibility_score

            if "novelty" in scoring_criteria:
                novelty_score = self._score_novelty(rec)
                scores["novelty"] = novelty_score

            # Calculate overall recommendation score
            overall_score = statistics.mean(scores.values()) if scores else 0.5

            scored_recommendations.append({
                "recommendation": rec,
                "scores": scores,
                "overall_score": overall_score,
                "confidence": self._calculate_recommendation_confidence(scores)
            })

        # Sort by overall score
        scored_recommendations.sort(key=lambda x: x["overall_score"], reverse=True)

        response_data = {
            "scored_recommendations": scored_recommendations,
            "top_recommendation": scored_recommendations[0] if scored_recommendations else None,
            "scoring_summary": self._generate_scoring_summary(scored_recommendations)
        }

        return self._create_response(response_data)

    def _handle_experience_comparison(self, query: MCPMessage) -> MCPMessage:
        """Compare different activity or location experiences"""

        experiences = query.data.get("experiences", [])
        comparison_dimensions = query.data.get("dimensions", ["satisfaction", "cost", "duration", "convenience"])

        if len(experiences) < 2:
            return self._create_response(
                {"error": "At least 2 experiences required for comparison"},
                status="error"
            )

        comparison_results = {}

        for dimension in comparison_dimensions:
            dimension_scores = []
            for exp in experiences:
                score = self._evaluate_experience_dimension(exp, dimension)
                dimension_scores.append({
                    "experience": exp.get("name", "Unknown"),
                    "score": score
                })

            # Sort by score for this dimension
            dimension_scores.sort(key=lambda x: x["score"], reverse=True)
            comparison_results[dimension] = dimension_scores

        # Calculate overall rankings
        overall_rankings = self._calculate_overall_rankings(experiences, comparison_results)

        response_data = {
            "comparison_results": comparison_results,
            "overall_rankings": overall_rankings,
            "winner": overall_rankings[0] if overall_rankings else None,
            "comparison_insights": self._generate_comparison_insights(comparison_results)
        }

        return self._create_response(response_data)

    # Helper methods for evaluation

    def _get_evaluation_weights(self, activity_type: str) -> Dict[str, float]:
        """Get personal evaluation weights for activity type"""

        default_weights = {"satisfaction": 0.4, "duration": 0.2, "cost": 0.2, "accessibility": 0.2}

        # Get personalized weights from memory
        personal_weights = {}
        for criterion in default_weights:
            weight_key = f"{activity_type}_{criterion}_weight"
            personal_weights[criterion] = self._get_personal_preference(weight_key, default_weights[criterion])

        # Normalize weights
        total_weight = sum(personal_weights.values())
        if total_weight > 0:
            personal_weights = {k: v / total_weight for k, v in personal_weights.items()}

        return personal_weights

    def _get_personal_standards(self, activity_type: str) -> Dict[str, Any]:
        """Get personal standards for activity evaluation"""

        return {
            "min_satisfaction": self._get_personal_preference(f"{activity_type}_min_satisfaction", 0.6),
            "max_cost": self._get_personal_preference(f"{activity_type}_max_cost", 100),
            "preferred_duration": self._get_personal_preference(f"{activity_type}_preferred_duration", 60),
            "accessibility_requirement": self._get_personal_preference(f"{activity_type}_accessibility_req", 0.7)
        }

    # Additional helper methods would continue here...