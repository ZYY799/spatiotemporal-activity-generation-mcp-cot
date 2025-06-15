#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : quality_evaluator.py
# @Time    : 2025/6/15 15:37
# @Desc    : Quality evaluation system for generated spatiotemporal behaviors


import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import geopandas as gpd
from scipy import stats
from scipy.spatial.distance import jensenshannon
import json
import requests
import statistics

from generator import DailyTrajectory, ActivityInstance


class SpatiotemporalQualityEvaluator:
    """
    Combines subjective LLM evaluation with objective validation
    """

    def __init__(self,
                 mobile_signaling_data: pd.DataFrame,
                 spatial_units_shp: gpd.GeoDataFrame,
                 llm_api_key: str,
                 llm_base_url: str = "https://api.openai.com/v1",
                 reference_data: Optional[Dict[str, Any]] = None):

        self.mobile_data = mobile_signaling_data
        self.spatial_units = spatial_units_shp
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.reference_data = reference_data or {}
        self.evaluation_criteria = self._define_evaluation_criteria()

        self._prepare_reference_distributions()

    def evaluate_trajectory_quality(self, trajectory: DailyTrajectory) -> Dict[str, Any]:
        """Evaluate quality of a single trajectory"""

        quality_scores = {
            "temporal_consistency": self._evaluate_temporal_consistency(trajectory),
            "activity_coherence": self._evaluate_activity_coherence(trajectory),
            "spatial_realism": self._evaluate_spatial_realism(trajectory),
            "behavioral_authenticity": self._evaluate_behavioral_authenticity(trajectory),
            "constraint_satisfaction": self._evaluate_constraint_satisfaction(trajectory)
        }

        # Calculate weighted overall score
        weights = {
            "temporal_consistency": 0.25,
            "activity_coherence": 0.25,
            "spatial_realism": 0.20,
            "behavioral_authenticity": 0.20,
            "constraint_satisfaction": 0.10
        }

        overall_score = sum(
            quality_scores[criterion] * weights[criterion]
            for criterion in quality_scores
        )

        return {
            "individual_scores": quality_scores,
            "overall_score": overall_score,
            "quality_level": self._classify_quality_level(overall_score),
            "evaluation_timestamp": datetime.now().isoformat(),
            "trajectory_id": f"{trajectory.person_id}_{trajectory.date.isoformat()}"
        }

    def evaluate_population_quality(self, trajectories: List[DailyTrajectory],
                                    output_path: str = "evaluation_results.csv") -> Dict[str, Any]:
        """Evaluate quality across population of trajectories"""

        if not trajectories:
            return {"error": "No trajectories provided for evaluation"}

        # Evaluate individual trajectories
        individual_evaluations = []
        for i, trajectory in enumerate(trajectories):
            print(f"Evaluating trajectory {i + 1}/{len(trajectories)}")
            evaluation = self.evaluate_trajectory_quality(trajectory)
            individual_evaluations.append(evaluation)

        # Calculate population-level statistics
        population_stats = self._calculate_population_statistics(individual_evaluations)

        # Evaluate population diversity
        diversity_metrics = self._evaluate_population_diversity(trajectories)

        # Compare with reference data
        reference_comparison = self._compare_with_reference_data(trajectories)

        # Save to CSV
        self._save_results_to_csv(individual_evaluations, trajectories, output_path)

        return {
            "population_size": len(trajectories),
            "individual_evaluations": individual_evaluations,
            "population_statistics": population_stats,
            "diversity_metrics": diversity_metrics,
            "reference_comparison": reference_comparison,
            "output_file": output_path,
            "evaluation_summary": self._generate_evaluation_summary(
                population_stats, diversity_metrics, reference_comparison
            )
        }

    def _evaluate_temporal_consistency(self, trajectory: DailyTrajectory) -> float:
        """Evaluate temporal consistency using LLM assessment"""

        try:
            llm_scores = self._llm_evaluate_trajectory(trajectory)
            return llm_scores.get("temporal_consistency", 7.0)
        except:
            return self._rule_based_temporal_consistency(trajectory)

    def _evaluate_activity_coherence(self, trajectory: DailyTrajectory) -> float:
        """Evaluate logical coherence of activity sequence"""

        try:
            llm_scores = self._llm_evaluate_trajectory(trajectory)
            return llm_scores.get("activity_coherence", 7.0)
        except:
            return self._rule_based_activity_coherence(trajectory)

    def _evaluate_spatial_realism(self, trajectory: DailyTrajectory) -> float:
        """Evaluate spatial realism using objective validation"""

        spatial_js = self._calculate_spatial_js_divergence(trajectory)
        spatial_score = max(0, 10 * (1 - spatial_js / np.log(2)))

        return min(10.0, spatial_score)

    def _evaluate_behavioral_authenticity(self, trajectory: DailyTrajectory) -> float:
        """Evaluate authenticity of behavioral patterns"""

        temporal_ks = self._calculate_temporal_ks_statistic(trajectory)
        authenticity_score = max(0, 10 * (1 - temporal_ks))

        return min(10.0, authenticity_score)

    def _evaluate_constraint_satisfaction(self, trajectory: DailyTrajectory) -> float:
        """Evaluate constraint satisfaction"""

        score = 10.0

        # Time constraints
        total_time = sum(a.duration for a in trajectory.activities)
        if total_time > 20 * 60:
            score -= 3.0
        elif total_time < 4 * 60:
            score -= 2.0

        # Activity count constraints
        if len(trajectory.activities) > 10:
            score -= 2.0
        elif len(trajectory.activities) < 2:
            score -= 2.0

        return max(0.0, score)

    def _llm_evaluate_trajectory(self, trajectory: DailyTrajectory) -> Dict[str, float]:
        """LLM-based trajectory evaluation"""

        prompt = self._build_llm_evaluation_prompt(trajectory)
        response = self._call_llm_api(prompt)
        return self._parse_llm_response(response)

    def _build_llm_evaluation_prompt(self, trajectory: DailyTrajectory) -> str:
        """Build evaluation prompt for LLM"""

        activities_text = ""
        for i, activity in enumerate(trajectory.activities):
            activities_text += f"{i + 1}. {activity.activity_type} | "
            activities_text += f"{activity.start_time.strftime('%H:%M')}-{activity.end_time.strftime('%H:%M')} | "
            activities_text += f"{activity.duration}min"
            if activity.location:
                activities_text += f" | Location: ({activity.location[0]:.4f}, {activity.location[1]:.4f})"
            if activity.companions:
                activities_text += f" | Companions: {activity.companions}"
            activities_text += "\n"

        day_type = "Weekday" if trajectory.date.weekday() < 5 else "Weekend"

        prompt = f"""Evaluate this daily activity-travel chain across four dimensions. Score each dimension from 1.0 to 10.0 based on detailed criteria.

TRAJECTORY DATA:
Date: {trajectory.date.strftime('%Y-%m-%d')} ({day_type})
Person ID: {trajectory.person_id}
Total Activities: {len(trajectory.activities)}

ACTIVITY SEQUENCE:
{activities_text}

SCORING CRITERIA:

1. TEMPORAL_CONSISTENCY (Weight: 25%)
Score 9-10: Strong time logic, optimal utility allocation
- Activity sequence completely reasonable with no time conflicts
- Time allocation highly realistic, conforming to life patterns
- Sufficient and feasible transition time between activities
- Fully reflects individual utility maximization under time constraints

Score 7-8: Good time logic, reasonable utility allocation
- Activity sequence basically reasonable with occasional minor time pressure
- Time allocation relatively realistic
- Transition time between activities basically feasible
- Better reflects time utility optimization principles

Score 5-6: Average time logic, problematic utility allocation
- Activity sequence has certain unreasonableness
- Partial time allocation insufficiently realistic
- Transition time between activities too tight or too loose
- Time utility allocation needs optimization

Score 3-4: Poor time logic, inappropriate utility allocation
- Activity sequence unreasonable in multiple places
- Time allocation obviously deviates from reality
- Infeasible transition time between activities
- Fails to reflect reasonable time utility consideration

Score 1-2: Chaotic time logic, no utility optimization
- Severely disordered activity sequence
- Completely unrealistic time allocation
- Obvious time conflicts exist
- Completely ignores time utility maximization principles

2. ACTIVITY_COHERENCE (Weight: 25%)
Score 9-10: Excellent purpose coherence, optimal utility trade-off
- Strong logical correlation in activity sequence
- Perfect match between travel purpose and activity types
- Highly reasonable overall activity chain
- Fully reflects utility trade-off optimization in multi-objective decisions

Score 7-8: Good purpose coherence, reasonable utility trade-off
- Strong logical correlation in activity sequence
- High match between travel purpose and activity types
- Overall activity chain relatively reasonable
- Better reflects utility trade-off mechanisms

Score 5-6: Average purpose coherence, flawed utility trade-off
- Average logical correlation in activity sequence
- Medium match between travel purpose and activity types
- Activity chain has certain unreasonableness
- Insufficient reflection of utility trade-off mechanisms

Score 3-4: Poor purpose coherence, imbalanced utility trade-off
- Weak logical correlation in activity sequence
- Low match between travel purpose and activity types
- Overall activity chain insufficiently reasonable
- Lacks effective utility trade-off consideration

Score 1-2: Very poor purpose coherence, no utility trade-off
- Activity sequence lacks logical correlation
- Serious mismatch between travel purpose and activity types
- Overall chaotic activity chain
- Completely ignores utility trade-off principles

3. PERSONA_CONFORMITY (Weight: 25%)
Score 9-10: Extremely high persona matching, accurate utility preferences
- Activity patterns highly match persona characteristics
- Sufficient reflection of occupation, age, lifestyle habits characteristics
- Perfect reflection of typical utility preferences of the group
- Outstanding and reasonable personalized characteristics

Score 7-8: Good persona matching, relatively accurate utility preferences
- Activity patterns match persona characteristics well
- Main characteristics clearly reflected
- Better reflection of group utility preference characteristics
- Basically reasonable personalized characteristics

Score 5-6: Average persona matching, vague utility preferences
- Medium matching between activity patterns and persona characteristics
- Insufficient reflection of some characteristics
- General reflection of group utility preferences
- Insufficiently prominent personalized characteristics

Score 3-4: Poor persona matching, incorrect utility preferences
- Low matching between activity patterns and persona characteristics
- Most characteristics not obviously reflected
- Inappropriate reflection of group utility preferences
- Missing or unreasonable personalized characteristics

Score 1-2: Very poor persona matching, no utility preference reflection
- Activity patterns seriously inconsistent with persona characteristics
- Chaotic or missing characteristic reflection
- Completely ignores group utility preferences
- No reasonable personalization

4. BEHAVIORAL_AUTHENTICITY (Weight: 25%)
Score 9-10: Extremely high activity authenticity, optimal rational choice
- Activity content completely conforms to reality
- Highly reasonable duration
- Optimal location selection
- Fully reflects rational choice behavior under resource constraints

Score 7-8: Good activity authenticity, reasonable rational choice
- Activity content basically conforms to reality
- Relatively reasonable duration
- Appropriate location selection
- Better reflects rational choice principles

Score 5-6: Average activity authenticity, biased rational choice
- Activity content roughly conforms to reality
- Duration has certain deviations
- Average location selection
- Insufficient reflection of rational choice

Score 3-4: Poor activity authenticity, inappropriate rational choice
- Activity content deviates from reality
- Obviously unreasonable duration
- Inappropriate location selection
- Lacks rational choice consideration

Score 1-2: Very poor activity authenticity, no rational choice
- Activity content seriously deviates from reality
- Completely unreasonable duration
- Absurd location selection
- Completely ignores rational choice behavior

REQUIRED OUTPUT FORMAT (JSON only):
{{
"temporal_consistency": X.X,
"activity_coherence": X.X,
"persona_conformity": X.X,
"behavioral_authenticity": X.X
}}"""

        return prompt

    def _call_llm_api(self, prompt: str) -> str:
        """Call LLM API for evaluation"""

        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in human activity pattern analysis. Evaluate trajectories objectively and return only the requested JSON format with numerical scores."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 150
        }

        response = requests.post(
            f"{self.llm_base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"API call failed: {response.status_code}")

        return response.json()["choices"][0]["message"]["content"]

    def _parse_llm_response(self, response: str) -> Dict[str, float]:
        """Parse LLM evaluation response"""

        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found")

            json_str = response[start_idx:end_idx]
            scores = json.loads(json_str)

            return {
                "temporal_consistency": float(scores.get("temporal_consistency", 7.0)),
                "activity_coherence": float(scores.get("activity_coherence", 7.0)),
                "persona_conformity": float(scores.get("persona_conformity", 7.0)),
                "behavioral_authenticity": float(scores.get("behavioral_authenticity", 7.0))
            }
        except Exception as e:
            raise ValueError(f"Failed to parse response: {e}")

    def _calculate_spatial_js_divergence(self, trajectory: DailyTrajectory) -> float:
        """Calculate spatial Jensen-Shannon divergence"""

        traj_spatial_dist = np.zeros(78)

        for activity in trajectory.activities:
            if activity.location:
                unit_id = self._get_spatial_unit_id(activity.location)
                if 0 <= unit_id < 78:
                    traj_spatial_dist[unit_id] += 1

        if traj_spatial_dist.sum() > 0:
            traj_spatial_dist = traj_spatial_dist / traj_spatial_dist.sum()
        else:
            return 1.0

        return jensenshannon(self.reference_spatial_dist, traj_spatial_dist)

    def _calculate_temporal_ks_statistic(self, trajectory: DailyTrajectory) -> float:
        """Calculate temporal KS statistic"""

        if not trajectory.activities:
            return 1.0

        start_times = [
            act.start_time.hour + act.start_time.minute / 60
            for act in trajectory.activities
        ]

        ks_statistic, _ = stats.ks_2samp(start_times, self.reference_temporal_dist)
        return ks_statistic

    def _get_spatial_unit_id(self, location: Tuple[float, float]) -> int:
        """Get spatial unit ID from coordinates"""

        from shapely.geometry import Point

        point = Point(location[1], location[0])

        for idx, unit in self.spatial_units.iterrows():
            if unit.geometry.contains(point):
                return idx

        return -1

    def _rule_based_temporal_consistency(self, trajectory: DailyTrajectory) -> float:
        """Rule-based temporal consistency evaluation"""

        if not trajectory.activities:
            return 0.0

        score = 8.0

        # Check overlaps
        for i in range(len(trajectory.activities) - 1):
            current = trajectory.activities[i]
            next_act = trajectory.activities[i + 1]
            if current.end_time > next_act.start_time:
                score -= 2.0

        # Check durations
        for activity in trajectory.activities:
            if activity.duration < 5 or activity.duration > 480:
                score -= 1.0

        return max(0.0, min(10.0, score))

    def _rule_based_activity_coherence(self, trajectory: DailyTrajectory) -> float:
        """Rule-based activity coherence evaluation"""

        if not trajectory.activities:
            return 0.0

        score = 7.0
        activity_types = [a.activity_type for a in trajectory.activities]

        if "home" in activity_types:
            score += 1.0
        if "work" in activity_types and trajectory.date.weekday() < 5:
            score += 1.0

        unique_ratio = len(set(activity_types)) / len(activity_types)
        score += unique_ratio * 1.0

        return min(10.0, score)

    def _prepare_reference_distributions(self):
        """Prepare reference distributions"""

        # Spatial distribution
        if 'spatial_unit_id' in self.mobile_data.columns:
            spatial_counts = self.mobile_data['spatial_unit_id'].value_counts().sort_index()
            self.reference_spatial_dist = np.zeros(78)
            for unit_id, count in spatial_counts.items():
                if 0 <= unit_id < 78:
                    self.reference_spatial_dist[unit_id] = count
            self.reference_spatial_dist = self.reference_spatial_dist / self.reference_spatial_dist.sum()
        else:
            self.reference_spatial_dist = np.ones(78) / 78

        # Temporal distribution
        if 'start_time' in self.mobile_data.columns:
            start_times = pd.to_datetime(self.mobile_data['start_time'])
            self.reference_temporal_dist = (start_times.dt.hour + start_times.dt.minute / 60).values
        else:
            self.reference_temporal_dist = np.concatenate([
                np.random.normal(8, 1, 100),
                np.random.normal(12, 0.5, 50),
                np.random.normal(18, 2, 100)
            ])

    def _calculate_population_statistics(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate population statistics"""

        overall_scores = [eval_data["overall_score"] for eval_data in evaluations]

        criterion_stats = {}
        for criterion in ["temporal_consistency", "activity_coherence",
                          "spatial_realism", "behavioral_authenticity", "constraint_satisfaction"]:
            criterion_scores = [
                eval_data["individual_scores"][criterion]
                for eval_data in evaluations
            ]

            criterion_stats[criterion] = {
                "mean": statistics.mean(criterion_scores),
                "median": statistics.median(criterion_scores),
                "std": statistics.stdev(criterion_scores) if len(criterion_scores) > 1 else 0,
                "min": min(criterion_scores),
                "max": max(criterion_scores)
            }

        return {
            "overall_statistics": {
                "mean": statistics.mean(overall_scores),
                "median": statistics.median(overall_scores),
                "std": statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
                "min": min(overall_scores),
                "max": max(overall_scores)
            },
            "criterion_statistics": criterion_stats,
            "quality_distribution": self._calculate_quality_distribution(overall_scores)
        }

    def _save_results_to_csv(self, evaluations: List[Dict[str, Any]],
                             trajectories: List[DailyTrajectory], output_path: str):
        """Save evaluation results to CSV"""

        results = []
        for i, (evaluation, trajectory) in enumerate(zip(evaluations, trajectories)):
            result = {
                "trajectory_id": evaluation["trajectory_id"],
                "person_id": trajectory.person_id,
                "date": trajectory.date.isoformat(),
                "num_activities": len(trajectory.activities),
                "temporal_consistency": evaluation["individual_scores"]["temporal_consistency"],
                "activity_coherence": evaluation["individual_scores"]["activity_coherence"],
                "spatial_realism": evaluation["individual_scores"]["spatial_realism"],
                "behavioral_authenticity": evaluation["individual_scores"]["behavioral_authenticity"],
                "constraint_satisfaction": evaluation["individual_scores"]["constraint_satisfaction"],
                "overall_score": evaluation["overall_score"],
                "quality_level": evaluation["quality_level"],
                "evaluation_timestamp": evaluation["evaluation_timestamp"]
            }
            results.append(result)

        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    def _calculate_entropy(self, values: List[Any]) -> float:
        """Calculate entropy"""

        if not values:
            return 0.0

        from collections import Counter
        counts = Counter(values)
        total = len(values)

        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def _classify_quality_level(self, score: float) -> str:
        """Classify quality level"""

        if score >= 8.5:
            return "excellent"
        elif score >= 7.0:
            return "good"
        elif score >= 5.5:
            return "average"
        elif score >= 4.0:
            return "poor"
        else:
            return "very_poor"

    def _calculate_quality_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calculate quality distribution"""

        distribution = {"excellent": 0, "good": 0, "average": 0, "poor": 0, "very_poor": 0}

        for score in scores:
            level = self._classify_quality_level(score)
            distribution[level] += 1

        return distribution

    # deprecated
    def _define_evaluation_criteria(self) -> Dict[str, Dict[str, Any]]:
        """Define evaluation criteria"""

        return {
            "temporal_consistency": {
                "description": "Logical temporal ordering and realistic durations",
                "weight": 0.25,
                "thresholds": {"excellent": 9.0, "good": 7.5, "average": 6.0}
            },
            "activity_coherence": {
                "description": "Logical activity sequences and relationships",
                "weight": 0.25,
                "thresholds": {"excellent": 8.5, "good": 7.0, "average": 5.5}
            },
            "spatial_realism": {
                "description": "Realistic spatial patterns and travel times",
                "weight": 0.20,
                "thresholds": {"excellent": 8.0, "good": 6.5, "average": 5.0}
            },
            "behavioral_authenticity": {
                "description": "Authentic human behavioral patterns",
                "weight": 0.20,
                "thresholds": {"excellent": 8.0, "good": 6.5, "average": 5.0}
            },
            "constraint_satisfaction": {
                "description": "Adherence to known constraints",
                "weight": 0.10,
                "thresholds": {"excellent": 9.5, "good": 8.0, "average": 6.5}
            }
        }