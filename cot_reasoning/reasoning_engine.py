#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : reasoning_engine.py
# @Time    : 2025/6/15 15:34
# @Desc    : Core Chain-of-Thought Reasoning Engine implementing the five-stage cognitive framework


from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import re


@dataclass
class ReasoningContext:
    """Context for reasoning process"""
    person_profile: Dict[str, Any]
    current_situation: Dict[str, Any]
    memory_context: Dict[str, Any]
    environmental_context: Dict[str, Any]
    constraints: Dict[str, Any]


@dataclass
class ReasoningStep:
    """Individual reasoning step in CoT"""
    stage: str
    step_id: str
    description: str
    input_data: Dict[str, Any]
    reasoning_process: str
    output_data: Dict[str, Any]
    confidence: float
    timestamp: datetime


class ChainOfThoughtEngine:
    """
    Core reasoning engine implementing the five-stage cognitive framework:
    1. Situational Awareness and Problem Definition
    2. Constraint Identification and Goal Clarification
    3. Option Generation and Preliminary Screening
    4. Multi-factor Evaluation and Comparison
    5. Decision Formation and Consequence Prediction
    """

    def __init__(self, mcp_tool_manager=None, memory_manager=None):
        self.mcp_tool_manager = mcp_tool_manager
        self.memory_manager = memory_manager
        self.reasoning_templates = self._load_reasoning_templates()
        self.current_reasoning_chain = []

    def execute_reasoning_chain(self, context: ReasoningContext,
                                reasoning_goal: str) -> Dict[str, Any]:
        """Execute complete five-stage reasoning chain"""

        self.current_reasoning_chain = []

        try:
            # Stage 1: Situational Awareness and Problem Definition
            stage1_result = self._stage1_situational_awareness(context, reasoning_goal)

            # Stage 2: Constraint Identification and Goal Clarification
            stage2_result = self._stage2_constraint_identification(context, stage1_result)

            # Stage 3: Option Generation and Preliminary Screening
            stage3_result = self._stage3_option_generation(context, stage2_result)

            # Stage 4: Multi-factor Evaluation and Comparison
            stage4_result = self._stage4_multifactor_evaluation(context, stage3_result)

            # Stage 5: Decision Formation and Consequence Prediction
            stage5_result = self._stage5_decision_formation(context, stage4_result)

            # Compile final reasoning result
            final_result = {
                "reasoning_chain": self.current_reasoning_chain,
                "final_decision": stage5_result,
                "reasoning_quality": self._assess_reasoning_quality(),
                "confidence": self._calculate_overall_confidence(),
                "alternative_options": stage4_result.get("alternative_options", [])
            }

            return final_result

        except Exception as e:
            return {
                "error": f"Reasoning chain failed: {str(e)}",
                "partial_chain": self.current_reasoning_chain,
                "failure_stage": len(self.current_reasoning_chain) + 1
            }

    def _stage1_situational_awareness(self, context: ReasoningContext,
                                      reasoning_goal: str) -> Dict[str, Any]:
        """Stage 1: Situational Awareness and Problem Definition"""

        reasoning_process = f"""
        STAGE 1: SITUATIONAL AWARENESS AND PROBLEM DEFINITION

        Goal: {reasoning_goal}

        Current Situation Analysis:
        - Time: {context.current_situation.get('current_time', 'Unknown')}
        - Location: {context.current_situation.get('current_location', 'Unknown')}
        - Day Type: {context.current_situation.get('day_type', 'Unknown')}
        - Weather: {context.environmental_context.get('weather', 'Unknown')}

        Personal Context:
        - Age: {context.person_profile.get('demographics', {}).get('age', 'Unknown')}
        - Occupation: {context.person_profile.get('demographics', {}).get('occupation', 'Unknown')}
        - Current Energy Level: {context.current_situation.get('energy_level', 'Unknown')}

        Problem Definition:
        The individual needs to make a decision about {reasoning_goal} considering their personal preferences,
        current situation, and environmental factors. This requires analyzing available options and selecting
        the most suitable course of action.

        Key Factors Identified:
        1. Personal preferences and historical patterns
        2. Current temporal and spatial constraints
        3. Environmental conditions and social context
        4. Available resources and accessibility
        """

        # Extract key situational factors
        situational_factors = {
            "temporal_context": {
                "current_time": context.current_situation.get('current_time'),
                "day_type": context.current_situation.get('day_type'),
                "time_constraints": context.constraints.get('temporal', {})
            },
            "spatial_context": {
                "current_location": context.current_situation.get('current_location'),
                "location_constraints": context.constraints.get('spatial', {})
            },
            "personal_context": {
                "demographics": context.person_profile.get('demographics', {}),
                "preferences": context.person_profile.get('activity_preferences', {}),
                "energy_level": context.current_situation.get('energy_level', 0.7)
            },
            "environmental_context": context.environmental_context,
            "social_context": context.current_situation.get('social_context', {})
        }

        # Query memory for relevant past experiences
        relevant_memories = []
        if self.memory_manager:
            memory_query_context = {
                "hour": datetime.now().hour,
                "location": context.current_situation.get('current_location'),
                "activity_type": reasoning_goal
            }
            memories = self.memory_manager.retrieve_relevant_memories(memory_query_context, k=3)
            relevant_memories = memories.get('events', [])

        result = {
            "stage": "situational_awareness",
            "problem_definition": reasoning_goal,
            "situational_factors": situational_factors,
            "relevant_memories": relevant_memories,
            "key_considerations": [
                "Personal preference alignment",
                "Temporal feasibility",
                "Spatial accessibility",
                "Environmental suitability",
                "Resource availability"
            ]
        }

        # Log reasoning step
        step = ReasoningStep(
            stage="Stage 1",
            step_id="situational_awareness",
            description="Analyze current situation and define problem",
            input_data={"context": context.__dict__, "goal": reasoning_goal},
            reasoning_process=reasoning_process,
            output_data=result,
            confidence=0.9,
            timestamp=datetime.now()
        )
        self.current_reasoning_chain.append(step)

        return result

    def _stage2_constraint_identification(self, context: ReasoningContext,
                                          stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: Constraint Identification and Goal Clarification"""

        reasoning_process = f"""
        STAGE 2: CONSTRAINT IDENTIFICATION AND GOAL CLARIFICATION

        Based on Stage 1 analysis, identifying specific constraints and clarifying goals:

        TEMPORAL CONSTRAINTS:
        - Available time window
        - Scheduling conflicts
        - Personal energy cycles
        - Deadline pressures

        SPATIAL CONSTRAINTS:
        - Current location limitations
        - Transportation accessibility
        - Distance tolerance
        - Location availability

        RESOURCE CONSTRAINTS:
        - Financial budget
        - Physical capabilities
        - Equipment/tool availability
        - Social companion availability

        PERSONAL CONSTRAINTS:
        - Individual preferences and aversions
        - Health considerations
        - Social comfort zones
        - Habit patterns

        ENVIRONMENTAL CONSTRAINTS:
        - Weather conditions
        - Crowd levels
        - Special events
        - Seasonal factors

        GOAL CLARIFICATION:
        Primary goal: {stage1_result['problem_definition']}
        Secondary considerations: Personal satisfaction, efficiency, social connection
        """

        # Identify specific constraints using MCP tools
        identified_constraints = {
            "temporal": self._identify_temporal_constraints(context, stage1_result),
            "spatial": self._identify_spatial_constraints(context, stage1_result),
            "resource": self._identify_resource_constraints(context, stage1_result),
            "personal": self._identify_personal_constraints(context, stage1_result),
            "environmental": self._identify_environmental_constraints(context, stage1_result)
        }

        # Clarify goals based on personal preferences
        goal_clarification = self._clarify_goals(context, stage1_result)

        # Identify potential conflicts
        constraint_conflicts = self._identify_constraint_conflicts(identified_constraints)

        result = {
            "stage": "constraint_identification",
            "identified_constraints": identified_constraints,
            "goal_clarification": goal_clarification,
            "constraint_conflicts": constraint_conflicts,
            "constraint_hierarchy": self._prioritize_constraints(identified_constraints, context)
        }

        step = ReasoningStep(
            stage="Stage 2",
            step_id="constraint_identification",
            description="Identify constraints and clarify goals",
            input_data=stage1_result,
            reasoning_process=reasoning_process,
            output_data=result,
            confidence=0.85,
            timestamp=datetime.now()
        )
        self.current_reasoning_chain.append(step)

        return result

    def _stage3_option_generation(self, context: ReasoningContext,
                                  stage2_result: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: Option Generation and Preliminary Screening"""

        reasoning_process = f"""
        STAGE 3: OPTION GENERATION AND PRELIMINARY SCREENING

        Generating diverse options considering identified constraints:

        OPTION GENERATION STRATEGIES:
        1. Memory-based options: Drawing from successful past experiences
        2. Preference-based options: Aligned with personal preferences
        3. Constraint-optimized options: Designed to satisfy key constraints
        4. Novel options: Exploring new possibilities within comfort zone
        5. Compromise options: Balancing competing constraints

        PRELIMINARY SCREENING CRITERIA:
        - Feasibility within hard constraints
        - Basic preference alignment
        - Resource availability
        - Risk assessment
        - Time compatibility

        SCREENING PROCESS:
        Eliminating options that violate critical constraints while maintaining
        diverse alternatives for detailed evaluation.
        """

        # Generate initial option pool
        generated_options = self._generate_option_pool(context, stage2_result)

        # Apply preliminary screening
        screened_options = self._preliminary_screening(generated_options, stage2_result)

        # Categorize remaining options
        option_categories = self._categorize_options(screened_options)

        # Assess option diversity
        diversity_assessment = self._assess_option_diversity(screened_options)

        result = {
            "stage": "option_generation",
            "generated_options": generated_options,
            "screened_options": screened_options,
            "option_categories": option_categories,
            "diversity_assessment": diversity_assessment,
            "screening_summary": {
                "total_generated": len(generated_options),
                "passed_screening": len(screened_options),
                "elimination_reasons": self._summarize_elimination_reasons(generated_options, screened_options)
            }
        }

        step = ReasoningStep(
            stage="Stage 3",
            step_id="option_generation",
            description="Generate and screen potential options",
            input_data=stage2_result,
            reasoning_process=reasoning_process,
            output_data=result,
            confidence=0.8,
            timestamp=datetime.now()
        )
        self.current_reasoning_chain.append(step)

        return result

    def _stage4_multifactor_evaluation(self, context: ReasoningContext,
                                       stage3_result: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 4: Multi-factor Evaluation and Comparison"""

        reasoning_process = f"""
        STAGE 4: MULTI-FACTOR EVALUATION AND COMPARISON

        Conducting comprehensive evaluation of screened options:

        EVALUATION DIMENSIONS:
        1. Personal Preference Alignment (40% weight)
           - Historical satisfaction patterns
           - Stated preferences
           - Personality compatibility

        2. Constraint Satisfaction (25% weight)
           - Temporal feasibility
           - Resource requirements
           - Accessibility factors

        3. Expected Outcomes (20% weight)
           - Satisfaction probability
           - Goal achievement likelihood
           - Positive experience potential

        4. Risk and Uncertainty (10% weight)
           - Failure probability
           - Negative consequence severity
           - Unpredictability factors

        5. Opportunity Cost (5% weight)
           - Alternative forgone
           - Future option implications
           - Resource allocation efficiency

        COMPARISON METHODOLOGY:
        Using weighted multi-criteria decision analysis with personal preference weights
        derived from memory patterns and stated preferences.
        """

        screened_options = stage3_result["screened_options"]

        # Evaluate each option across multiple factors
        evaluated_options = []
        for option in screened_options:
            evaluation = self._comprehensive_option_evaluation(option, context, stage3_result)
            evaluated_options.append({
                "option": option,
                "evaluation": evaluation,
                "overall_score": evaluation["overall_score"],
                "confidence": evaluation["confidence"]
            })

        # Rank options by overall score
        evaluated_options.sort(key=lambda x: x["overall_score"], reverse=True)

        # Perform sensitivity analysis
        sensitivity_analysis = self._perform_sensitivity_analysis(evaluated_options, context)

        # Identify trade-offs
        trade_off_analysis = self._analyze_trade_offs(evaluated_options)

        result = {
            "stage": "multifactor_evaluation",
            "evaluated_options": evaluated_options,
            "top_options": evaluated_options[:3],
            "sensitivity_analysis": sensitivity_analysis,
            "trade_off_analysis": trade_off_analysis,
            "evaluation_summary": {
                "best_option": evaluated_options[0] if evaluated_options else None,
                "score_range": (evaluated_options[-1]["overall_score"],
                                evaluated_options[0]["overall_score"]) if evaluated_options else (0, 0),
                "high_confidence_options": [opt for opt in evaluated_options if opt["confidence"] > 0.8]
            }
        }

        step = ReasoningStep(
            stage="Stage 4",
            step_id="multifactor_evaluation",
            description="Evaluate and compare options comprehensively",
            input_data=stage3_result,
            reasoning_process=reasoning_process,
            output_data=result,
            confidence=0.85,
            timestamp=datetime.now()
        )
        self.current_reasoning_chain.append(step)

        return result

    def _stage5_decision_formation(self, context: ReasoningContext,
                                   stage4_result: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 5: Decision Formation and Consequence Prediction"""

        reasoning_process = f"""
        STAGE 5: DECISION FORMATION AND CONSEQUENCE PREDICTION

        Finalizing decision based on comprehensive evaluation:

        DECISION CRITERIA:
        1. Highest overall score with sufficient confidence
        2. Acceptable risk level
        3. Alignment with personal values
        4. Feasibility confirmation
        5. Positive expected outcomes

        CONSEQUENCE PREDICTION:
        - Immediate outcomes and satisfaction
        - Impact on future activities and choices
        - Resource utilization effects
        - Social and relationship implications
        - Learning and experience value

        CONTINGENCY PLANNING:
        - Alternative options if primary choice fails
        - Adaptation strategies for unexpected situations
        - Monitoring and adjustment mechanisms

        DECISION CONFIDENCE FACTORS:
        - Evaluation robustness
        - Personal experience relevance
        - Constraint satisfaction certainty
        - Option quality and diversity
        """

        evaluated_options = stage4_result["evaluated_options"]

        if not evaluated_options:
            return {
                "stage": "decision_formation",
                "decision": None,
                "error": "No viable options available for decision"
            }

        # Select primary decision
        primary_decision = self._select_primary_decision(evaluated_options, context)

        # Predict consequences
        consequence_prediction = self._predict_consequences(primary_decision, context)

        # Develop contingency plan
        contingency_plan = self._develop_contingency_plan(evaluated_options, primary_decision)

        # Calculate decision confidence
        decision_confidence = self._calculate_decision_confidence(primary_decision, stage4_result)

        # Generate implementation plan
        implementation_plan = self._generate_implementation_plan(primary_decision, context)

        result = {
            "stage": "decision_formation",
            "primary_decision": primary_decision,
            "consequence_prediction": consequence_prediction,
            "contingency_plan": contingency_plan,
            "decision_confidence": decision_confidence,
            "implementation_plan": implementation_plan,
            "reasoning_summary": {
                "decision_rationale": self._generate_decision_rationale(primary_decision, stage4_result),
                "key_factors": self._identify_key_decision_factors(primary_decision),
                "risk_mitigation": self._identify_risk_mitigation_strategies(primary_decision, consequence_prediction)
            }
        }

        step = ReasoningStep(
            stage="Stage 5",
            step_id="decision_formation",
            description="Form final decision and predict consequences",
            input_data=stage4_result,
            reasoning_process=reasoning_process,
            output_data=result,
            confidence=decision_confidence,
            timestamp=datetime.now()
        )
        self.current_reasoning_chain.append(step)

        return result

    # Helper methods for constraint identification

    def _identify_temporal_constraints(self, context: ReasoningContext,
                                       stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        """Identify temporal constraints using memory and MCP tools"""

        temporal_constraints = {}

        # Get time-related preferences from memory
        if self.memory_manager:
            # Personal time preferences
            temporal_constraints["preferred_times"] = {
                "morning_preference": self.memory_manager.get_preference("morning_person", 0.5),
                "evening_preference": self.memory_manager.get_preference("evening_person", 0.5),
                "peak_hours": self.memory_manager.get_preference("peak_activity_hours", [12, 18, 20])
            }

            # Time allocation patterns
            temporal_constraints["time_allocation"] = {
                "max_daily_activities": self.memory_manager.get_preference("max_activities_per_day", 6),
                "preferred_break_duration": self.memory_manager.get_preference("preferred_break_duration", 30),
                "energy_pattern": self.memory_manager.get_preference("energy_pattern", "moderate")
            }

        # Current time constraints
        temporal_constraints["current_constraints"] = context.constraints.get('temporal', {})

        # Use temporal MCP tool for additional analysis
        if self.mcp_tool_manager:
            from mcp_tools.base_tool import MCPMessage

            query = MCPMessage(
                message_type="query",
                data={
                    "query_type": "feasibility_check",
                    "schedule": [],  # Current empty schedule for checking availability
                    "date": datetime.now().isoformat()
                },
                metadata={},
                timestamp=datetime.now(),
                message_id="temporal_constraint_check"
            )

            response = self.mcp_tool_manager.process_query("temporal", query)
            if response.metadata.get("status") == "success":
                temporal_constraints["feasibility_analysis"] = response.data

        return temporal_constraints

    def _identify_spatial_constraints(self, context: ReasoningContext,
                                      stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        """Identify spatial constraints using memory and MCP tools"""

        spatial_constraints = {}

        # Get spatial preferences from memory
        if self.memory_manager:
            spatial_constraints["preferences"] = {
                "max_travel_distance": self.memory_manager.get_preference("max_walking_distance", 1500),
                "transportation_preferences": {
                    mode: self.memory_manager.get_preference(f"transport_{mode}_preference", 0.5)
                    for mode in ["walking", "cycling", "public_transit", "car", "taxi"]
                },
                "exploration_tendency": self.memory_manager.get_preference("exploration_tendency", 0.5)
            }

        # Current location constraints
        current_location = context.current_situation.get('current_location')
        spatial_constraints["current_location"] = current_location
        spatial_constraints["location_constraints"] = context.constraints.get('spatial', {})

        # Use spatial MCP tool for accessibility analysis
        if self.mcp_tool_manager and current_location:
            from mcp_tools.base_tool import MCPMessage

            query = MCPMessage(
                message_type="query",
                data={
                    "query_type": "accessibility_analysis",
                    "location": current_location,
                    "activity_types": [stage1_result['problem_definition']]
                },
                metadata={},
                timestamp=datetime.now(),
                message_id="spatial_constraint_check"
            )

            response = self.mcp_tool_manager.process_query("spatial", query)
            if response.metadata.get("status") == "success":
                spatial_constraints["accessibility_analysis"] = response.data

        return spatial_constraints

    def _identify_resource_constraints(self, context: ReasoningContext,
                                       stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        """Identify resource constraints (financial, physical, social)"""

        resource_constraints = {}

        # Financial constraints from memory
        if self.memory_manager:
            activity_type = stage1_result['problem_definition']
            resource_constraints["financial"] = {
                "budget_range": self.memory_manager.get_preference(f"{activity_type}_budget_range", (0, 100)),
                "cost_sensitivity": self.memory_manager.get_preference("cost_sensitivity", 0.5)
            }

            # Physical constraints
            resource_constraints["physical"] = {
                "energy_level": context.current_situation.get('energy_level', 0.7),
                "physical_limitations": self.memory_manager.get_preference("physical_limitations", [])
            }

            # Social constraints
            resource_constraints["social"] = {
                "social_energy": self.memory_manager.get_preference("current_social_energy", 0.7),
                "available_companions": context.current_situation.get('available_companions', [])
            }

        return resource_constraints

    def _identify_personal_constraints(self, context: ReasoningContext,
                                       stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        """Identify personal preference and habit constraints"""

        personal_constraints = {}

        if self.memory_manager:
            activity_type = stage1_result['problem_definition']

            # Activity-specific preferences
            personal_constraints["activity_preferences"] = {
                "frequency_preference": self.memory_manager.get_preference(f"{activity_type}_frequency_preference",
                                                                           0.5),
                "social_preference": self.memory_manager.get_preference(f"{activity_type}_social_preference", 0.5),
                "duration_preference": self.memory_manager.get_preference(f"{activity_type}_preferred_duration", 60)
            }

            # General personal constraints
            personal_constraints["general"] = {
                "comfort_zone": self.memory_manager.get_preference("comfort_zone_preference", 0.6),
                "novelty_seeking": self.memory_manager.get_preference("novelty_seeking", 0.4),
                "planning_style": self.memory_manager.get_preference("planning_style", "moderate")
            }

        return personal_constraints

    def _identify_environmental_constraints(self, context: ReasoningContext,
                                            stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        """Identify environmental constraints using MCP tools"""

        environmental_constraints = {}

        # Get environmental sensitivity from memory
        if self.memory_manager:
            environmental_constraints["sensitivity"] = {
                "weather_sensitivity": self.memory_manager.get_preference("weather_sensitivity", 0.5),
                "crowd_tolerance": self.memory_manager.get_preference("crowd_tolerance", 0.5),
                "noise_tolerance": self.memory_manager.get_preference("noise_tolerance", 0.5)
            }

        # Use environmental MCP tool
        if self.mcp_tool_manager:
            from mcp_tools.base_tool import MCPMessage

            current_location = context.current_situation.get('current_location')
            if current_location:
                query = MCPMessage(
                    message_type="query",
                    data={
                        "query_type": "suitability_check",
                        "activity_type": stage1_result['problem_definition'],
                        "location": current_location,
                        "timestamp": datetime.now().isoformat()
                    },
                    metadata={},
                    timestamp=datetime.now(),
                    message_id="environmental_constraint_check"
                )

                response = self.mcp_tool_manager.process_query("environmental", query)
                if response.metadata.get("status") == "success":
                    environmental_constraints["current_conditions"] = response.data

        return environmental_constraints

    # Additional helper methods for option generation and evaluation

    def _generate_option_pool(self, context: ReasoningContext,
                              stage2_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate initial pool of options"""

        options = []
        activity_type = stage2_result.get('goal_clarification', {}).get('primary_goal', 'general_activity')

        # Memory-based options
        if self.memory_manager:
            relevant_memories = self.memory_manager.retrieve_relevant_memories({
                "activity_type": activity_type,
                "hour": datetime.now().hour,
                "location": context.current_situation.get('current_location')
            }, k=5)

            for memory in relevant_memories.get('events', []):
                option = {
                    "type": "memory_based",
                    "activity_type": memory.activity_type,
                    "location": memory.location,
                    "duration": memory.duration,
                    "companions": memory.companions,
                    "source": "personal_memory",
                    "confidence": 0.8,
                    "past_satisfaction": memory.satisfaction
                }
                options.append(option)

        # Spatial POI-based options
        if self.mcp_tool_manager:
            from mcp_tools.base_tool import MCPMessage

            current_location = context.current_situation.get('current_location')
            if current_location:
                query = MCPMessage(
                    message_type="query",
                    data={
                        "query_type": "poi_search",
                        "location": current_location,
                        "activity_type": activity_type,
                        "radius": 3000,
                        "limit": 10
                    },
                    metadata={},
                    timestamp=datetime.now(),
                    message_id="poi_option_generation"
                )

                response = self.mcp_tool_manager.process_query("spatial", query)
                if response.metadata.get("status") == "success":
                    for result in response.data.get("results", [])[:5]:
                        poi = result["poi"]
                        option = {
                            "type": "location_based",
                            "activity_type": activity_type,
                            "location": poi["location"],
                            "poi_info": poi,
                            "distance": result["distance"],
                            "source": "spatial_search",
                            "confidence": 0.7,
                            "estimated_satisfaction": result["score"]
                        }
                        options.append(option)

        # Preference-based synthetic options
        if self.memory_manager:
            preferred_duration = self.memory_manager.get_preference(f"{activity_type}_preferred_duration", 60)
            social_preference = self.memory_manager.get_preference(f"{activity_type}_social_preference", 0.5)

            synthetic_option = {
                "type": "preference_optimized",
                "activity_type": activity_type,
                "duration": preferred_duration,
                "social_setting": "group" if social_preference > 0.6 else "solo",
                "source": "preference_synthesis",
                "confidence": 0.6,
                "preference_alignment": 0.9
            }
            options.append(synthetic_option)

        return options

    def _preliminary_screening(self, options: List[Dict[str, Any]],
                               stage2_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply preliminary screening to filter options"""

        screened_options = []
        constraints = stage2_result["identified_constraints"]

        for option in options:
            # Check basic feasibility
            if self._passes_basic_feasibility(option, constraints):
                option["screening_passed"] = True
                option["screening_reasons"] = []
                screened_options.append(option)
            else:
                option["screening_passed"] = False
                option["screening_reasons"] = self._get_screening_failure_reasons(option, constraints)

        return screened_options

    def _passes_basic_feasibility(self, option: Dict[str, Any],
                                  constraints: Dict[str, Any]) -> bool:
        """Check if option passes basic feasibility screening"""

        # Temporal feasibility
        temporal_constraints = constraints.get("temporal", {})
        option_duration = option.get("duration", 60)
        max_duration = temporal_constraints.get("time_allocation", {}).get("max_activity_duration", 240)

        if option_duration > max_duration:
            return False

        # Spatial feasibility
        spatial_constraints = constraints.get("spatial", {})
        max_distance = spatial_constraints.get("preferences", {}).get("max_travel_distance", 5000)
        option_distance = option.get("distance", 0)

        if option_distance > max_distance:
            return False

        # Resource feasibility
        resource_constraints = constraints.get("resource", {})
        financial_budget = resource_constraints.get("financial", {}).get("budget_range", (0, 1000))
        option_cost = option.get("estimated_cost", 0)

        if option_cost > financial_budget[1]:
            return False

        return True

    def _comprehensive_option_evaluation(self, option: Dict[str, Any],
                                         context: ReasoningContext,
                                         stage3_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive evaluation of an option"""

        evaluation = {
            "preference_alignment": self._evaluate_preference_alignment(option, context),
            "constraint_satisfaction": self._evaluate_constraint_satisfaction(option, stage3_result),
            "expected_outcomes": self._evaluate_expected_outcomes(option, context),
            "risk_assessment": self._evaluate_risk_factors(option, context),
            "opportunity_cost": self._evaluate_opportunity_cost(option, context)
        }

        # Calculate weighted overall score
        weights = {
            "preference_alignment": 0.40,
            "constraint_satisfaction": 0.25,
            "expected_outcomes": 0.20,
            "risk_assessment": 0.10,
            "opportunity_cost": 0.05
        }

        overall_score = sum(
            evaluation[factor] * weights[factor]
            for factor in evaluation
        )

        evaluation["overall_score"] = overall_score
        evaluation["confidence"] = self._calculate_evaluation_confidence(evaluation)

        return evaluation

    def _evaluate_preference_alignment(self, option: Dict[str, Any],
                                       context: ReasoningContext) -> float:
        """Evaluate how well option aligns with personal preferences"""

        if not self.memory_manager:
            return 0.5

        activity_type = option.get("activity_type", "general")

        # Activity frequency preference
        freq_pref = self.memory_manager.get_preference(f"{activity_type}_frequency_preference", 0.5)

        # Social preference alignment
        social_pref = self.memory_manager.get_preference(f"{activity_type}_social_preference", 0.5)
        option_social = 1.0 if option.get("companions") or option.get("social_setting") == "group" else 0.0
        social_alignment = 1.0 - abs(social_pref - option_social)

        # Duration preference alignment
        preferred_duration = self.memory_manager.get_preference(f"{activity_type}_preferred_duration", 60)
        option_duration = option.get("duration", 60)
        duration_alignment = max(0, 1.0 - abs(option_duration - preferred_duration) / preferred_duration)

        # Location familiarity alignment
        exploration_tendency = self.memory_manager.get_preference("exploration_tendency", 0.5)
        option_familiarity = option.get("personal_familiarity", False)
        familiarity_alignment = exploration_tendency if not option_familiarity else (1.0 - exploration_tendency)

        # Weighted combination
        preference_score = (
                freq_pref * 0.3 +
                social_alignment * 0.3 +
                duration_alignment * 0.25 +
                familiarity_alignment * 0.15
        )

        return preference_score

    # Additional evaluation methods would continue here...

    def _load_reasoning_templates(self) -> Dict[str, str]:
        """Load reasoning templates for different scenarios"""

        return {
            "activity_selection": """
            When selecting an activity, consider:
            1. Personal preferences and past experiences
            2. Current constraints and limitations
            3. Expected satisfaction and outcomes
            4. Resource requirements and availability
            5. Alternative options and opportunity costs
            """,
            "location_choice": """
            When choosing a location, evaluate:
            1. Accessibility and transportation options
            2. Personal familiarity and comfort level
            3. Activity suitability and facilities
            4. Environmental conditions and ambiance
            5. Cost and value considerations
            """,
            "timing_decision": """
            When deciding on timing, consider:
            1. Personal energy and preference patterns
            2. Schedule constraints and conflicts
            3. Environmental factors (weather, crowds)
            4. Activity-specific optimal timing
            5. Integration with other planned activities
            """
        }

    def _assess_reasoning_quality(self) -> Dict[str, float]:
        """Assess the quality of the reasoning chain"""

        if not self.current_reasoning_chain:
            return {"overall_quality": 0.0}

        # Evaluate completeness
        expected_stages = 5
        completeness = len(self.current_reasoning_chain) / expected_stages

        # Evaluate consistency
        consistency = self._evaluate_chain_consistency()

        # Evaluate depth
        depth = self._evaluate_reasoning_depth()

        # Evaluate logical coherence
        coherence = self._evaluate_logical_coherence()

        overall_quality = (completeness * 0.3 + consistency * 0.25 +
                           depth * 0.25 + coherence * 0.2)

        return {
            "overall_quality": overall_quality,
            "completeness": completeness,
            "consistency": consistency,
            "depth": depth,
            "coherence": coherence
        }

    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence in the reasoning chain"""

        if not self.current_reasoning_chain:
            return 0.0

        confidences = [step.confidence for step in self.current_reasoning_chain]
        return sum(confidences) / len(confidences)

    # Additional helper methods would continue here...