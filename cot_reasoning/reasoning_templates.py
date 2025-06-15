#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : reasoning_templates.py
# @Time    : 2025/6/15 15:34
# @Desc    : Templates and prompts for Chain-of-Thought reasoning


class ReasoningTemplates:
    """
    Templates for different types of reasoning scenarios
    """

    ACTIVITY_SELECTION_TEMPLATE = """
    ACTIVITY SELECTION REASONING:

    STAGE 1 - SITUATIONAL ANALYSIS:
    Current Context: {context}
    Available Time: {available_time}
    Location: {location}
    Personal State: {personal_state}

    STAGE 2 - CONSTRAINT IDENTIFICATION:
    Time Constraints: {time_constraints}
    Spatial Constraints: {spatial_constraints}
    Resource Constraints: {resource_constraints}
    Personal Preferences: {preferences}

    STAGE 3 - OPTION GENERATION:
    Potential Activities: {options}
    Screening Criteria: {screening_criteria}
    Viable Options: {viable_options}

    STAGE 4 - EVALUATION:
    Evaluation Criteria: {evaluation_criteria}
    Option Scores: {option_scores}
    Trade-offs: {tradeoffs}

    STAGE 5 - DECISION:
    Selected Activity: {selected_activity}
    Rationale: {rationale}
    Contingency Plan: {contingency}
    """

    LOCATION_SELECTION_TEMPLATE = """
    LOCATION SELECTION REASONING:

    STAGE 1 - SITUATIONAL ANALYSIS:
    Activity Type: {activity_type}
    Current Location: {current_location}
    Transportation Available: {transport_modes}
    Time Available: {time_available}

    STAGE 2 - CONSTRAINT IDENTIFICATION:
    Distance Tolerance: {distance_tolerance}
    Accessibility Requirements: {accessibility}
    Budget Constraints: {budget}
    Environmental Preferences: {env_preferences}

    STAGE 3 - OPTION GENERATION:
    Nearby Locations: {nearby_locations}
    Familiar Locations: {familiar_locations}
    Recommended Locations: {recommendations}

    STAGE 4 - EVALUATION:
    Accessibility Scores: {accessibility_scores}
    Preference Alignment: {preference_scores}
    Cost Analysis: {cost_analysis}

    STAGE 5 - DECISION:
    Selected Location: {selected_location}
    Route Plan: {route_plan}
    Backup Options: {backup_options}
    """

    TIMING_OPTIMIZATION_TEMPLATE = """
    TIMING OPTIMIZATION REASONING:

    STAGE 1 - SITUATIONAL ANALYSIS:
    Planned Activities: {planned_activities}
    Fixed Commitments: {fixed_commitments}
    Energy Levels: {energy_levels}
    External Factors: {external_factors}

    STAGE 2 - CONSTRAINT IDENTIFICATION:
    Time Windows: {time_windows}
    Sequence Dependencies: {dependencies}
    Travel Times: {travel_times}
    Personal Preferences: {timing_preferences}

    STAGE 3 - OPTION GENERATION:
    Possible Schedules: {schedule_options}
    Optimization Strategies: {optimization_strategies}

    STAGE 4 - EVALUATION:
    Efficiency Scores: {efficiency_scores}
    Preference Alignment: {preference_alignment}
    Risk Assessment: {risk_assessment}

    STAGE 5 - DECISION:
    Optimal Schedule: {optimal_schedule}
    Implementation Plan: {implementation_plan}
    Adjustment Mechanisms: {adjustments}
    """

    @classmethod
    def format_template(cls, template_type: str, **kwargs) -> str:
        """Format a template with provided variables"""

        templates = {
            "activity_selection": cls.ACTIVITY_SELECTION_TEMPLATE,
            "location_selection": cls.LOCATION_SELECTION_TEMPLATE,
            "timing_optimization": cls.TIMING_OPTIMIZATION_TEMPLATE
        }

        template = templates.get(template_type, "")

        try:
            return template.format(**kwargs)
        except KeyError as e:
            return f"Template formatting error: Missing variable {e}"

    @classmethod
    def get_reasoning_prompts(cls, reasoning_type: str) -> List[str]:
        """Get reasoning prompts for specific reasoning type"""

        prompts = {
            "activity_selection": [
                "What activities align with my current mood and energy level?",
                "Which options best satisfy my immediate and long-term goals?",
                "How do environmental factors influence my activity choices?",
                "What are the potential outcomes of each activity option?",
                "Which choice provides the best balance of satisfaction and feasibility?"
            ],
            "location_selection": [
                "Which locations are most accessible given my current situation?",
                "How do different locations align with my preferences and past experiences?",
                "What are the trade-offs between familiar and new locations?",
                "How do environmental conditions affect location suitability?",
                "Which location offers the best value for my intended activity?"
            ],
            "timing_optimization": [
                "When is my energy level most suitable for this activity?",
                "How do external factors influence optimal timing?",
                "What sequence of activities maximizes overall satisfaction?",
                "How can I balance efficiency with personal preferences?",
                "What contingencies should I plan for timing disruptions?"
            ]
        }

        return prompts.get(reasoning_type, [])

