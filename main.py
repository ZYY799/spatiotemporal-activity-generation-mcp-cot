#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : main.py
# @Time    : 2025/6/15 15:23
# @Desc    : Main entry point for MCP-enhanced CoT spatiotemporal behavior generation


import argparse
import os
import sys
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory import PersonalMemoryManager, PersonalMemoryGenerator
from mcp_tools import MCPToolManager
from cot_reasoning import ChainOfThoughtEngine
from generator import SpatiotemporalBehaviorGenerator, PersonProfile
from parallel import ParallelGenerationManager
from evaluation import SpatiotemporalQualityEvaluator
from utils import DataUtils, TimeUtils, LoggingUtils


class MCPCoTApplication:
    """Main application class for spatiotemporal behavior generation"""

    def __init__(self, config_path: str = "config/generation_config.yaml"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.memory_generator = PersonalMemoryGenerator()
        self.quality_evaluator = SpatiotemporalQualityEvaluator()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            # Return default config if file not found
            return DataUtils.create_sample_config()
        except yaml.YAMLError as e:
            print(f"Error loading config: {e}")
            return DataUtils.create_sample_config()

    def _setup_logging(self) -> Any:
        """Setup logging configuration"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        log_file = f"spatiotemporal_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        return LoggingUtils.setup_logging(log_level, log_file)

    def generate_population_memories(self, num_people: int,
                                     output_dir: str = "data/generated") -> List[PersonProfile]:
        """Generate initial memory data for population"""

        self.logger.info(f"Generating memory data for {num_people} people...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate person profiles
        person_profiles = []

        for i in range(num_people):
            person_id = f"person_{i:04d}"

            # Generate profile
            profile_data = self.memory_generator.generate_person_profile(person_id)

            # Convert to PersonProfile object
            person_profile = PersonProfile(
                person_id=person_id,
                demographics=profile_data['demographics'],
                preferences=profile_data['activity_preferences'],
                constraints={
                    'temporal': profile_data['temporal_preferences'],
                    'spatial': {'max_distance': 20000},  # 20km max
                    'financial': {'daily_budget': 500}  # 500 RMB daily budget
                },
                spatial_anchors={
                    'home': profile_data['spatial_preferences']['home_location'],
                    'work': profile_data['spatial_preferences']['work_location']
                }
            )

            person_profiles.append(person_profile)

            # Generate and save memory
            memory_manager = self.memory_generator.generate_initial_memories(profile_data, num_days=30)
            memory_manager.save_memory()

            if (i + 1) % 10 == 0:
                self.logger.info(f"Generated memories for {i + 1}/{num_people} people")

        # Save profiles summary
        profiles_summary = {
            'profiles': [
                {
                    'person_id': p.person_id,
                    'demographics': p.demographics,
                    'spatial_anchors': p.spatial_anchors
                }
                for p in person_profiles
            ],
            'generation_timestamp': datetime.now().isoformat(),
            'total_people': len(person_profiles)
        }

        summary_path = os.path.join(output_dir, "population_profiles_summary.json")
        DataUtils.save_json(profiles_summary, summary_path)

        self.logger.info(f"Memory generation completed. Profiles saved to {summary_path}")

        return person_profiles

    def generate_trajectories(self, person_profiles: List[PersonProfile],
                              target_date: datetime,
                              output_dir: str = "output") -> List[Dict[str, Any]]:
        """Generate spatiotemporal trajectories for population"""

        self.logger.info(f"Starting trajectory generation for {len(person_profiles)} people on {target_date.date()}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Setup parallel generation manager
        parallel_config = self.config.get('parallel_processing', {})
        parallel_manager = ParallelGenerationManager(parallel_config)

        # Progress tracking
        start_time = datetime.now()

        def progress_callback(completed: int, total: int):
            LoggingUtils.log_generation_progress(self.logger, completed, total, start_time)

        # Generate trajectories
        trajectories = parallel_manager.generate_population_trajectories(
            person_profiles, target_date, progress_callback
        )

        # Log performance statistics
        performance_stats = parallel_manager.get_performance_summary()
        LoggingUtils.log_performance_stats(self.logger, performance_stats)

        # Convert trajectories to serializable format
        trajectory_data = []
        for trajectory in trajectories:
            trajectory_dict = {
                'person_id': trajectory.person_id,
                'date': trajectory.date.isoformat(),
                'activities': [
                    {
                        'activity_id': activity.activity_id,
                        'activity_type': activity.activity_type,
                        'start_time': activity.start_time.isoformat(),
                        'end_time': activity.end_time.isoformat(),
                        'duration': activity.duration,
                        'location': activity.location,
                        'location_name': activity.location_name,
                        'companions': activity.companions,
                        'transportation_mode': activity.transportation_mode,
                        'route_info': activity.route_info,
                        'satisfaction_prediction': activity.satisfaction_prediction,
                        'confidence': activity.confidence
                    }
                    for activity in trajectory.activities
                ],
                'total_active_time': trajectory.total_active_time,
                'total_travel_time': trajectory.total_travel_time,
                'trajectory_quality': trajectory.trajectory_quality,
                'generation_metadata': trajectory.generation_metadata
            }
            trajectory_data.append(trajectory_dict)

        # Save trajectories
        output_file = os.path.join(output_dir, f"trajectories_{target_date.strftime('%Y%m%d')}.json")
        DataUtils.save_json({
            'trajectories': trajectory_data,
            'generation_metadata': {
                'total_people': len(person_profiles),
                'target_date': target_date.isoformat(),
                'generation_time': datetime.now().isoformat(),
                'performance_stats': performance_stats
            }
        }, output_file)

        # Export to CSV for analysis
        csv_output = os.path.join(output_dir, f"trajectories_{target_date.strftime('%Y%m%d')}.csv")
        DataUtils.export_trajectories_csv(trajectory_data, csv_output)

        self.logger.info(f"Trajectory generation completed. Results saved to {output_file}")

        return trajectory_data

    def evaluate_quality(self, trajectory_data: List[Dict[str, Any]],
                         output_dir: str = "output") -> Dict[str, Any]:
        """Evaluate quality of generated trajectories"""

        self.logger.info(f"Evaluating quality of {len(trajectory_data)} trajectories...")

        # Convert trajectory data to DailyTrajectory objects for evaluation
        from generator import DailyTrajectory, ActivityInstance

        trajectories = []
        for traj_data in trajectory_data:
            activities = []
            for act_data in traj_data['activities']:
                activity = ActivityInstance(
                    activity_id=act_data['activity_id'],
                    activity_type=act_data['activity_type'],
                    start_time=TimeUtils.parse_time_string(act_data['start_time']),
                    end_time=TimeUtils.parse_time_string(act_data['end_time']),
                    duration=act_data['duration'],
                    location=tuple(act_data['location']) if act_data['location'] else (0, 0),
                    location_name=act_data['location_name'],
                    companions=act_data['companions'],
                    transportation_mode=act_data['transportation_mode'],
                    route_info=act_data['route_info'],
                    satisfaction_prediction=act_data['satisfaction_prediction'],
                    confidence=act_data['confidence'],
                    reasoning_chain=[]
                )
                activities.append(activity)

            trajectory = DailyTrajectory(
                person_id=traj_data['person_id'],
                date=TimeUtils.parse_time_string(traj_data['date']),
                activities=activities,
                total_active_time=traj_data['total_active_time'],
                total_travel_time=traj_data['total_travel_time'],
                trajectory_quality=traj_data['trajectory_quality'],
                generation_metadata=traj_data['generation_metadata']
            )
            trajectories.append(trajectory)

        # Evaluate quality
        quality_evaluation = self.quality_evaluator.evaluate_population_quality(trajectories)

        # Save evaluation results
        evaluation_file = os.path.join(output_dir, "quality_evaluation.json")
        DataUtils.save_json(quality_evaluation, evaluation_file)

        # Log summary
        summary = quality_evaluation.get('evaluation_summary', {})
        self.logger.info(f"Quality Evaluation Summary:")
        self.logger.info(f"  Overall Quality: {summary.get('overall_quality', 'unknown')}")
        self.logger.info(f"  Overall Score: {summary.get('overall_score', 0):.3f}")
        self.logger.info(f"  Diversity: {summary.get('diversity_assessment', 'unknown')}")

        if summary.get('key_strengths'):
            self.logger.info(f"  Strengths: {', '.join(summary['key_strengths'])}")

        if summary.get('areas_for_improvement'):
            self.logger.info(f"  Areas for Improvement: {', '.join(summary['areas_for_improvement'])}")

        return quality_evaluation

    def run_complete_pipeline(self, num_people: int, target_date: Optional[datetime] = None,
                              output_dir: str = "output") -> Dict[str, Any]:
        """Run complete generation and evaluation pipeline"""

        if target_date is None:
            target_date = datetime.now().date()
            target_date = datetime.combine(target_date, datetime.min.time())

        self.logger.info("Starting complete spatiotemporal behavior generation pipeline...")

        try:
            # Step 1: Generate population memories
            person_profiles = self.generate_population_memories(num_people, output_dir)

            # Step 2: Generate trajectories
            trajectory_data = self.generate_trajectories(person_profiles, target_date, output_dir)

            # Step 3: Evaluate quality
            quality_evaluation = self.evaluate_quality(trajectory_data, output_dir)

            # Step 4: Generate summary statistics
            statistics = DataUtils.aggregate_activity_statistics(trajectory_data)

            # Compile final results
            results = {
                'pipeline_summary': {
                    'num_people': num_people,
                    'target_date': target_date.isoformat(),
                    'total_trajectories': len(trajectory_data),
                    'completion_time': datetime.now().isoformat(),
                    'output_directory': output_dir
                },
                'quality_evaluation': quality_evaluation,
                'statistics': statistics,
                'success': True
            }

            # Save final summary
            summary_file = os.path.join(output_dir, "pipeline_summary.json")
            DataUtils.save_json(results, summary_file)

            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Results saved to: {output_dir}")

            return results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return {
                'pipeline_summary': {
                    'num_people': num_people,
                    'target_date': target_date.isoformat() if target_date else None,
                    'failure_time': datetime.now().isoformat(),
                    'error': str(e)
                },
                'success': False
            }


def main():
    """Main function for command-line interface"""

    parser = argparse.ArgumentParser(description='MCP-enhanced CoT Spatiotemporal Behavior Generation')

    parser.add_argument('--num-people', type=int, default=50,
                        help='Number of people to generate trajectories for (default: 50)')

    parser.add_argument('--target-date', type=str, default=None,
                        help='Target date for trajectory generation (YYYY-MM-DD format, default: today)')

    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for results (default: output)')

    parser.add_argument('--config', type=str, default='config/generation_config.yaml',
                        help='Configuration file path (default: config/generation_config.yaml)')

    parser.add_argument('--mode', type=str, choices=['memory', 'generate', 'evaluate', 'complete'],
                        default='complete',
                        help='Operation mode (default: complete)')

    args = parser.parse_args()

    # Parse target date
    target_date = None
    if args.target_date:
        try:
            target_date = datetime.strptime(args.target_date, '%Y-%m-%d')
        except ValueError:
            print(f"Invalid date format: {args.target_date}. Use YYYY-MM-DD format.")
            return 1

    # Initialize application
    try:
        app = MCPCoTApplication(args.config)
    except Exception as e:
        print(f"Failed to initialize application: {e}")
        return 1

    # Run based on mode
    try:
        if args.mode == 'memory':
            app.generate_population_memories(args.num_people, args.output_dir)

        elif args.mode == 'generate':
            # Load existing profiles or generate new ones
            profiles_file = os.path.join(args.output_dir, "population_profiles_summary.json")
            if os.path.exists(profiles_file):
                # Load existing profiles (simplified loading)
                print("Loading existing profiles not fully implemented. Generating new ones...")
                person_profiles = app.generate_population_memories(args.num_people, args.output_dir)
            else:
                person_profiles = app.generate_population_memories(args.num_people, args.output_dir)

            if target_date is None:
                target_date = datetime.now()

            app.generate_trajectories(person_profiles, target_date, args.output_dir)

        elif args.mode == 'evaluate':
            # Load existing trajectory data
            trajectories_file = None
            if args.target_date:
                date_str = datetime.strptime(args.target_date, '%Y-%m-%d').strftime('%Y%m%d')
                trajectories_file = os.path.join(args.output_dir, f"trajectories_{date_str}.json")

            if trajectories_file and os.path.exists(trajectories_file):
                trajectory_data = DataUtils.load_json(trajectories_file)['trajectories']
                app.evaluate_quality(trajectory_data, args.output_dir)
            else:
                print("No trajectory data found for evaluation.")
                return 1

        elif args.mode == 'complete':
            results = app.run_complete_pipeline(args.num_people, target_date, args.output_dir)

            if results['success']:
                print("Pipeline completed successfully!")
                summary = results['quality_evaluation']['evaluation_summary']
                print(f"Overall Quality: {summary.get('overall_quality', 'unknown')}")
                print(f"Overall Score: {summary.get('overall_score', 0):.3f}")
            else:
                print("Pipeline failed. Check logs for details.")
                return 1

        return 0

    except Exception as e:
        print(f"Execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)