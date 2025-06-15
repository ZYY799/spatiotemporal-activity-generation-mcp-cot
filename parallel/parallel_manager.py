#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : parallel_manager.py
# @Time    : 2025/6/15 15:36
# @Desc    : Parallel processing manager for large-scale spatiotemporal behavior generation


import multiprocessing as mp
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable
import time
import queue
import psutil
import pickle
from datetime import datetime
import logging

from generator import SpatiotemporalBehaviorGenerator, PersonProfile, DailyTrajectory
from memory import PersonalMemoryManager
from mcp_tools import MCPToolManager


class ParallelGenerationManager:
    """
    Manager for parallel spatiotemporal behavior generation with optimized resource usage
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_workers = config.get("max_workers", mp.cpu_count())
        self.batch_size = config.get("batch_size", 10)
        self.memory_limit = config.get("memory_limit_gb", 8) * 1024 * 1024 * 1024  # Convert to bytes

        # Performance monitoring
        self.performance_stats = {
            "total_generated": 0,
            "generation_times": [],
            "memory_usage": [],
            "error_count": 0,
            "worker_utilization": {}
        }

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def generate_population_trajectories(self,
                                         person_profiles: List[PersonProfile],
                                         target_date: datetime,
                                         progress_callback: Optional[Callable] = None) -> List[DailyTrajectory]:
        """Generate trajectories for entire population using parallel processing"""

        start_time = time.time()
        self.logger.info(f"Starting parallel generation for {len(person_profiles)} people")

        # Divide work into batches
        batches = self._create_batches(person_profiles)

        # Initialize result collection
        all_trajectories = []
        completed_count = 0

        # Choose execution strategy based on workload
        if len(person_profiles) <= 50:
            # Use thread-based parallelism for smaller workloads
            trajectories = self._generate_with_threads(batches, target_date, progress_callback)
        else:
            # Use process-based parallelism for larger workloads
            trajectories = self._generate_with_processes(batches, target_date, progress_callback)

        # Collect and validate results
        for trajectory_batch in trajectories:
            if trajectory_batch:
                all_trajectories.extend(trajectory_batch)
                completed_count += len(trajectory_batch)

                if progress_callback:
                    progress_callback(completed_count, len(person_profiles))

        # Record performance statistics
        end_time = time.time()
        self.performance_stats["total_generated"] = len(all_trajectories)
        self.performance_stats["total_time"] = end_time - start_time

        self.logger.info(f"Completed generation: {len(all_trajectories)} trajectories in {end_time - start_time:.2f}s")

        return all_trajectories

    def _create_batches(self, person_profiles: List[PersonProfile]) -> List[List[PersonProfile]]:
        """Create optimized batches for parallel processing"""

        batches = []
        current_batch = []

        for i, profile in enumerate(person_profiles):
            current_batch.append(profile)

            # Create batch when batch_size reached or at end of list
            if len(current_batch) >= self.batch_size or i == len(person_profiles) - 1:
                batches.append(current_batch)
                current_batch = []

        self.logger.info(f"Created {len(batches)} batches with batch_size={self.batch_size}")
        return batches

    def _generate_with_threads(self, batches: List[List[PersonProfile]],
                               target_date: datetime,
                               progress_callback: Optional[Callable] = None) -> List[List[DailyTrajectory]]:
        """Generate trajectories using thread-based parallelism"""

        results = []

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batches))) as executor:
            # Submit batch processing tasks
            future_to_batch = {
                executor.submit(self._process_batch_thread, batch, target_date): batch
                for batch in batches
            }

            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_result = future.result(timeout=300)  # 5 minute timeout per batch
                    results.append(batch_result)

                    # Monitor memory usage
                    self._monitor_memory_usage()

                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")
                    self.performance_stats["error_count"] += 1
                    results.append([])  # Empty result for failed batch

        return results

    def _generate_with_processes(self, batches: List[List[PersonProfile]],
                                 target_date: datetime,
                                 progress_callback: Optional[Callable] = None) -> List[List[DailyTrajectory]]:
        """Generate trajectories using process-based parallelism"""

        results = []

        # Serialize configuration for worker processes
        serialized_config = pickle.dumps(self.config)

        with ProcessPoolExecutor(max_workers=min(self.max_workers, len(batches))) as executor:
            # Submit batch processing tasks
            future_to_batch = {
                executor.submit(
                    _process_batch_worker,
                    batch,
                    target_date,
                    serialized_config
                ): batch
                for batch in batches
            }

            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_result = future.result(timeout=600)  # 10 minute timeout per batch
                    results.append(batch_result)

                    # Monitor system resources
                    self._monitor_system_resources()

                except Exception as e:
                    self.logger.error(f"Process batch failed: {e}")
                    self.performance_stats["error_count"] += 1
                    results.append([])  # Empty result for failed batch

        return results

    def _process_batch_thread(self, batch: List[PersonProfile],
                              target_date: datetime) -> List[DailyTrajectory]:
        """Process a batch of profiles in a thread"""

        batch_start_time = time.time()
        results = []

        # Initialize components for this thread
        try:
            # Note: In thread-based execution, we share the same model instance
            # This requires careful memory management and thread safety

            for profile in batch:
                try:
                    # Initialize memory manager for this person
                    memory_manager = PersonalMemoryManager(profile.person_id)

                    # Initialize MCP tool manager
                    mcp_tool_manager = MCPToolManager(memory_manager)

                    # Create generator (would need to be thread-safe in practice)
                    generator = SpatiotemporalBehaviorGenerator(
                        model=None,  # Model would be shared or thread-local
                        memory_manager=memory_manager,
                        mcp_tool_manager=mcp_tool_manager,
                        config=self.config
                    )

                    # Generate trajectory
                    trajectory = generator.generate_daily_trajectory(profile, target_date)
                    results.append(trajectory)

                except Exception as e:
                    self.logger.error(f"Failed to generate trajectory for {profile.person_id}: {e}")
                    # Create empty trajectory for failed generation
                    empty_trajectory = DailyTrajectory(
                        person_id=profile.person_id,
                        date=target_date,
                        activities=[],
                        total_active_time=0,
                        total_travel_time=0,
                        trajectory_quality=0.0,
                        generation_metadata={"error": str(e), "status": "failed"}
                    )
                    results.append(empty_trajectory)

            batch_time = time.time() - batch_start_time
            self.performance_stats["generation_times"].append(batch_time)

        except Exception as e:
            self.logger.error(f"Batch processing failed completely: {e}")
            return []

        return results

    def _monitor_memory_usage(self) -> None:
        """Monitor current memory usage"""

        memory_info = psutil.Process().memory_info()
        memory_usage = memory_info.rss  # Resident Set Size

        self.performance_stats["memory_usage"].append(memory_usage)

        # Warning if approaching memory limit
        if memory_usage > self.memory_limit * 0.8:
            self.logger.warning(f"High memory usage: {memory_usage / 1024 / 1024:.1f} MB")

    def _monitor_system_resources(self) -> None:
        """Monitor system-wide resource usage"""

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()

        # Update performance stats
        if "system_stats" not in self.performance_stats:
            self.performance_stats["system_stats"] = []

        self.performance_stats["system_stats"].append({
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available": memory.available
        })

        # Log warnings for high resource usage
        if cpu_percent > 90:
            self.logger.warning(f"High CPU usage: {cpu_percent}%")

        if memory.percent > 85:
            self.logger.warning(f"High memory usage: {memory.percent}%")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance statistics"""

        summary = {
            "total_generated": self.performance_stats["total_generated"],
            "total_time": self.performance_stats.get("total_time", 0),
            "error_count": self.performance_stats["error_count"],
            "success_rate": 0,
            "average_generation_time": 0,
            "peak_memory_usage": 0,
            "throughput": 0
        }

        # Calculate success rate
        if self.performance_stats["total_generated"] > 0:
            summary["success_rate"] = (
                    (self.performance_stats["total_generated"] - self.performance_stats["error_count"]) /
                    self.performance_stats["total_generated"]
            )

        # Calculate average generation time
        if self.performance_stats["generation_times"]:
            summary["average_generation_time"] = (
                    sum(self.performance_stats["generation_times"]) /
                    len(self.performance_stats["generation_times"])
            )

        # Calculate peak memory usage
        if self.performance_stats["memory_usage"]:
            summary["peak_memory_usage"] = max(self.performance_stats["memory_usage"])

        # Calculate throughput (trajectories per minute)
        if summary["total_time"] > 0:
            summary["throughput"] = (
                    self.performance_stats["total_generated"] / (summary["total_time"] / 60)
            )

        return summary


def _process_batch_worker(batch: List[PersonProfile],
                          target_date: datetime,
                          serialized_config: bytes) -> List[DailyTrajectory]:
    """Worker function for process-based batch processing"""

    # Deserialize configuration
    config = pickle.loads(serialized_config)

    results = []

    for profile in batch:
        try:
            # Initialize components for this worker process
            memory_manager = PersonalMemoryManager(profile.person_id)
            mcp_tool_manager = MCPToolManager(memory_manager)

            # Create generator
            generator = SpatiotemporalBehaviorGenerator(
                model=None,  # Model would be initialized per process
                memory_manager=memory_manager,
                mcp_tool_manager=mcp_tool_manager,
                config=config
            )

            # Generate trajectory
            trajectory = generator.generate_daily_trajectory(profile, target_date)
            results.append(trajectory)

        except Exception as e:
            # Create empty trajectory for failed generation
            empty_trajectory = DailyTrajectory(
                person_id=profile.person_id,
                date=target_date,
                activities=[],
                total_active_time=0,
                total_travel_time=0,
                trajectory_quality=0.0,
                generation_metadata={"error": str(e), "status": "failed"}
            )
            results.append(empty_trajectory)

    return results