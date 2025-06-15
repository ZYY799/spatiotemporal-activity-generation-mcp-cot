# MCP-Enhanced Chain-of-Thought Spatiotemporal Behavior Generation Framework

A comprehensive framework for generating realistic individual spatiotemporal behaviors using Model Context Protocol (MCP) enhanced Chain-of-Thought reasoning with Large Language Models.

---
**Disclaimer:**

Due to time constraints and limited personal bandwidth, portions of this codebase have not been fully organized into a production-ready, open-source project. The current repository primarily represents a structural reorganization of previously written code, which may contain unforeseen bugs and some unmodified logic implementations.

**Important Notes:**

- This code is provided **for reference purposes only**
- Some methods and implementations may not reflect the latest updates
- Potential bugs and logic inconsistencies may exist throughout the codebase
- For accurate functionality demonstration, please refer to the live demo site: [demo website](http://117.72.208.136/)
- Due to the non-deterministic nature of training data and LLM outputs, some features may not be fully reproducible
- Actual performance should be evaluated through your own deployment and testing
- The MCP (Model Context Protocol) components will be open-sourced and deployed in subsequent releases

**Recommendation:**

Please conduct thorough testing in your own environment before any use.

## Overview

This framework implements the methodology described in "A Study on Individual Spatiotemporal Activity Generation Method Using MCP-Enhanced Chain-of-Thought Large Language Models" for generating authentic human activity-travel patterns in urban environments.

### Key Features

- **MCP-Enhanced Tool System**: Six specialized tool categories for comprehensive spatiotemporal reasoning
- **Five-Stage CoT Reasoning**: Systematic decision-making process mimicking human cognition
- **Personal Memory System**: Three-tier memory architecture with dynamic consolidation
- **Parallel Processing**: Scalable generation for large populations
- **Quality Evaluation**: Comprehensive assessment across multiple dimensions
- **Zero-Shot Deployment**: Works without additional training data

## Architecture

```
mcp-cot-framework/
├── main.py                           # Main program entry
├── requirements.txt                  # Dependencies
├── config/
│   ├── models.yaml                   # Model configuration
│   ├── mcp_config.yaml              # MCP tools configuration
│   └── generation_config.yaml       # Generation parameters
├── models/                           # Model management
├── data/                            # Data and knowledge bases
├── memory/                          # Personal memory system
├── mcp_tools/                       # MCP tool categories
├── cot_reasoning/                   # Chain-of-thought engine
├── generator/                       # Behavior generation engine
├── parallel/                        # Parallel processing system
├── evaluation/                      # Quality evaluation system
└── utils/                          # Utility functions
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/mcp-cot-framework.git
cd mcp-cot-framework

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 2. Basic Usage

Generate spatiotemporal behaviors for 50 people:

```bash
# Run complete pipeline
python main.py --num-people 50 --mode complete

# Generate only memory data
python main.py --num-people 50 --mode memory

# Generate trajectories for specific date
python main.py --num-people 50 --target-date 2024-01-15 --mode generate

# Evaluate existing trajectories
python main.py --mode evaluate --target-date 2024-01-15
```

### 3. Python API Usage

```python
from main import MCPCoTApplication
from datetime import datetime

# Initialize application
app = MCPCoTApplication("config/generation_config.yaml")

# Generate population memories
person_profiles = app.generate_population_memories(num_people=100)

# Generate trajectories
target_date = datetime(2024, 1, 15)
trajectories = app.generate_trajectories(person_profiles, target_date)

# Evaluate quality
quality_results = app.evaluate_quality(trajectories)

print(f"Generated {len(trajectories)} trajectories")
print(f"Quality score: {quality_results['evaluation_summary']['overall_score']:.3f}")
```

## Core Components

### Memory System

Three-tier memory architecture:
- **Event Memory**: Specific activity experiences
- **Pattern Memory**: Behavioral regularities
- **Summary Memory**: Abstract preferences and habits

```python
from memory import PersonalMemoryManager

memory_manager = PersonalMemoryManager("person_001")
memories = memory_manager.retrieve_relevant_memories(context, k=5)
```

### MCP Tools

Six specialized tool categories:
- **Temporal Management**: Time-related queries and scheduling
- **Spatial Navigation**: Location search and route planning
- **Environmental Perception**: Weather and environmental conditions
- **Social Collaboration**: Social interaction and group coordination
- **Experience Evaluation**: Activity and location assessment
- **Personal Memory**: Individual memory management

```python
from mcp_tools import MCPToolManager, MCPMessage

tool_manager = MCPToolManager(memory_manager)
query = MCPMessage(message_type="query", data={"query_type": "poi_search"})
response = tool_manager.process_query("spatial", query)
```

### Chain-of-Thought Reasoning

Five-stage cognitive framework:
1. **Situational Awareness**: Analyze current context
2. **Constraint Identification**: Identify limitations and goals
3. **Option Generation**: Create and screen alternatives
4. **Multi-factor Evaluation**: Comprehensive assessment
5. **Decision Formation**: Final decision and contingency planning

```python
from cot_reasoning import ChainOfThoughtEngine, ReasoningContext

reasoning_engine = ChainOfThoughtEngine(tool_manager, memory_manager)
result = reasoning_engine.execute_reasoning_chain(context, "select_lunch_location")
```

### Behavior Generation

Main generation engine integrating all components:

```python
from generator import SpatiotemporalBehaviorGenerator, PersonProfile

generator = SpatiotemporalBehaviorGenerator(
    model=model,
    memory_manager=memory_manager,
    mcp_tool_manager=tool_manager,
    config=config
)

trajectory = generator.generate_daily_trajectory(person_profile, target_date)
```

## Configuration

### Generation Configuration

Configure generation parameters in `config/generation_config.yaml`:

```yaml
generation:
  model_settings:
    temperature: 0.7
    max_tokens: 9216
  
  behavior_generation:
    max_activities_per_day: 8
    min_activity_duration: 15
    daily_time_window: [5, 23]
  
  parallel_processing:
    max_workers: 8
    batch_size: 10
    memory_limit_gb: 8
```

### MCP Tools Configuration

Configure MCP tools in `config/mcp_config.yaml`:

```yaml
mcp_tools:
  temporal_management:
    enabled: true
    parameters:
      max_daily_activities: 8
      min_activity_duration: 15
  
  spatial_navigation:
    enabled: true
    parameters:
      default_search_radius: 3000
      max_travel_distance: 50000
```

## Evaluation

The framework provides comprehensive quality evaluation:

- **Temporal Consistency**: Logical temporal ordering
- **Activity Coherence**: Logical activity sequences
- **Spatial Realism**: Realistic spatial patterns
- **Behavioral Authenticity**: Authentic human patterns
- **Constraint Satisfaction**: Adherence to constraints

Quality scores range from 0-10 with classifications:
- Excellent: 8.5-10
- Good: 7.0-8.4
- Average: 5.5-6.9
- Poor: 4.0-5.4
- Very Poor: 0-3.9

## Performance

Parallel processing capabilities:
- **Scalable**: Up to 12 parallel workers tested
- **Efficient**: 0.17 minutes per sample at 12 workers
- **Memory Optimized**: Configurable memory limits
- **Error Tolerant**: Graceful failure handling

Performance metrics:
- Generation speed: 77% reduction (2→12 workers)
- Memory usage: Scales predictably
- Success rate: >95% under normal conditions

## Use Cases

1. **Urban Planning**: Simulate pedestrian flows and activity patterns
2. **Transportation Management**: Predict mobility demands
3. **Commercial Site Selection**: Understand customer behavior patterns
4. **Smart City Development**: Model citizen activity needs
5. **Emergency Planning**: Simulate population responses
6. **Social Research**: Study behavioral patterns and trends

## Data Requirements

### Minimum Data

- Geographic boundaries (POI locations)
- Basic activity type definitions
- Temporal constraints (business hours, etc.)

### Enhanced Data

- Historical mobile signaling data
- Activity diary surveys
- POI ratings and reviews
- Transportation network data
- Weather and event data

## Extending the Framework

### Adding New MCP Tools

1. Inherit from `BaseMCPTool`
2. Implement required methods
3. Register with `MCPToolManager`

```python
from mcp_tools.base_tool import BaseMCPTool

class CustomTool(BaseMCPTool):
    def __init__(self, memory_manager=None):
        super().__init__("custom_tool", memory_manager)
    
    def process_query(self, query):
        # Implementation here
        pass
    
    def get_capabilities(self):
        return ["custom_capability"]
```

### Custom Reasoning Stages

Extend the CoT reasoning engine:

```python
from cot_reasoning import ChainOfThoughtEngine

class CustomReasoningEngine(ChainOfThoughtEngine):
    def custom_reasoning_stage(self, context, previous_result):
        # Custom reasoning logic
        pass
```

### Custom Evaluation Metrics

Add custom quality metrics:

```python
from evaluation import SpatiotemporalQualityEvaluator

class CustomEvaluator(SpatiotemporalQualityEvaluator):
    def evaluate_custom_metric(self, trajectory):
        # Custom evaluation logic
        pass
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or max workers
2. **Generation Failures**: Check model configuration and API access
3. **Quality Issues**: Verify knowledge base and memory initialization
4. **Performance Issues**: Optimize parallel processing settings

### Debug Mode

Enable detailed logging:

```python
app = MCPCoTApplication("config/generation_config.yaml")
app.config['logging'] = {'level': 'DEBUG'}
```

### Memory Management

Monitor memory usage:

```bash
# Check memory usage during generation
python main.py --num-people 100 --mode complete --config config/low_memory_config.yaml
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{zhang2025studyindividualspatiotemporalactivity,
      title={A Study on Individual Spatiotemporal Activity Generation Method Using MCP-Enhanced Chain-of-Thought Large Language Models}, 
      author={Yu Zhang and Yang Hu and De Wang},
      year={2025},
      eprint={2506.10853},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.10853}, 
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgments

- Thanks to all contributors and researchers in spatiotemporal behavior modeling
- Built upon foundational work in time geography and activity-based modeling
- Leverages advances in large language models and chain-of-thought reasoning
- Grateful to our advisor and lab mates for their invaluable guidance.

---
