# 基于 MCP 增强思维链大型语言模型的个体时空活动生成方法研究

## 项目结构
```text
mcp-cot-framework/
├── main.py                           # Main program entry
├── requirements.txt                  # Dependencies
├── config/
│   ├── models.yaml                   # Model configuration
│   ├── mcp_config.yaml              # MCP tools configuration
│   └── generation_config.yaml       # Generation parameters
├── models/
│   ├── __init__.py
│   ├── config.py                     # Model configuration management
│   ├── loader.py                     # Model loader
│   └── fine_tuned/                   # Fine-tuned models storage
│       ├── glm-4-9b-spatiotemporal/
│       ├── deepseek-r1-spatiotemporal/
│       └── ...
├── data/
│   ├── __init__.py
│   ├── data_processor.py             # Data processor
│   ├── knowledge/
│   │   └── spatiotemporal_behavior_alpaca.json
│   ├── local/
│   │   └── lujiazui_behavior_alpaca.json
│   ├── cot/
│   │   └── reasoning_examples_alpaca.json
│   ├── mcp/
│   │   └── tool_usage_alpaca.json
│   └── memory/
│       ├── personal_profiles.json    # Individual profile templates
│       └── memory_templates.json    # Memory initialization templates
├── training/
│   ├── __init__.py
│   ├── enhanced_lora_trainer.py
│   └── foundation_layer_manager.py
├── memory/
│   ├── __init__.py
│   ├── memory_manager.py             # Core memory management
│   ├── personal_memory.py            # Individual memory system
│   ├── memory_generator.py           # Memory initialization generator
│   └── memory_storage.py             # Memory persistence layer
├── mcp_tools/
│   ├── __init__.py
│   ├── base_tool.py                  # Base MCP tool class
│   ├── temporal_tools.py             # Temporal management tools
│   ├── spatial_tools.py              # Spatial navigation tools
│   ├── environmental_tools.py        # Environmental perception tools
│   ├── social_tools.py               # Social collaboration tools
│   ├── evaluation_tools.py           # Experience evaluation tools
│   └── tool_manager.py               # MCP tools coordinator
├── cot_reasoning/
│   ├── __init__.py
│   ├── reasoning_engine.py           # Core CoT reasoning engine
│   ├── cognitive_stages.py           # Five-stage cognitive framework
│   └── reasoning_templates.py        # CoT templates and prompts
├── generator/
│   ├── __init__.py
│   ├── behavior_generator.py         # Main behavior generation engine
│   ├── activity_chain_builder.py     # Activity chain construction
│   └── trajectory_validator.py       # Generated trajectory validation
├── parallel/
│   ├── __init__.py
│   ├── parallel_manager.py           # Parallel processing coordinator
│   ├── task_distributor.py           # Task distribution and scheduling
│   └── result_aggregator.py          # Results collection and aggregation
├── evaluation/
│   ├── __init__.py
│   ├── quality_evaluator.py          # Generation quality assessment
│   ├── diversity_evaluator.py        # Generation diversity assessment
│   ├── performance_evaluator.py      # Computational performance assessment
│   └── benchmark_comparator.py       # Benchmark comparison
└── utils/
    ├── __init__.py
    ├── geo_utils.py                  # Geographic utilities
    ├── time_utils.py                 # Time processing utilities
    ├── data_utils.py                 # Data processing utilities
    └── logging_utils.py              # Logging configuration
```