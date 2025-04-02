# BugBusterAI Documentation

## Table of Contents
- [Installation](#installation)
- [Configuration](#configuration)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [API Reference](#api-reference)

## Installation
```bash
# Install with GPU support (recommended)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
python setup.py install
```

## Configuration
Create `config.ini`:
```ini
[analysis]
parallel_workers = 4
incremental = true
gpu_acceleration = true

[database]
knowledge_graph_path = ./knowledge.db
```

## Basic Usage
### Command Line
```bash
# Analyze single file
bugbuster analyze example.py

# Analyze project directory
bugbuster analyze project/ --incremental
```

### Python API
```python
from bugbuster import BugBuster

analyzer = BugBuster(
    parallel_workers=4,
    incremental=True,
    gpu=True
)

results = analyzer.analyze("project/")
results.save_report("report.html")
```

## Advanced Features
### Incremental Analysis
```python
# First run (full analysis)
analyzer.analyze("project/", incremental=False)

# Subsequent runs (only analyze changes)
analyzer.analyze("project/", incremental=True)
```

### Custom Bug Patterns
```python
from bugbuster.patterns import register_pattern

@register_pattern
def check_null_comparison(node):
    # Custom bug detection logic
    pass
```

## API Reference
### Core Classes
- `ASTParser`: Parallel code parser with type inference
- `KnowledgeGraph`: Persistent code relationship storage
- `MCTSBugSearch`: Neural-guided bug search
- `BugBuster`: Main analysis interface

### Methods
```python
analyze(path, incremental=False, use_gpu=True)
get_bugs(severity=None)
save_knowledge_graph(path)
generate_report(format='html')