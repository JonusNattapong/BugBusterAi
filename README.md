# BugBuster AI - Advanced Code Bug Detection System

## Overview
BugBuster AI is an advanced system that combines Enhanced MCTS with Neural Networks for multi-level temporal reasoning to detect, localize, and recommend fixes for code bugs.

## Current Implementation Status
The core framework is complete with:
- ✅ AST parser for code analysis
- ✅ Knowledge graph for code relationships  
- ✅ MCTS algorithm for bug search
- ✅ Neural network models (policy, value, fix generator)
- ✅ Command line interface
- ✅ Basic testing framework

## Key Features
- Multi-level bug detection (syntax → logic → runtime → performance)
- Knowledge Graph integration for code relationship mapping
- Temporal reasoning for execution flow modeling
- Adaptive MCTS for efficient bug search
- Neural networks for bug prediction and fix generation

## Project Structure
```
BugBusterAI/
├── core/               # Core analysis components
│   ├── ast_parser/     # AST processing (implemented)
│   ├── knowledge_graph/ # Code relationship mapping (implemented)  
│   ├── mcts/           # Monte Carlo Tree Search (basic implementation)
│   └── neural_nets/    # Neural network models (defined)
├── interfaces/         # System interfaces
│   ├── cli/            # Command line interface (implemented)
│   └── ide/            # IDE integration components (structure)
├── tests/              # Test cases (basic tests)
├── requirements.txt    # Python dependencies
└── setup.py            # Project setup
```

## Getting Started
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the CLI: `python -m interfaces.cli.main path/to/file.py`

## Example Usage
```python
# Analyze a Python file
from core.ast_parser.parser import ASTParser

parser = ASTParser()
with open('example.py') as f:
    ast = parser.parse(f.read())
    bugs = parser.detect_potential_bugs()
    print(f"Found {len(bugs)} potential bugs")
```

## Roadmap
- [x] Core framework implementation
- [ ] Enhanced bug detection patterns
- [ ] MCTS heuristic improvements  
- [ ] Neural network training
- [ ] IDE plugin development
- [ ] Multi-language support

## Contributing
1. Fork the repository
2. Create a feature branch
3. Submit a pull request
4. Include tests for new functionality

## License
MIT License (see LICENSE file)