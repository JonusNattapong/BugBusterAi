import argparse
from pathlib import Path
from core.ast_parser.parser import ASTParser
from core.knowledge_graph.builder import KnowledgeGraphBuilder
from core.mcts.search import MCTSBugSearch

def analyze_file(file_path: str):
    """Analyze a Python file for bugs."""
    print(f"Analyzing {file_path}...")
    
    # Read and parse the file
    with open(file_path, 'r') as f:
        code = f.read()
    
    # Run through analysis pipeline
    parser = ASTParser()
    ast_tree = parser.parse(code)
    
    builder = KnowledgeGraphBuilder()
    kg = builder.build_from_source(code)
    
    searcher = MCTSBugSearch(kg)
    potential_bugs = searcher.search()
    
    # Display results
    print("\nAnalysis Results:")
    print("=" * 40)
    print(f"Found {len(potential_bugs)} potential bugs")
    
    for i, bug in enumerate(potential_bugs, 1):
        print(f"\nBug #{i}:")
        print(f"Type: {bug.get('type', 'Unknown')}")
        print(f"Location: Line {bug.get('lineno', '?')}")
        print(f"Description: {bug.get('message', 'No description')}")
        print("-" * 20)

def main():
    parser = argparse.ArgumentParser(description='BugBuster AI - Code Analysis Tool')
    parser.add_argument('file', help='Python file to analyze')
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"Error: File {args.file} not found")
        return
    
    analyze_file(args.file)

if __name__ == "__main__":
    main()