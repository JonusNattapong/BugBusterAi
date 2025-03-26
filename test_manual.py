from core.ast_parser.parser import ASTParser
from core.knowledge_graph.builder import KnowledgeGraphBuilder
from core.mcts.search import MCTSBugSearch

# Sample code with intentional bug
sample_code = """
def divide(a, b):
    return a / b

def risky_operation():
    x = 0
    return divide(5, x)
"""

print("Testing AST Parser...")
parser = ASTParser()
ast_tree = parser.parse(sample_code)

print("\nFunction Definitions:")
for func in parser.get_function_definitions():
    print(f"- {func['name']} (lines: {func['lineno']})")

print("\nPotential Bugs:")
for bug in parser.detect_potential_bugs():
    print(f"- {bug['type']} at line {bug['lineno']}: {bug['message']}")

print("\nBuilding Knowledge Graph...")
builder = KnowledgeGraphBuilder()
kg = builder.build_from_source(sample_code)

print("\nRunning MCTS Bug Search...")
searcher = MCTSBugSearch(kg)
potential_bugs = searcher.search(max_iterations=100)  # Reduced for demo

print("\nFinal Bug Report:")
for i, bug in enumerate(potential_bugs, 1):
    print(f"Bug #{i}: {bug.get('type', 'Unknown')} at line {bug.get('lineno', '?')}")