import ast
import astor
from typing import Dict, List, Optional, Set
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ASTParser:
    """Enhanced AST parser with cross-function analysis."""

    def __init__(self):
        self.ast_tree = None
        self.source_code = ""
        self.var_assignments = {}  # Track variable assignments
        self.function_calls = {}  # Track function call relationships
        self.function_params = {}  # Track function parameters

    def parse(self, source_code: str) -> ast.AST:
        """Parse source code and build analysis structures."""
        self.source_code = source_code
        self.ast_tree = ast.parse(source_code)

        # Build analysis structures
        self._track_assignments()
        self._build_call_graph()
        return self.ast_tree

    def _track_assignments(self):
        """Track variable assignments with scope awareness."""
        current_scope = []

        def visit_node(node):
            if isinstance(node, ast.FunctionDef):
                current_scope.append(node.name)
                for child in ast.iter_child_nodes(node):
                    visit_node(child)
                current_scope.pop()
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        scope = current_scope[-1] if current_scope else "global"
                        self.var_assignments[var_name] = {
                            "lineno": node.lineno,
                            "value": self._get_node_value(node.value),
                            "scope": scope,
                        }
            else:
                for child in ast.iter_child_nodes(node):
                    visit_node(child)

        visit_node(self.ast_tree)

    def _build_call_graph(self):
        """Build enhanced call graph with parameter mapping."""
        self.function_calls = {}

        # First pass - collect function parameters
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef):
                params = [arg.arg for arg in node.args.args]
                self.function_params[node.name] = params

        # Second pass - track calls with parameter mapping
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                caller = self._get_enclosing_function(node)
                callee = node.func.id

                if caller not in self.function_calls:
                    self.function_calls[caller] = []

                # Map arguments to parameters
                param_map = {}
                if callee in self.function_params:
                    params = self.function_params[callee]
                    for i, arg in enumerate(node.args):
                        if i < len(params):
                            if isinstance(arg, ast.Constant):
                                param_map[params[i]] = arg.value
                            elif isinstance(arg, ast.Name):
                                param_map[params[i]] = arg.id

                self.function_calls[caller].append(
                    {
                        "callee": callee,
                        "args": node.args,
                        "param_map": param_map,
                        "lineno": node.lineno,
                    }
                )

    def _get_enclosing_function(self, node):
        """Get the name of the function enclosing this node."""
        for ancestor in ast.walk(self.ast_tree):
            if (
                isinstance(ancestor, ast.FunctionDef)
                and node.lineno >= ancestor.lineno
                and (not ancestor.body or node.lineno <= ancestor.body[-1].lineno)
            ):
                return ancestor.name
        return "global"

    def _get_node_value(self, node):
        """Get constant value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python <3.8
            return node.n
        return None

    def detect_potential_bugs(self) -> List[Dict]:
        """Detect bugs with cross-function analysis."""
        bugs = []

        for node in ast.walk(self.ast_tree):
            # Detect division by zero
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                if isinstance(node.right, ast.Constant) and node.right.value == 0:
                    bugs.append(
                        self._create_bug(
                            "ZeroDivision", node.lineno, "Direct division by zero"
                        )
                    )
                elif isinstance(node.right, ast.Name):
                    var_name = node.right.id
                    if self._is_zero_in_scope(var_name, node):
                        bugs.append(
                            self._create_bug(
                                "ZeroDivision",
                                node.lineno,
                                f"Division by {var_name} which may be zero",
                            )
                        )
        return bugs

    def _is_zero_in_scope(self, var_name: str, node: ast.AST) -> bool:
        """Check if variable may be zero in current context."""
        logger.debug(f"Checking if {var_name} might be zero at line {node.lineno}")

        # Check direct assignments
        if var_name in self.var_assignments:
            val = self.var_assignments[var_name]["value"]
            logger.debug(
                f"Found assignment: {val} in scope {self.var_assignments[var_name]['scope']}"
            )
            if val == 0:
                return True

        # Check function parameters and call chains
        func = self._get_enclosing_function(node)
        logger.debug(f"Current function scope: {func}")

        if func in self.function_calls:
            for call in self.function_calls[func]:
                if "param_map" in call and var_name in call["param_map"]:
                    param_value = call["param_map"][var_name]
                    logger.debug(f"Parameter {var_name} receives {param_value}")

                    # Check constant zero
                    if param_value == 0:
                        return True

                    # Check variable assignments
                    if isinstance(param_value, str):
                        # Check if assigned zero in caller
                        if (
                            param_value in self.var_assignments
                            and self.var_assignments[param_value]["value"] == 0
                        ):
                            return True

                        # Recursively check call chains
                        if self._is_zero_in_scope(param_value, node):
                            return True

        # Check if this is a parameter that might receive zero
        func = self._get_enclosing_function(node)
        if func in self.function_params and var_name in self.function_params[func]:
            # Find all calls to this function
            for caller, calls in self.function_calls.items():
                for call in calls:
                    if call["callee"] == func:
                        param_index = self.function_params[func].index(var_name)
                        if param_index < len(call["args"]):
                            arg = call["args"][param_index]
                            if isinstance(arg, ast.Constant) and arg.value == 0:
                                return True
                            elif isinstance(arg, ast.Name):
                                if self._is_zero_in_scope(arg.id, node):
                                    return True

        return False

    def _create_bug(self, bug_type: str, lineno: int, message: str) -> Dict:
        """Create a bug report dictionary."""
        return {"type": bug_type, "lineno": lineno, "message": message}

    def get_function_definitions(self) -> List[Dict]:
        """Get all function definitions in the code."""
        if not self.ast_tree:
            raise ValueError("No AST tree available. Call parse() first.")

        functions = []
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(
                    {
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "lineno": node.lineno,
                        "source": astor.to_source(node),
                    }
                )
        return functions

    def get_variable_usage(self) -> Dict[str, List[int]]:
        """Track variable usage across the code."""
        if not self.ast_tree:
            raise ValueError("No AST tree available. Call parse() first.")

        variables = {}
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Name):
                var_name = node.id
                if var_name not in variables:
                    variables[var_name] = []
                if isinstance(node.ctx, ast.Load):
                    variables[var_name].append(node.lineno)
                elif isinstance(node.ctx, ast.Store):
                    variables[var_name].append(node.lineno)

            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name not in variables:
                    variables[func_name] = []
                variables[func_name].append(node.lineno)

        return variables
