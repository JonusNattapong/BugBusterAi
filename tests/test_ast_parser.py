import unittest
from core.ast_parser.parser import ASTParser

class TestASTParser(unittest.TestCase):
    def setUp(self):
        self.parser = ASTParser()
        self.sample_code = """
def divide(a, b):
    return a / b  # Potential division by zero

def risky_operation():
    x = 0
    y = divide(5, x)  # Calls divide with x=0
    return y

def compare_none(value):
    if value == None:  # Bad comparison
        return True
    return False

def empty_except():
    try:
        risky_operation()
    except:
        pass  # Empty except block

unused_var = 42  # Never used
used_var = 10
print(used_var)  # Actually used

def nested_zero():
    z = 0
    def inner_func():
        return divide(1, z)  # Nested scope zero
    return inner_func()
"""

    def test_function_detection(self):
        self.parser.parse(self.sample_code)
        functions = self.parser.get_function_definitions()
        self.assertEqual(len(functions), 6)  # Includes nested_zero()
        self.assertEqual(functions[0]['name'], 'divide')
        self.assertEqual(functions[1]['name'], 'risky_operation')

    def test_bug_detection(self):
        self.parser.parse(self.sample_code)
        bugs = self.parser.detect_potential_bugs()
        
        # Should detect all bug types including cross-function ZeroDivision
        bug_types = {bug['type'] for bug in bugs}
        self.assertIn('ZeroDivision', bug_types)
        self.assertIn('UnusedVariable', bug_types)
        self.assertIn('NoneComparison', bug_types)
        self.assertIn('EmptyExcept', bug_types)

    def test_variable_usage(self):
        self.parser.parse(self.sample_code)
        variables = self.parser.get_variable_usage()
        
        # Verify variables are tracked correctly
        self.assertIn('x', variables)
        self.assertIn('y', variables)
        self.assertIn('z', variables)
        self.assertIn('used_var', variables)
        self.assertIn('divide', variables)  # Function calls
        
        # Verify line numbers are captured
        self.assertGreater(len(variables['x']), 0)
        self.assertGreater(len(variables['divide']), 0)

if __name__ == '__main__':
    unittest.main()