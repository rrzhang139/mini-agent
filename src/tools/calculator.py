# src/tools/calculator.py

import ast
import operator

# Allowed operators (safe subset)
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,  # unary minus
}


def _eval(node):
    """Recursively evaluate an AST node safely."""
    if isinstance(node, ast.Num):  # constant number
        return node.n
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
    elif isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        left = _eval(node.left)
        right = _eval(node.right)
        return _ALLOWED_OPS[type(node.op)](left, right)
    else:
        raise ValueError("Unsupported operation")


def safe_calculate(expr: str) -> float:
    """
    Safely evaluate a basic arithmetic expression.
    Supports +, -, *, /, and **.
    Examples:
        >>> safe_calculate("2 + 3 * 4")
        14
        >>> safe_calculate("-(5 + 2)")
        -7
    """
    try:
        parsed = ast.parse(expr, mode="eval")
        result = _eval(parsed.body)
        if abs(result) > 1e9:  # sanity bound
            raise ValueError("Result out of allowed range")
        return round(result, 6)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


if __name__ == "__main__":
    while True:
        expr = input("Enter expression (or 'q' to quit): ")
        if expr.lower() == "q":
            break
        try:
            print("=", safe_calculate(expr))
        except Exception as err:
            print("Error:", err)
