"""
Shared functions for ecmean. Set of formula based tools
"""

#####################
# FORMULA FUNCTIONS #
#####################

import ast
import logging
import operator

loggy = logging.getLogger(__name__)


def formula_wrapper(var, face, xfield):
    """
    Wrapper function to extract data-array from a dataset (xfield)
    and to apply the mathematical operation required if it is a derived variable
    """

    if "derived" in face["variables"][var]:
        cmd = face["variables"][var]["derived"]
        field = _eval_formula(cmd, xfield)
        field.name = var
        return field
    return xfield[var]


# Parser and evaluator for the CDO-like formulas of the interface files.
# Formulas are parsed with the Python parser, which provides precedence, parentheses,
# numbers (decimals, scientific notation) and whitespace. Only variable names, numbers,
# the four basic operations, unary minus and parentheses are accepted: nothing is executed.

_OPERATIONS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.USub, ast.Name, ast.Load, ast.Constant, *_OPERATIONS)


def _allowed(node):
    """Tell if a syntax tree node is accepted in a formula: numbers are the only constants allowed."""

    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float))
    return isinstance(node, _ALLOWED)


def _parse_formula(mystring):
    """Parse a formula into a syntax tree, raising ValueError for unsupported syntax."""

    try:
        tree = ast.parse(mystring.strip(), mode="eval")
    except SyntaxError as err:
        raise ValueError(f"Unsupported syntax in formula '{mystring}'") from err
    if not all(_allowed(node) for node in ast.walk(tree)):
        raise ValueError(f"Unsupported syntax in formula '{mystring}'")
    return tree.body


def formula_variables(mystring):
    """Return the unique variable names required by a formula, in order of appearance
    (the units of a derived variable are taken from the first one)."""

    names = sorted((n for n in ast.walk(_parse_formula(mystring)) if isinstance(n, ast.Name)), key=lambda n: n.col_offset)
    return list(dict.fromkeys(n.id for n in names))


def _eval_formula(mystring, xdataset):
    """Evaluate a formula of the interface files on the variables of a dataset."""

    def evaluate(node):
        if isinstance(node, ast.BinOp):
            return _OPERATIONS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp):
            return -evaluate(node.operand)
        if isinstance(node, ast.Name):
            return xdataset[node.id]
        return float(node.value)

    return evaluate(_parse_formula(mystring))
