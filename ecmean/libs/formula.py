"""
Shared functions for ecmean. Set of formula based tools
"""

#####################
# FORMULA FUNCTIONS #
#####################

import ast
import logging
import operator
import re

loggy = logging.getLogger(__name__)


def formula_wrapper(var, face, xfield):
    """
    Wrapper function to extract data-array from a dataset (xfield)
    and to apply the mathematical operation required if it is a derived variable
    """

    if "derived" in face["variables"][var]:
        cmd = face["variables"][var]["derived"]
        return _eval_formula(cmd, xfield)
    return xfield[var]


# Parser and evaluator for the CDO-like formulas of the interface files.
# Formulas are parsed with the Python parser, which provides precedence, parentheses,
# numbers (decimals, scientific notation) and whitespace. Only variable names, numbers,
# the four basic operations, unary minus and parentheses are accepted: nothing is executed.
# Variable names may start with a digit (e.g. ecCodes '2t', '10u'): they are not Python identifiers, so they are
# replaced by placeholders before parsing and restored in the syntax tree. Numbers in scientific notation
# ('1e5', '1e-5') are not names and are left untouched.

_OPERATIONS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.USub, ast.Name, ast.Load, ast.Constant, *_OPERATIONS)
_DIGIT_NAME = re.compile(r"(?<![\w.])(?!\d+[eE][+-]?\d+\b)\d+[A-Za-z_]\w*")


def _allowed(node):
    """Tell if a syntax tree node is accepted in a formula: numbers are the only constants allowed."""

    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float))
    return isinstance(node, _ALLOWED)


def _parse_formula(mystring):
    """Parse a formula into a syntax tree, raising ValueError for unsupported syntax.

    Variable names starting with a digit (e.g. '2t') are parsed through placeholders and restored
    in the returned tree, so its Name nodes always hold the original variable names.

    Args:
        mystring: the formula, e.g. '2t-273.15' or 'rsdt-rsut-rlut'.

    Returns:
        The root node of the expression syntax tree.
    """

    aliases = {}

    def alias(match):
        return aliases.setdefault(match.group(), f"__ecmean_var{len(aliases)}__")

    try:
        tree = ast.parse(_DIGIT_NAME.sub(alias, mystring.strip()), mode="eval")
    except SyntaxError as err:
        raise ValueError(f"Unsupported syntax in formula '{mystring}'") from err
    if not all(_allowed(node) for node in ast.walk(tree)):
        raise ValueError(f"Unsupported syntax in formula '{mystring}'")
    original = {placeholder: name for name, placeholder in aliases.items()}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = original.get(node.id, node.id)
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
