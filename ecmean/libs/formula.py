#!/usr/bin/env python3
'''
Shared functions for ecmean. Set of formula based tools
'''

#####################
# FORMULA FUNCTIONS #
#####################

import operator
import re
import logging

loggy = logging.getLogger(__name__)


def formula_wrapper(var, face, xfield):
    """
    Wrapper function to extract data-array from a dataset (xfield)
    and to apply the mathematical operation required if it is a derived variable
    """

    if 'derived' in face['variables'][var].keys():
        cmd = face['variables'][var]['derived']
        return _eval_formula(cmd, xfield)
    return xfield[var]

# this is a tool to parse a CDO-based formula into mathematical operatos
# there might exists something more intelligent such as the pyparsing package

def _eval_formula(mystring, xdataset):
    """Evaluate the cmd string provided by the yaml file
    producing a parsing for the derived variables"""

    # Tokenize the original string
    token = [i for i in re.split('(\\W+)', mystring) if i]
    if len(token) > 1:
        # Use order of operations
        return _operation(token, xdataset)
    return xdataset[token[0]]

def _operation(token, xdataset):
    """Parsing of the CDO-based commands using operator package
    and an ad-hoc dictionary. Could be improved, working with four basic
    operations only."""

    # define math operators: order is important, since defines
    # which operation is done at first!
    ops = {
        '/': operator.truediv,
        "*": operator.mul,
        "-": operator.sub,
        "+": operator.add
    }

    # use a dictionary to store xarray field and call them easily
    dct = {}
    for k in token:
        if k not in ops:
            if k.isdigit():
                dct[k] = float(k)
            else:
                dct[k] = xdataset[k]

    # apply operators to all occurrences, from top priority
    # so far this is not parsing parenthesis
    code = 0
    for p in ops:
        while p in token:
            code += 1
            # print(token)
            x = token.index(p)
            name = 'op' + str(code)
            replacer = ops.get(p)(dct[token[x - 1]], dct[token[x + 1]])
            dct[name] = replacer
            token[x - 1] = name
            del token[x:x + 2]
    return replacer
