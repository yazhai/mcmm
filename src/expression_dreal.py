import re, copy

from dreal import *

from src.expression_dreal import *


def create_dreal_variables(variable_names: list):
    """
    Create a list of variables for dReal
    Note: the function's variable names are either 'x' or ['x[n]']

    params:
    variable_names: list of variable names, list of str or str

    return:
    variable_list: list of variables, list of Variable
    """

    variable_list = []

    pattern = r"(\w+)(?:\[(\d+)\])?"

    if isinstance(variable_names, str):
        variable_names = [variable_names]

    for name in variable_names:
        match = re.match(pattern, name)
        base, dim = None, 0
        if match:
            base = match.group(1)
            dim = int(match.group(2)) if match.group(2) else 0

        if base is not None:
            if dim == 0:
                variable_list.append(Variable(base))
            else:
                _variable_list = [Variable(f"{base}[{i}]") for i in range(dim)]
                variable_list.extend(_variable_list)

    return variable_list


def make_expression_dreal_constraint(
    variable_list: list, function_exp: str, output_name: str = None
):
    """
    Convert a function expression to a dReal constraint
    Note: the function's variables are in a vector of n : x[0], x[1], ..., x[n-1]

    params:
    variable_list: list of variables in the function, list of Variable
    function_exp: function expression, str

    return:
    fx: output variable, Variable
    constraint: dReal constraint of the function
    """
    default_value_name = "f_x"

    # Input Variables
    # Make reference to the input variable
    x = variable_list

    # Output Variable
    if output_name is None:
        output_name = default_value_name
    f_x = Variable(f"{output_name}")

    # Function expression: f_x = function_exp
    expression = f"{function_exp} == f_x"

    # Clean up the expression to match dReal syntax
    # Remove spaces
    expression = expression.replace(" ", "")
    # Replace ^ with ** to match dReal syntax
    expression = expression.replace("^", "**")

    # # Following may not be necessary
    # expression = expression.replace("exp", "dreal.exp")
    # expression = expression.replace("sin", "dreal.sin")
    # expression = expression.replace("cos", "dreal.cos")
    # expression = expression.replace("sqrt", "dreal.sqrt")

    # Remove preceding + sign: (+ x) -> (x)
    # Preceding - sign is fine: (- x) -> (-x)
    to_remove_preceding_sign = True
    while to_remove_preceding_sign:
        expression_init = copy.deepcopy(expression)

        expression = re.sub(r"\(\s*\+", "(", expression)
        # expression = re.sub(r"\(\s*\-", "(", expression)

        to_remove_preceding_sign = expression_init != expression

    # Make the function as a constraint
    function_as_constraint = And(eval(expression))

    return f_x, function_as_constraint


def make_interval_dreal_constraint(variable_list: list, interval_list: list):
    """
    Change the box/interval list to dReal constraint

    params:
    variable_list: list of variables, list of Variable
    interval_list: list of intervals, list of list of float

    return:
    box_as_constraint: dReal constraint of the box
    """

    assert len(interval_list) == len(variable_list)
    assert len(interval_list[0]) == 2

    dims = len(variable_list)

    var = variable_list[0]
    lb, ub = interval_list[0]
    box_as_constraint = And(var >= lb, var <= ub)

    for i in range(1, dims):
        var = variable_list[i]
        lb, ub = interval_list[i]
        box_as_constraint = And(box_as_constraint, var >= lb, var <= ub)

    return box_as_constraint
