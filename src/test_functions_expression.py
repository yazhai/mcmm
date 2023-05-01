def expression_levy(dims: int = 2) -> str:
    """
    Return the levy function expression in string format

    params:
    dims: dimension of the levy function, int

    return:
    variables in the levy function, list of str
    expression of the levy function, str
    """

    variables = [f"x[{dims}]"]
    w = [f"(1 + (x[{i}] - 1) / 4)" for i in range(dims)]

    # write the expression for levy function
    expression = f"sin(pi * {w[0]})^2"

    for i in range(dims - 1):
        expression += f"+ ({w[i]}-1) ^2 * (1 + 10 * sin(pi * {w[i]} + 1)^2) "

    expression += f"+ ({w[-1]}-1)^2 * (1 + sin(2 * pi * {w[-1]})^2)"

    return variables, expression
