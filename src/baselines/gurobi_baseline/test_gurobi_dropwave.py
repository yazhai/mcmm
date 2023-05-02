
import gurobipy as gp
from gurobipy import GRB
import numpy as np

try:

    # Create a new model
    m = gp.Model("dropwave")

    # Create variables
    x1 = m.addVar(vtype=GRB.CONTINUOUS, name="x1")
    x2 = m.addVar(vtype=GRB.CONTINUOUS, name="x2")

    cos_in = m.addVar(vtype=GRB.CONTINUOUS, name="cos_in")
    cos_out = m.addVar(vtype=GRB.CONTINUOUS, name="cos_out")

    sqrt_in = m.addVar(vtype=GRB.CONTINUOUS, name="sqrt_in")
    sqrt_out = m.addVar(vtype=GRB.CONTINUOUS, name="sqrt_out")

    m.addConstr(sqrt_in == x1 ** 2 + x2 ** 2, "sqrt_in")
    m.addGenConstrPow(sqrt_in, sqrt_out, 0.5, "s1")

    m.addConstr(cos_in == 12 * sqrt_out, "cos_in")
    m.addGenConstrCos(cos_in, cos_out, "c1")

    numerator = m.addVar(vtype=GRB.CONTINUOUS, name="numerator")
    m.addConstr(numerator == 1 + cos_out, "numerator")

    denominator = m.addVar(vtype=GRB.CONTINUOUS, name="denominator")
    m.addConstr(denominator == 2 + 0.5 * (x1 ** 2 + x2 ** 2), "denominator")

    # Set objective
    m.setObjective(- numerator / denominator, GRB.MINIMIZE)

    # Add constraint: -10 <= x1 <= 10
    m.addConstr(x1 >= -10, "x1 lb")
    m.addConstr(x1 <= 10, "x1 ub")

    # Add constraint: -10 <= x2 <= 10
    m.addConstr(x2 >= -10, "x2 lb")
    m.addConstr(x2 <= 10, "x2 ub")

    m.params.NonConvex = 2

    # Optimize model
    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.VarName, v.X))

    print('Obj: %g' % m.ObjVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError as e:
    print('Encountered an attribute error: ' + str(e))