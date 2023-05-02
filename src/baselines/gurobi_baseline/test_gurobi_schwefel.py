2
import gurobipy as gp
from gurobipy import GRB
import numpy as np

try:

    # Create a new model
    m = gp.Model("schwefel")

    dimension = 1

    x = []
    for i in range(dimension):
        x.append(m.addVar(vtype=GRB.CONTINUOUS, name="x" + str(i)))
    
    summation = 0
    # Loop thorugh to get summation of all terms
    for i in range(dimension):
        abs_in = m.addVar(vtype=GRB.CONTINUOUS, name="abs_in" + str(i))
        abs_out = m.addVar(vtype=GRB.CONTINUOUS, name="abs_out" + str(i))

        m.addConstr(abs_in == x[i], "abs_in" + str(i))
        m.addGenConstrAbs(abs_in, abs_out, "abs_out" + str(i))

        sqrt_in = m.addVar(vtype=GRB.CONTINUOUS, name="sqrt_in" + str(i))
        sqrt_out = m.addVar(vtype=GRB.CONTINUOUS, name="sqrt_out" + str(i))

        m.addConstr(sqrt_in == abs_out, "sqrt_in" + str(i))
        m.addGenConstrPow(sqrt_in, sqrt_out, 0.5, "sqrt_out" + str(i))

        sin_in = m.addVar(vtype=GRB.CONTINUOUS, name="sin_in" + str(i))
        sin_out = m.addVar(vtype=GRB.CONTINUOUS, name="sin_out" + str(i))

        m.addConstr(sin_in == sqrt_out, "sin_in" + str(i))
        m.addGenConstrSin(sin_in, sin_out, "sin_out" + str(i))

        summation += sin_out * x[i]


    
    # Set objective
    m.setObjective(418.9829 * dimension - summation, GRB.MINIMIZE)

    # Add constraint: -500 <= x1 <= 500
    for i in range(dimension):
        m.addConstr(x[i] >= 400, "x" + str(i) + " lb")
        m.addConstr(x[i] <= 500, "x" + str(i) + " ub")

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