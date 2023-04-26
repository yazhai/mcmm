2
import gurobipy as gp
from gurobipy import GRB
import numpy as np

try:

    # Create a new model
    m = gp.Model("sum square")

    dimension = 10

    x = []
    for i in range(dimension):
        x.append(m.addVar(vtype=GRB.CONTINUOUS, name="x" + str(i)))
    
    summation = 0
    # Loop thorugh to get summation of all terms
    for i in range(dimension):
        summation += (i + 1) * x[i] ** 2
    
    # Set objective
    m.setObjective(summation, GRB.MINIMIZE)

    # Add constraint: -10 <= x1 <= 10
    for i in range(dimension):
        m.addConstr(x[i] >= -10, "x" + str(i) + " lb")
        m.addConstr(x[i] <= 10, "x" + str(i) + " ub")

    m.params.NonConvex = 1

    # Optimize model
    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.VarName, v.X))

    print('Obj: %g' % m.ObjVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError as e:
    print('Encountered an attribute error: ' + str(e))