2
import gurobipy as gp
from gurobipy import GRB
import numpy as np

try:

    # Create a new model
    m = gp.Model("michalewicz")

    dimension = 10
    param_m = 10

    x = []
    for i in range(dimension):
        x.append(m.addVar(vtype=GRB.CONTINUOUS, name="x" + str(i)))
    
    summation = 0
    # Loop thorugh to get summation of all terms
    for i in range(dimension):
        sin_1_in = m.addVar(vtype=GRB.CONTINUOUS, name="sin_1_in" + str(i))
        sin_1_in_out = m.addVar(vtype=GRB.CONTINUOUS, name="sin_1_in_out" + str(i))

        m.addConstr(sin_1_in == x[i], "sin_1_in" + str(i))
        m.addGenConstrSin(sin_1_in, sin_1_in_out, "sin_1_in_out" + str(i))

        sin_2_in = m.addVar(vtype=GRB.CONTINUOUS, name="sin_2_in" + str(i))
        sin_2_in_out = m.addVar(vtype=GRB.CONTINUOUS, name="sin_2_in_out" + str(i))

        m.addConstr(sin_2_in == (i + 1) * x[i] ** 2 / np.pi, "sin_2_in" + str(i))
        m.addGenConstrSin(sin_2_in, sin_2_in_out, "sin_2_in_out" + str(i))

        pow_in = m.addVar(vtype=GRB.CONTINUOUS, name="pow_in" + str(i))
        pow_out = m.addVar(vtype=GRB.CONTINUOUS, name="pow_out" + str(i))

        m.addConstr(pow_in == sin_2_in_out, "pow_in" + str(i))
        m.addGenConstrPow(pow_in, pow_out, 2 * param_m, "pow_out" + str(i))

        summation += sin_1_in_out * pow_out
    
    # Set objective
    m.setObjective(-summation, GRB.MINIMIZE)

    # Add constraint: 0 <= x1 <= pi
    for i in range(dimension):
        m.addConstr(x[i] >= 0, "x" + str(i) + " lb")
        m.addConstr(x[i] <= np.pi, "x" + str(i) + " ub")

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