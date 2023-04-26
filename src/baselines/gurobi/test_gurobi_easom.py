
import gurobipy as gp
from gurobipy import GRB
import numpy as np

try:

    # Create a new model
    m = gp.Model("easom")

    x1 = m.addVar(vtype=GRB.CONTINUOUS, name="x1")
    x2 = m.addVar(vtype=GRB.CONTINUOUS, name="x2")

    # First term
    cos_1_in = m.addVar(vtype=GRB.CONTINUOUS, name="cos_1_in")
    cos_1_out = m.addVar(vtype=GRB.CONTINUOUS, name="cos_1_out")

    m.addConstr(cos_1_in == x1, "cos_1_in")
    m.addGenConstrCos(cos_1_in, cos_1_out, "cos_1_out")

    # Second term
    cos_2_in = m.addVar(vtype=GRB.CONTINUOUS, name="cos_2_in")
    cos_2_out = m.addVar(vtype=GRB.CONTINUOUS, name="cos_2_out")

    m.addConstr(cos_2_in == x2, "cos_2_in")
    m.addGenConstrCos(cos_2_in, cos_2_out, "cos_2_out")

    # Third term
    exp_in = m.addVar(vtype=GRB.CONTINUOUS, name="exp_in")
    exp_out = m.addVar(vtype=GRB.CONTINUOUS, name="exp_out")

    m.addConstr(exp_in == -((x1 - np.pi) ** 2 + (x2 - np.pi) ** 2), "exp_in")
    m.addGenConstrExp(exp_in, exp_out, "exp_out")
    
    # Intermediate variable
    intermediate = m.addVar(vtype=GRB.CONTINUOUS, name="intermediate")
    m.addConstr(intermediate == -cos_1_out * cos_2_out, "intermediate")
    # m.addConstr(intermediate == cos_2_out * exp_out, "intermediate")

    m.setObjective(intermediate * exp_out, GRB.MINIMIZE)
    # m.setObjective(- cos_1_out * intermediate , GRB.MINIMIZE)

    # Add constraint: -100 <= x1 <= 100
    m.addConstr(x1 >= -10)
    m.addConstr(x1 <= 10)

    # Add constraint: -100 <= x2 <= 100
    m.addConstr(x2 >= -10)
    m.addConstr(x2 <= 10)

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