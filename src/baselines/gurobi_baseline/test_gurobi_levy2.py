
import gurobipy as gp
from gurobipy import GRB
import numpy as np

try:

    # Create a new model
    m = gp.Model("levy")

    dimension = 3

    # Create variables

    x1 = m.addVar(vtype=GRB.CONTINUOUS, name="x1")
    x2 = m.addVar(vtype=GRB.CONTINUOUS, name="x2")

    w1 = m.addVar(vtype=GRB.CONTINUOUS, name="w1")
    w2 = m.addVar(vtype=GRB.CONTINUOUS, name="w2")

    m.addConstr(w1 == 1 + (x1 - 1) / 4, "w1")
    m.addConstr(w2 == 1 + (x2 - 1) / 4, "w2")

    # first term
    sin_1_in = m.addVar(vtype=GRB.CONTINUOUS, name="sin_1_in")
    sin_1_out = m.addVar(vtype=GRB.CONTINUOUS, name="sin_1_out")
    m.addConstr(sin_1_in == np.pi * w1, "sin_1_in")
    m.addGenConstrSin(sin_1_in, sin_1_out, "sin_1_out")
    term1 = sin_1_out ** 2

    # last term
    sin_last_in = m.addVar(vtype=GRB.CONTINUOUS, name="sin_last_in")
    sin_last_out = m.addVar(vtype=GRB.CONTINUOUS, name="sin_last_out")
    m.addConstr(sin_last_in == 2 * np.pi * w2, "sin_last_in")
    m.addGenConstrSin(sin_last_in, sin_last_out, "sin_last_out")
    
    term_last_1 = m.addVar(vtype=GRB.CONTINUOUS, name="term_last_1")
    term_last_2 = m.addVar(vtype=GRB.CONTINUOUS, name="term_last_2")
    m.addConstr(term_last_1 == (w2 - 1) ** 2, "term_last_1")
    m.addConstr(term_last_2 == 1 + sin_last_out ** 2, "term_last_2")

    term_last = term_last_1 * term_last_2

    # # middle term
    # sin_mid_in = m.addVar(vtype=GRB.CONTINUOUS, name="sin_mid_in")
    # sin_mid_out = m.addVar(vtype=GRB.CONTINUOUS, name="sin_mid_out")
    # m.addConstr(sin_last_in == np.pi * w1 + 1, "sin_mid_in")
    # m.addGenConstrSin(sin_last_in, sin_last_out, "sin_mid_out")
    
    # term_mid_1 = m.addVar(vtype=GRB.CONTINUOUS, name="term_mid_1")
    # term_mid_2 = m.addVar(vtype=GRB.CONTINUOUS, name="term_mid_2")
    # m.addConstr(term_mid_1 == (w1 - 1) ** 2, "term_mid_1")
    # m.addConstr(term_mid_2 == 1 + 10 * sin_last_out ** 2, "term_mid_2")

    # term_mid = term_last_1 * term_last_2

    # Set objective
    m.setObjective(term1 + term_last, GRB.MINIMIZE)

    # Add constraint: -10 <= xi <= 10
    m.addConstr(x1 >= -10, "x1" + " lb")
    m.addConstr(x1 <= 10, "x1" + " ub")
    m.addConstr(x2 >= -10, "x2" + " lb")
    m.addConstr(x2 <= 10, "x2" + " ub")

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