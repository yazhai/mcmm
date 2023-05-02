
import gurobipy as gp
from gurobipy import GRB
import numpy as np

try:

    # Create a new model
    m = gp.Model("levy")

    dimension = 3

    # Create variables

    x = []
    for i in range(dimension):
        x.append(m.addVar(vtype=GRB.CONTINUOUS, name="x" + str(i)))
    
    w = []
    for i in range(dimension):
        w.append(m.addVar(vtype=GRB.CONTINUOUS, name="w" + str(i)))
        m.addConstr(w[i] == 1 + (x[i] - 1) / 4, "w" + str(i))

    # first term
    sin_1_in = m.addVar(vtype=GRB.CONTINUOUS, name="sin_1_in")
    sin_1_out = m.addVar(vtype=GRB.CONTINUOUS, name="sin_1_out")
    m.addConstr(sin_1_in == np.pi * w[0], "sin_1_in")
    m.addGenConstrSin(sin_1_in, sin_1_out, "sin_1_out")
    term1 = sin_1_out ** 2

    # last term
    sin_last_in = m.addVar(vtype=GRB.CONTINUOUS, name="sin_last_in")
    sin_last_out = m.addVar(vtype=GRB.CONTINUOUS, name="sin_last_out")
    m.addConstr(sin_last_in == 2 * np.pi * w[dimension - 1], "sin_last_in")
    m.addGenConstrSin(sin_last_in, sin_last_out, "sin_last_out")
    
    term_last_1 = m.addVar(vtype=GRB.CONTINUOUS, name="term_last_1")
    term_last_2 = m.addVar(vtype=GRB.CONTINUOUS, name="term_last_2")
    m.addConstr(term_last_1 == (w[dimension - 1] - 1) ** 2, "term_last_1")
    m.addConstr(term_last_2 == 1 + sin_last_out ** 2, "term_last_2")

    term_last = term_last_1 * term_last_2

    # middle terms
    summation = 0
    for i in range(dimension - 1):
        sin_inter_in = m.addVar(vtype=GRB.CONTINUOUS, name="var_sin_inter_in" + str(i))
        sin_inter_out = m.addVar(vtype=GRB.CONTINUOUS, name="var_sin_inter_out" + str(i))
        m.addConstr(sin_inter_in == np.pi * w[i] + 1, "const_sin_inter_in" + str(i))
        m.addGenConstrSin(sin_inter_in, sin_inter_out, "const_sin_inter_out" + str(i))

        term_inter_1 = m.addVar(vtype=GRB.CONTINUOUS, name="var_term_inter_1" + str(i))
        term_inter_2 = m.addVar(vtype=GRB.CONTINUOUS, name="var_term_inter_2" + str(i))
        m.addConstr(term_inter_1 == (w[i] - 1) ** 2, "const_term_inter_1" + str(i))
        m.addConstr(term_inter_2 == 1 + 10 * sin_inter_out ** 2, "const_term_inter_2" + str(i))
        summation_item = m.addVar(vtype=GRB.CONTINUOUS, name="var_summation_item" + str(i))
        m.addConstr(summation_item == term_inter_1 * term_inter_2, "const_summation_item" + str(i))

        summation += summation_item

    # Set objective
    m.setObjective(term1 + summation + term_last, GRB.MINIMIZE)

    # Add constraint: -10 <= x1 <= 10
    for i in range(dimension):
        m.addConstr(x[i] >= -10, "x" + str(i) + " lb")
        m.addConstr(x[i] <= 10, "x" + str(i) + " ub")

    m.params.NonConvex = 2

    print("Start optimizing")
    # Optimize model
    m.write("levy.lp");
    m.optimize()
    print("Finish optimizing")

    for v in m.getVars():
        print('%s %g' % (v.VarName, v.X))

    print('Obj: %g' % m.ObjVal)
    print('Runtime: %g' % m.Runtime)
    print(m.status)
    print('MIPGap: %g' % m.MIPGap)
    m.write("levy.sol")

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError as e:
    print('Encountered an attribute error: ' + str(e))