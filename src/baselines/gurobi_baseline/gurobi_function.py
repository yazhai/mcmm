import numpy as np
from gurobipy import GRB

class GurobiFunction():
    @staticmethod
    def EncodeAckley(m, x, b=0.2, a=20, c=2 * np.pi):
        dimension = len(x)

        # First term

        # Inner summation
        squared_summation = 0
        for i in range(dimension):
            squared_summation += x[i] ** 2
        
        # Square root
        sqrt_in = m.addVar(vtype=GRB.CONTINUOUS, name="sqrt_in")
        sqrt_out = m.addVar(vtype=GRB.CONTINUOUS, name="sqrt_out")

        m.addConstr(sqrt_in == squared_summation / dimension, "sqrt_in")
        m.addGenConstrPow(sqrt_in, sqrt_out, 0.5, "sqrt")

        exp_1_in = m.addVar(vtype=GRB.CONTINUOUS, name="exp_1_in")
        exp_1_out = m.addVar(vtype=GRB.CONTINUOUS, name="exp_1_out")

        m.addConstr(exp_1_in == -b * sqrt_out, "exp_1_in")

        m.addGenConstrExp(exp_1_in, exp_1_out, "exp_1")

        term1 = -a * exp_1_out

        # Second term
        cos_summation = 0
        for d in range(dimension):
            cos_in = m.addVar(vtype=GRB.CONTINUOUS, name="cos_in" + str(d))
            cos_out = m.addVar(vtype=GRB.CONTINUOUS, name="cos_out" + str(d))
            m.addConstr(cos_in == c * x[d], "cos_in" + str(d))
            m.addGenConstrCos(cos_in, cos_out, "cos_out" + str(d))
            cos_summation += cos_out
        
        exp_2_in = m.addVar(vtype=GRB.CONTINUOUS, name="exp_2_in")
        exp_2_out = m.addVar(vtype=GRB.CONTINUOUS, name="exp_2_out")

        m.addConstr(exp_2_in == (1 / dimension) * cos_summation, "exp_2_in")
        m.addGenConstrExp(exp_2_in, exp_2_out, "exp_2")

        term2 = -exp_2_out

        m.setObjective(term1 + term2 + a + np.exp(1), GRB.MINIMIZE)

        # Add constraint: -10 <= x1 <= 10
        for i in range(dimension):
            m.addConstr(x[i] >= -10, "x" + str(i) + " lb")
            m.addConstr(x[i] <= 10, "x" + str(i) + " ub")
