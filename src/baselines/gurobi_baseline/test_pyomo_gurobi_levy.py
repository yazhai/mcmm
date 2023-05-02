from pyomo import environ as pe
import os
import numpy as np

# formulate optimization model
model = pe.ConcreteModel()

model.x1 = pe.Var(domain=pe.Reals, bounds=(-10, 10))
model.x2 = pe.Var(domain=pe.Reals, bounds=(-10, 10))

w1 = 1 + (model.x1 - 1) / 4
w2 = 1 + (model.x2 - 1) / 4

term1 = pe.sin(np.pi * w1) ** 2
term2 = (w2 - 1) ** 2 * (1 + pe.sin(2 * np.pi * w2) ** 2)
obj_expr = term1 + term2
model.obj = pe.Objective(sense=pe.minimize, expr=obj_expr)


solver_manager = pe.SolverFactory('gurobi')

results = solver_manager.solve(model)
print(results)
print(model.x1.value, model.x2.value, model.x3.value, model.x4.value, model.x5.value)