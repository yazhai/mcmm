from pyomo import environ as pe
import os

# provide an email address
os.environ['NEOS_EMAIL'] = 'zhizhenqin@ucsd.edu'

# formulate optimization model
model = pe.ConcreteModel()

model.x1 = pe.Var(domain=pe.Binary)
model.x2 = pe.Var(domain=pe.Binary)
model.x3 = pe.Var(domain=pe.Binary)
model.x4 = pe.Var(domain=pe.Binary)
model.x5 = pe.Var(domain=pe.Binary)

obj_expr = 3 * model.x1 + 4 * model.x2 + 5 * model.x3 + 8 * model.x4 + 9 * model.x5
model.obj = pe.Objective(sense=pe.maximize, expr=obj_expr)

con_expr = 2 * model.x1 + 3 * model.x2 + 4 * model.x3 + 5 * model.x4 + 9 * model.x5 <= 20
model.con = pe.Constraint(expr=con_expr)


solver_manager = pe.SolverManagerFactory('neos')

results = solver_manager.solve(model, solver = "minos")
print(results)
print(model.x1.value, model.x2.value, model.x3.value, model.x4.value, model.x5.value)

# model = ...
# solver_manager = pym.SolverManagerFactory('neos')
# results = solver_manager.solve(model, opt='cplex')
