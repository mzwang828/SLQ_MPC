import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt

# params
n = 2
m = 1
T = 5
t = 0
u_max = 1
u_min = -1
A = np.array([[1, 1], [0, 1]])
B = np.array([0, 1])
Q = np.eye(1)
R = 0.1*np.eye(1)
maxiteration = 16
# initial
x_0 = np.array([5, 0])
y_r = np.array([5, 7, 7, 5, 5, 4, 4, 3, 3, 2, 1,
                1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2])
# QP
x = Variable((n, T+1))
u = Variable((m, T))
y = Variable((T+1))
x_init = Parameter(n)
cost = 0
constraints = []
for k in range(T):
    cost += sum_squares(x[0, k]-y_r[t+k]) + sum_squares(u[:, k])
    constraints += [x[:, k+1] == A@x[:, k]+B@u[:, k]]
    constraints += [u[:, k] >= u_min, u[:, k] <= u_max]
constraints += [x[:, 0] == x_init]
prob = Problem(Minimize(cost), constraints)

# solve
x1_actual = []
x2_actual = []
u_applied = []
for k in range(maxiteration):
    x_init.value = x_0
    x1_actual.append(x_0[0])
    x2_actual.append(x_0[1])
    prob.solve(verbose=False, warm_start=True)
    x_0 = A@x_0 + B*u[:, 0].value
    u_applied.append(u[:, 0].value)
    t += 1
    cost = 0
    for k in range(T):
        cost += sum_squares(x[0, k]-y_r[t+k]) + sum_squares(u[:, k])
    prob = Problem(Minimize(cost), constraints)

# plot
plt.subplot(211)
plt.title('states')
plt.plot(x1_actual, '-r', label='x1')
plt.plot(x2_actual, '-b', label='x2')
plt.yticks(np.arange(-5, 8, step=2))
plt.grid(True)
plt.subplot(212)
plt.title('input')
plt.plot(u_applied, '-r', label='x1')
plt.yticks(np.arange(-1, 1, step=0.5))
plt.grid(True)
plt.show()
