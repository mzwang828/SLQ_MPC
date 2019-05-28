import numpy as np
import matplotlib.pyplot as plt
from slq_solver import slq_solver

# params
dt = 0.025
total_steps = 100
n_states = 3
n_inputs = 2
u_tol = 0.5

Q_f = 10*np.diag([1, 1, 1])
Q = np.diag([1, 1, 1])
R = 0.01*np.eye(n_inputs)

mpc_horizon = 5
# forward_simu to get test desired trajectory
def forward_simu(x_0, u, dt, n_steps):
    n_states = x_0.size
    x_sim = np.zeros((n_states, n_steps))
    x_sim[:, 0] = x_0
    for i in range(n_steps-1):
        x_sim[:, i+1] = np.array([u[0, i]*np.cos(x_sim[2, i]), u[0, i]
                                  * np.sin(x_sim[2, i]), u[1, i]])*dt + x_sim[:, i]
    return x_sim

# forward simulation and set goal
u_simu = np.zeros((n_inputs, total_steps - 1))
u_simu[0, :] = np.linspace(0.5, 1.5, total_steps-1)
u_simu[1, :] = -np.linspace(0.5, 1.5, total_steps-1)
x_0 = np.array([[0, 0, np.pi/4]])
x_des = forward_simu(x_0, u_simu, dt, total_steps)
u_des = np.zeros((n_inputs, total_steps - 1))

x_0 = np.array([[0.25, 0.25, np.pi/4]])
u_des[0, :] = 1
u_des[1, :] = 0.5
plt.title('Goal')
plt.plot(x_des[0, :], x_des[1, :], '-r', label='actual trajectory')
plt.xticks(np.arange(0, 5, step=0.5))
plt.yticks(np.arange(0, 5, step=0.5))
plt.grid(True)
plt.show()
# initialize
u_slq_des = u_des[:, 0:mpc_horizon-1]
x_actual = np.zeros((n_states, total_steps))
u_actual = np.zeros((n_inputs, total_steps-1))
for i in range(total_steps-1):
    if mpc_horizon > total_steps - i:
        mpc_horizon = total_steps - i
        u_slq_des = u_slq_des[:, 1:]
    
    x_slq_des = x_des[:, i:i+mpc_horizon]

    u_slq_des = slq_solver(u_slq_des, x_slq_des, x_0, dt, mpc_horizon, n_states, n_inputs, u_tol, Q_f, Q, R)

    x_0 = np.array([u_slq_des[0, 0]*np.cos(x_0[0,2]), u_slq_des[0, 0] * np.sin(x_0[0,2]), u_slq_des[1, 0]])*dt + x_0

    x_actual[:,i] = x_0
    u_actual[:,i] = u_slq_des[:,0]

    print(i)


plt.title('Result')
plt.subplot(211)
plt.plot(x_des[0, :], x_des[1, :], '-r', label='desired trajectory')
plt.plot(x_actual[0, :], x_actual[1, :], '-b', label='actual trajectory')
plt.subplot(212)
plt.plot(u_actual[0, :], '-r', label='actual input')
plt.plot(u_actual[1, :], '-b', label='actual input')
plt.xticks(np.arange(0, 5, step=0.5))
plt.yticks(np.arange(0, 5, step=0.5))
plt.grid(True)
plt.show()

print(x_actual)