import numpy as np
import matplotlib.pyplot as plt

# simulate the system using given initial condition and input


def forward_simu(x_0, u, dt, n_steps):
    n_states = x_0.size
    x_sim = np.zeros((n_states, n_steps))
    x_sim[:, 0] = x_0
    for i in range(n_steps-1):
        x_sim[:, i+1] = np.array([u[0, i]*np.cos(x_sim[2, i]), u[0, i]
                                  * np.sin(x_sim[2, i]), u[1, i]])*dt + x_sim[:, i]
    return x_sim

# calcluate the total cost


def total_cost(x_des, u_des, x_act, u_act, Q, Q_f, R, n_steps):
    cost = 0
    for i in range(n_steps - 1):
        cost = cost + (x_des[:, i]-x_act[:, i]).T@Q@(x_des[:, i]-x_act[:, i]) + \
            (u_des[:, i]-u_act[:, i]).T@R@(u_des[:, i]-u_act[:, i])
    cost = cost + (x_des[:, -1]-x_act[:, -1]).T@Q_f@(x_des[:, -1]-x_act[:, -1])
    return cost

# solve ricatti backward


def backward_ricatti_solve(Q, Q_f, R, u_act, x_act, u_des, x_des, dt, n_steps):
    n_inputs = R.shape[1]
    n_states = Q.shape[1]
    K = np.zeros((n_inputs, n_states, n_steps-1))  # 2*3*N
    l = np.zeros((n_inputs, n_steps-1))  # 2*N
    P = np.zeros((n_states, n_states, n_steps))  # 3*3*N
    p = np.zeros((n_states, n_steps))  # 3*N
    q = np.zeros((n_states, n_steps))  # 3*N
    q[:, 0:-1] = 2.0*Q@(x_act[:, 0:-1]-x_des[:, 0:-1])
    q[:, -1] = 2.0*Q_f@(x_act[:, -1]-x_des[:, -1]).T
    r = 2.0*R@(u_act-u_des)
    P[:, :, -1] = Q_f
    p[:, -1] = q[:, -1]

    for i in range(n_steps-1, 0, -1):
        A = np.array([[1, 0, -dt*u_act[0, i-1]*np.sin(x_act[2, i-1])],
                      [0, 1, dt*u_act[0, i-1]*np.cos(x_act[2, i-1])], [0, 0, 1]])
        B = np.array([[dt*np.cos(x_act[2, i-1]), 0],
                      [dt*np.sin(x_act[2, i-1]), 0], [0, dt]])
        g = r[:, i-1] + B.T@p[:, i]
        G = B.T@P[:, :, i]@A
        H = R+B.T@P[:, :, i]@B
        K[:, :, i-1] = -np.linalg.pinv(H)@G
        l[:, i-1] = -np.linalg.pinv(H)@g
        p[:, i-1] = q[:, i-1] + A.T@p[:, i] + K[:, :, i-1].T@H@l[:,
                                                                 i-1] + K[:, :, i-1].T@g + G.T@l[:, i-1]
        P[:, :, i-1] = Q + A.T@P[:, :, i]@A + K[:, :, i -
                                                1].T@H@K[:, :, i-1] + K[:, :, i-1].T@G + G.T@K[:, :, i-1]
    return [K, l]


def slq_solver(u_des, x_des, x_0, dt, n_steps, n_states, n_inputs, u_tol, Q_f, Q, R):

    max_slq_iteration = 150
    max_line_iteration = 50

    # SLQ Process
    slq_iter = 0
    u_step = 100
    u_act = u_des
    alpha = 1
    alphad = 1.2
    while slq_iter < max_slq_iteration and u_step > u_tol:
        # forword simulation
        x_act = forward_simu(x_0, u_act, dt, n_steps)
        [K, l] = backward_ricatti_solve(
            Q, Q_f, R, u_act, x_act, u_des, x_des, dt, n_steps)
        # perform line search
        line_iter = 0
        cost_new = np.inf
        cost_now = total_cost(x_des, u_des, x_act, u_act, Q, Q_f, R, n_steps)
        u_update = np.zeros((n_inputs, n_steps - 1))
        flag = 1
        alpha = 1
        alphad = 1.2
        # plt.title('step')
        # plt.plot(x_act[0, :], x_act[1, :], '-r', label='desired trajectory')
        # plt.grid(True)
        # plt.show()
        while line_iter < max_line_iteration and cost_new > cost_now and flag != 0:
            # update control
            for i in range(n_steps-1):
                u_update[:, i] = u_act[:, i] + alpha*l[:, i] + \
                    K[:, :, i]@(x_des[:, i]-x_act[:, i])
                if u_update[0, i] > 2:
                    u_update[0, i] = 2
                elif u_update[0, i] < -2:
                    u_update[0, i] = -2
                if u_update[1, i] > 2:
                    u_update[1, i] = 2
                elif u_update[1, i] < -2:
                    u_update[1, i] = -2

            x_update = forward_simu(x_0, u_update, dt, n_steps)
            cost_new = total_cost(x_des, u_des, x_update, u_update, Q, Q_f, R, n_steps)

            if (line_iter == 0):
                cost_prev = cost_new
            elif (line_iter == 1):
                if (cost_new > cost_prev):
                    flag = 2
                cost_prev = cost_new
            else:
                if (cost_new > cost_prev):
                    flag = 0
                cost_prev = cost_new

            if (flag == 1):
                alpha = alpha / alphad
            elif (flag == 2):
                alpha = alpha * alphad

            line_iter += 1

        u_step = np.linalg.norm(u_act-u_update)
        u_act = u_update
        slq_iter += 1

    return u_act
