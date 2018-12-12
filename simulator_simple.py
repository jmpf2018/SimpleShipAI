#!/usr/bin/python
#-*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import RK45

class Simulator:
    def __init__(self):
        self.last_global_state = None
        self.last_local_state = None
        self.current_action = None
        self.steps = 0
        self.time_span = 10           # 20 seconds for each iteration
        self.number_iterations = 100  # 100 iterations for each step
        self.integrator = None

        ##Vessel Constants

        self.M = 115000 *10**3 + 174050 * 10 ** 3
        self.Iz = 414000000 * 10 ** 3+364540000 * 10**3

        self.L = 244.74 #length
        self.Draft = 15.3

        ## Water constants
        self.pho = 1.025 * 10**3# water density
        self.mi = 10**-3  # water viscosity

        ## resistence coeff.
        self.Cy = 0.06           # coeff de arrasto lateral
        self.lp = 7.65 # cross-flow center
        self.Cb = 0.85           # block coefficient
        self.B = 42             # Beam
        self.S = 27342        # wet surface

        ## Rudder Constants
        self.A_rud = 68  # propulsor thrus
        self.r_aspect = 2  # aspect ration
        self.x_rudder = -115  # rudder position
        self.x_prop = -112 #propulsor position
        self.delta_x = self.x_prop - self.x_rudder  # distance between rudder and propulsor
        self.rudder_area = 68

        ## Propulsor constants:
        self.D_prop = 7.2  # Diameter
        self.n_prop = 1.6  # rotation

    def reset_start_pos(self, global_vector):
        x0, y0, theta0, vx0, vy0, theta_dot0 = global_vector[0], global_vector[1], global_vector[2], global_vector[3], global_vector[4], global_vector[5]
        self.last_global_state = np.array([x0, y0, theta0, vx0, vy0, theta_dot0])
        self.last_local_state = self._global_to_local(self.last_global_state)
        self.current_action = np.zeros(2)
        self.integrator = self.scipy_runge_kutta(self.simulate_scipy, self.get_state(), t_bound=self.time_span)

    def step(self, angle_level, rot_level):
        self.current_action = np.array([angle_level, rot_level])
        while not (self.integrator.status == 'finished'):
            self.integrator.step()
        self.last_global_state = self.integrator.y
        self.last_local_state = self._global_to_local(self.last_global_state)
        self.integrator = self.scipy_runge_kutta(self.simulate_scipy, self.get_state(), t0=self.integrator.t, t_bound=self.integrator.t+self.time_span)
        return self.last_global_state

    def simulate_scipy(self, t, global_states):
        local_states = self._global_to_local(global_states)
        return self._local_ds_global_ds(global_states[2], self.simulate(local_states))

    def simulate(self, local_states):
        """
        :param local_states: Space state
        :return df_local_states
        """
        x1 = local_states[0]  # u
        x2 = local_states[1]  # v
        x3 = local_states[2]  # theta (not used)
        x4 = local_states[3]  # du
        x5 = local_states[4]  # dv
        x6 = local_states[5]  # dtheta

        Frx, Fry, Frz = self.compute_rest_forces(local_states)

        # Propulsion model
        Fpx, Fpy, Fpz = self.compute_prop_forces(local_states)

        # Derivative function

        fx1 = x4
        fx2 = x5
        fx3 = x6

        # main model simple
        fx4 = (Frx + Fpx) / (self.M)
        fx5 = (Fry + Fpy) / (self.M)
        fx6 = (Frz + Fpz) / (self.Iz)

        fx = np.array([fx1, fx2, fx3, fx4, fx5, fx6])
        return fx

    def scipy_runge_kutta(self, fun, y0, t0=0, t_bound=10):
        return RK45(fun, t0, y0, t_bound,  rtol=self.time_span/self.number_iterations, atol=1e-4)


    def get_state(self):
        return self.last_global_state

    def get_local_state(self):
        return self.last_local_state

    def _local_to_global(self, local_state):
        # local_state: [ux, uy, theta, uxdot, uydot, thetadot]
        theta = local_state[2]
        c, s = np.cos(theta), np.sin(theta)
        A = np.array([[c, -s], [s, c]])
        B_l_pos = np.array([local_state[0], local_state[1]])
        B_l_vel = np.array([local_state[3], local_state[4]])

        B_g_pos = np.dot(A, B_l_pos.transpose())
        B_g_vel = np.dot(A, B_l_vel.transpose())
        return np.array([B_g_pos[0], B_g_pos[1], local_state[2], B_g_vel[0], B_g_vel[1], local_state[5]])

    def _global_to_local(self, global_state):
        # global_states: [x, y, theta, vx, vy, thetadot]
        theta = global_state[2]
        c, s = np.cos(theta), np.sin(theta)
        A = np.array([[c, s], [-s, c]])
        B_g_pos = np.array([global_state[0], global_state[1]])
        B_g_vel = np.array([global_state[3], global_state[4]])

        B_l_pos = np.dot(A, B_g_pos.transpose())
        B_l_vel = np.dot(A, B_g_vel.transpose())
        return np.array([B_l_pos[0], B_l_pos[1], global_state[2], B_l_vel[0], B_l_vel[1], global_state[5]])

    def _local_ds_global_ds(self, theta, local_states):
        """
        The function recieves two local states, one refering to the state before the runge-kutta and other refering to a
        state after runge-kutta and then compute the global state based on the transition
        :param local_states_0: Local state before the transition
        :param local_states_1: Local state after the transition
        :return: global states
        """
        c, s = np.cos(theta), np.sin(theta)
        A = np.array([[c, -s], [s, c]])
        B_l_pos = np.array([local_states[0], local_states[1]])
        B_l_vel = np.array([local_states[3], local_states[4]])

        B_g_pos = np.dot(A, B_l_pos.transpose())
        B_g_vel = np.dot(A, B_l_vel.transpose())

        return np.array([B_g_pos[0], B_g_pos[1], local_states[2], B_g_vel[0], B_g_vel[1], local_states[5]])

    def compute_prop_forces(self, local_states):
        x1 = local_states[0]  # u
        x2 = local_states[1]  # v
        x3 = local_states[2]  # theta (not used)
        x4 = local_states[3]  # du
        x5 = local_states[4]  # dv
        x6 = local_states[5]  # dtheta

        beta = self.current_action[0] * np.pi / 6  # rudder (-30 à 30)
        alpha = self.current_action[1]  # propulsor

        J = x4 * 0.6 / (1.6 * 7.2)
        kt = 0.5 - 0.5 * J
        n_prop_ctrl = self.n_prop * alpha
        Fpx = kt * self.pho * n_prop_ctrl ** 2 * self.D_prop ** 4

        kr = 0.5 + 0.5 / (1 + 0.15 * self.delta_x / self.D_prop)
        ur = np.sqrt(x4 ** 2 + kr * 4 * kt * n_prop_ctrl ** 2 * self.D_prop ** 2 / np.pi)
        vr = -0.8 * x5
        Ur = np.sqrt(ur ** 2 + vr ** 2)
        fa = 6.13 * self.r_aspect / (self.r_aspect + 2.25)
        ar = beta
        FN = 0.5 * self.pho * self.A_rud * fa * Ur ** 2 * np.sin(ar)
        Fpy = -FN * np.cos(beta)
        Fpz = -FN * np.cos(beta) * self.x_rudder
        return Fpx, Fpy, Fpz

    def compute_rest_forces(self, local_states):
        x1 = local_states[0]  # u
        x2 = local_states[1]  # v
        x3 = local_states[2]  # theta (not used)
        x4 = local_states[3]  # du
        x5 = local_states[4]  # dv
        x6 = local_states[5]  # dtheta

        gamma = np.pi + np.arctan2(x5, x4)
        vc = np.sqrt(x4 ** 2 + x5 ** 2)

        # Composing resistivity forces
        Re = self.pho * vc * self.L / self.mi
        if Re == 0:
            C0 = 0
        else:
            C0 = 0.0094 * self.S / (self.Draft * self.L) / (np.log10(Re) - 2) ** 2
        C1 = C0 * np.cos(gamma) + (-np.cos(3 * gamma) + np.cos(gamma)) * np.pi * self.Draft / (8 * self.L)
        Frx = 0.5 * self.pho * vc ** 2 * self.L * self.Draft * C1

        C2 = (self.Cy - 0.5 * np.pi * self.Draft / self.L) * np.sin(gamma) * np.abs(
            np.sin(gamma)) + 0.5 * np.pi * self.Draft / self.L * (
                     np.sin(gamma) ** 3) + np.pi * self.Draft / self.L * (
                         1 + 0.4 * self.Cb * self.B / self.Draft) * np.sin(gamma) * np.abs(np.cos(gamma))
        Fry = 0.5 * self.pho * vc ** 2 * self.L * self.Draft * C2

        C6 = -self.lp / self.L * self.Cy * np.sin(gamma) * np.abs(np.sin(gamma))
        C6 = C6 - np.pi * self.Draft / self.L * np.sin(gamma) * np.cos(gamma)
        C6 = C6 - (0.5 + 0.5 * np.abs(np.cos(gamma))) ** 2 * np.pi * self.Draft / self.L * (
                    0.5 - 2.4 * self.Draft / self.L) * np.sin(gamma) * np.abs(np.cos(gamma))
        Frz = 0.5 * self.pho * vc ** 2 * self.L ** 2 * self.Draft * C6

        return Frx, Fry, Frz
