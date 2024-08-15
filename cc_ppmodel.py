#!/home/yq18021/anaconda3/bin/python3.8
"""
Simulate state estimation in cerebellar cortex. Inference is about state
estimation in extracerebellar regions, which here constitute the generative
process, whereas the generative model is instantiated in the cerebellum.
"""
import pdb
import os
import sys
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal


class GP:
    """ Class for generative process: simulate physical states of the animal
    and world; the dynamics are a function of (hardwired) behavioural state of
    the animal (b_state); there are 8 behavioural variables (vector y), 1 for
    lc, two for wh, rs and lk (proper and phase), one for sn; action (function
    of VFE) changes rs variable through mapping (from phase to proper state)
    instead than through dynamics.
    """

    def __init__(
            self,
            dt=0.01,
            n_step=500,
            ω_wh=np.array([2, 4]) * 2 * np.pi,
            α_wh=[1, 0],
            μ_wh=[0, .0],
            ω_rs=np.array([2, 4]) * 2 * np.pi,
            α_rs=[1],
            dw=5,
            ω_offset=np.array([0, 0]),
            rng=np.random.RandomState(100),
            τ=np.ones(5)):
        """ Initialise  simulation and behavioural variables and parameters;
        frequency parameters in Hz (ref time is sec).
        """
        self.dt = dt
        self.n_step = n_step
        # Behavioural state intervals (state duration) and sequence
        self.state_interval = [
            int(self.n_step * 45 / 100),
            int(self.n_step * 5 / 100),
            int(self.n_step * 50 / 100),
            int(self.n_step * 0 / 100)
        ]
        self.state_seq = np.concatenate(
            (np.array([1] * self.state_interval[0]),
             np.array([2] * self.state_interval[1]),
             np.array([3] * self.state_interval[2]),
             np.array([4] * self.state_interval[3])))
        # State parameters
        self.ω_wh = ω_wh      # natural frequency wh
        self.dw_wh = 0        # noise to dynamics of phase wh
        self.α_wh = α_wh      # amplitude oscillation wh (fixed point)
        self.μ_wh = μ_wh      # setpoint wh
        self.ω_rs = ω_rs + ω_offset  # natural frequency rs
        self.dw_rs = 0               # noise to dynamics of phase rs
        self.α_rs = α_rs        # amplitude oscillations rs (constant)
        self.τ = τ              # time constats dynamics
        # \vec{y}: y0=wh, y1=α_wh, y2=Φ_wh, y3=rs, y4=Φ_rs
        # self.y = np.array([0, 0, 1, 0, 4])
        self.y = np.array([0.5, 0, 1, 0.5, 1])
        self.y[[0, 3]] = np.sin(self.y[[2, 4]])
        self.dw = dw            # common noise
        self.rng = rng          # rng seed

        # CheckCode: Save difference between wh-rs and rs-lc and action
        self.diff_wh_rs = []

    def dy(self, b_state, y_temp, CN_output, action=False, save=False):
        """ Update equations for behavioural variables y, dependent on
        behavioural state b_state; wh and rs are α*sin() mapping from
        φ_wh and φ_rs which follow Kuramoto equations; similar
        for lk but also discretised (0 or 1); α_wh in wh mapping decays
        exponentially towards b_state-dependent fixed point; noise is added
        to equations of motion and is b_state-dependent.
        """
        # Disable action if needed (k parameters)
        k = [5, 5] if action else [0, 0]  # a is strength CN-ExtraCereb syn

        if b_state == 1:
            return np.array([
                0, 5 * (self.α_wh[0] - y_temp[1]), self.ω_wh[0] + k[0] *
                np.sin(CN_output[0] - y_temp[2]) + self.dw_wh, 0, self.ω_rs[0] +
                k[1] * np.sin(CN_output[1] - y_temp[4]) + self.dw_rs
            ]) / self.τ

        elif b_state == 2:
            return np.array([
                0, 5 * (self.α_wh[1] - y_temp[1]), self.ω_wh[0] + k[0] *
                np.sin(CN_output[0] - y_temp[2]) + self.dw_wh, 0, self.ω_rs[0] +
                k[1] * np.sin(CN_output[1] - y_temp[4]) + self.dw_rs
            ]) / self.τ

        elif b_state == 3:
            return np.array([
                0, 5 * (self.α_wh[0] - y_temp[1]), self.ω_wh[1] + k[0] *
                np.sin(CN_output[0] - y_temp[2]) + self.dw_wh, 0, self.ω_rs[1] +
                k[1] * np.sin(CN_output[1] - y_temp[4]) + self.dw_rs
            ]) / self.τ

        elif b_state == 4:
            return np.array([
                0, 5 * (self.α_wh[1] - y_temp[1]), self.ω_wh[1] + k[0] *
                np.sin(CN_output[0] - y_temp[2]) + self.dw_wh, 0, self.ω_rs[0] +
                k[1] * np.sin(CN_output[1] - y_temp[4]) + self.dw_rs
            ]) / self.τ

    def mapping(self, b_state):
        """ Mapping sin() of wh, rs and lk from φ_wh, φ_rs, φ_lk; lk is
        additionally discretised (0 or 1); amplitude wh is dynamic state y[2]
        following α_wh (see dy function); mapping of rs depends also
        on action.
        """
        self.y[[0, 3]] = [self.y[1], self.α_rs[0]] * np.sin(self.y[[2, 4]])

    # Euler method
    def update(self, b_state, CN_output, action=True, perturb=False):
        # Save difference before update
        self.diff_wh_rs.append(self.y[0] - self.y[3])

        # Update dynamics noise
        self.dw_wh = self.dw * self.rng.normal()
        self.dw_rs = self.dw * self.rng.normal()

        self.y = self.y + self.dt * self.dy(
            b_state, self.y, CN_output, action=action, save=True)

        if perturb:
            self.y = self.y - np.array([0, 0, 3 * np.pi // 2, 0, 0])

        # Mappings
        self.mapping(b_state)

        return self.y


class GM:
    """ Class for genertive model: states u comprise hidden states x, their
    derivatives x_dot, and v; state inference follows x'"""

    def __init__(self, dt=0.01, expect_sync=True):
        self.save_pe_y = []
        self.save_pe_x = []
        self.dt = dt
        # Hidden causes, states, observations
        self.x = np.array([0, 0])
        self.x_prime = np.array([0, 0])
        self.v = np.array([0, 0])
        self.u = np.array((self.x, self.x_prime, self.v))
        self.y = np.array([0, 0, 0, 0, 0])
        # Variational free energy
        self.vfe = []
        # Parameters
        self.θ_g = np.eye(2)
        if expect_sync:
            self.θ_f = np.array([[1, 1], [1, 1]])
        else:
            self.θ_f = np.array([[1, 0], [0, 1]])
        # self.θ_g = np.array(([1, 0, 0, 0, 0], [0, 1, .5, 0,
        #                                        0], [0, 0.5, 1, 0.5,
        #                                             0], [0, 0, 0.5, 1,
        #                                                  0], [0, 0, 0, 0, 1]))
        # self.θ_f = np.array(([1, .9, 0], [0, 1, 0], [0, .7, 1]))
        # Covariance matrices
        # self.Ω_z = np.eye(2) * np.exp(1)
        # self.Ω_w = np.eye(2) * np.exp(2)
        self.Ω_z = np.eye(2) * np.exp(3)
        self.Ω_w = np.eye(2) * np.exp(3)
        self.Ω_v = np.eye(2) * np.exp(10)  # uninformative priors for v
        # self.Ω_v = np.eye(2) * np.exp(1.9)  # uninformative priors for v
        # self.Ω_z = np.array(([1, 0.0, 0, 0, 0], [0.0, .6, .5, 0, 0], [0, .5, .6, .05, 0], [0, 0, .05, 1, 0], [0, 0, 0, 0, 1])) * np.exp(3)
        # self.Ω_w = np.array(([1, 0.0, 0, 0, 0], [0.0, .6, .5, 0, 0], [0, .5, .6, .05, 0], [0, 0, .05, 1, 0], [0, 0, 0, 0, 1])) * np.exp(3)

        # Learning rates
        self.κ_x = 1 * 8
        self.κ_x_prime = 1 / 1
        self.κ_v = 1 * 6

        # self.κ_x = 1 / 0.5
        # self.κ_x_prime = 1 / 5
        # self.κ_v = 1 / 0.5

        # Save prediction errors ([y, x, v], [w, r, l])
        self.ε = np.empty([3, 3])

    def g(self, u_temp):
        """ Mapping; x-dependent """
        return np.dot(self.θ_g, u_temp[0])

    def f(self, u_temp):
        """ Dynamics; x- and v-dependent """
        # return -u_temp[0] + np.dot(self.θ_f, u_temp[2])
        return -u_temp[0] + np.dot(self.θ_f, u_temp[2])
        # return -u_temp[0]  + np.dot(self.θ_f, u_temp[2]) / 5 #(justify sayng that MF-DCN>Pjc-DCN)

    def Π_fun(self):  # dynamic Π
        """ Inverte covariance matrices; Ω_z could be state dependent """
        # Precision matrices
        Π_z = np.linalg.inv(self.Ω_z * np.ones(2))
        Π_w = np.linalg.inv(self.Ω_w)
        Π_v = np.linalg.inv(self.Ω_v)

        return np.array((Π_z, Π_w, Π_v))

    def ε_fun(self, u_temp, y):
        """ Prediction error terms"""
        ε_y = y - self.g(u_temp)
        ε_x = u_temp[1] - self.f(u_temp)
        # print(self.f(u_temp))
        ε_v = u_temp[2] - np.zeros(2)  # 0 mean prior for v
        self.ε = np.array((ε_y, ε_x, ε_v))
        # pdb.set_trace()

        return np.array((ε_y, ε_x, ε_v))

    def dFE_du(self, Πs, u_temp, y, save='no'):  # sin(θ + a)
        """ VFE gradient for state u which comprises hidden states x, its
        temporal derivative x_prime and hidden causes v; dx, dx_prime and dv
        are partial derivatives dU/dx, dU/dx_prime, dU/dv, where U is internal
        energy term (-lnp(u, y)) of VFE.
        """
        Π_z, Π_w, Π_v = Πs
        ε_y, ε_x, ε_v = self.ε_fun(u_temp, y)
        dx = self.κ_x * (-self.θ_g.T.dot(Π_z.dot(ε_y)) + Π_w.dot(ε_x))
        dx_prime = self.κ_x_prime * (Π_w.dot(ε_x))
        dv = self.κ_v * (-self.θ_f.T.dot(Π_w.dot(ε_x)) + Π_v.dot(ε_v))
        # pdb.set_trace()

        # Save variational action
        if save == 'yes':
            self.vfe.append(
                1 / 2 * (ε_y.T.dot(Π_z.dot(ε_y)) + ε_x.T.dot(Π_w.dot(ε_x)) +
                         ε_v.T.dot(Π_v.dot(ε_v))))
            self.save_pe_y.append(Π_z.dot(ε_y))
            self.save_pe_x.append(Π_w.dot(ε_x))

        return np.array((dx, dx_prime, dv))

    # def update(self, y):  # sin(θ + a) + added x-v resp
    #     # Precision matrices Πss
    #     Πs = self.Π_fun()
    #     # Encoded trajectory or expected motion for modified gradient descent
    #     du = np.array((self.u[1], np.zeros(2), np.zeros(2)))

    #     # Runge-Kutta method
    #     k1 = self.dt * self.dFE_du(Πs, self.u, y, save='yes')
    #     k2 = self.dt * self.dFE_du(
    #         Πs, self.u + self.dt / 2 * self.dFE_du(Πs, self.u + k1 / 2, y), y)
    #     k3 = self.dt * self.dFE_du(
    #         Πs, self.u + self.dt / 2 * self.dFE_du(Πs, self.u + k2 / 2, y), y)
    #     k4 = self.dt * self.dFE_du(
    #         Πs, self.u + self.dt * self.dFE_du(Πs, self.u + k3, y), y)
    #     # pdb.set_trace()
    #     self.u = self.u + self.dt * du - (1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

    def update(self, y):     # sin(θ + a)
        # Precision matrices Πss
        Πs = self.Π_fun()
        # Encoded trajectory or expected motion for modified gradient descent
        du = np.array((self.u[1], np.zeros(2), np.zeros(2)))
        # FE gradient
        FE_grad = self.dFE_du(Πs, self.u, y, save='yes')
        # Eular updates
        self.u = self.u + self.dt * (du - FE_grad)

def spectral_fun(diff_wh_rs, dt_gp, n_step):
    """ Spectral analysis of the difference between whisking and respiration
    """
    # pdb.set_trace()
    # sim_t = n_step * dt_gp      # total time simulation (sec)
    sim_t = np.arange(0, (n_step + 1) * dt_gp)  # total time simulation (sec)
    # sim_t = np.linspace(0, (n_step) * dt_gp, n_step)  # total time simulation (sec)
    fs = n_step / sim_t[-1]  # sampling frequncy

    # Windowed Fourier transform
    f_wh, t_wh, Zxx_wh = signal.stft(diff_wh_rs, fs=fs, nperseg=500)

    # Morlet wavelet transform
    w = 6.  # angular freq ω Mother
    # freq = np.linspace(.01, fs / 2, 100)  # fs/2 Niquist freq
    freq = np.linspace(.01, 10, 100)  # fs/2 Niquist freq
    width = w * fs / (2 * np.pi * freq)

    # Cone of influence - for edge effects
    coi = width * np.sqrt(2) * dt_gp
    coil = np.where(coi > sim_t[-1], sim_t[-1], coi) / dt_gp
    coir = sim_t[-1] - coi
    coir = np.where(coir < 0, 0, coir) / dt_gp

    diff_wt = signal.cwt(diff_wh_rs, signal.morlet2, width, w=w)

    # mw_wh = signal.cwt(save_gp[0, 1:], signal.morlet2, width, w=w)
    # mw_rs = signal.cwt(save_gp[3, 1:], signal.morlet2, width, w=w)

    # # Phase difference (scaled by power for each frequency)
    # dphs_scl = np.where(np.abs(np.angle(mw_wh) - np.angle(mw_rs)) < np.pi, np.abs(np.angle(mw_wh) - np.angle(mw_rs)), np.abs(np.angle(mw_wh) + np.angle(mw_rs)))

    # dphs_scl = np.where(np.abs(mw_wh) > 2, dphs_scl, 0)

    return diff_wt, coil, coir, freq, sim_t


def runsim(n_step=7000,
           dt_gp=0.01,
           dt_gm=1,
           dw=5,
           ω_offset=np.array([.1, .1]),
           rng=100,
           action=True,
           n_run=1,
           perturb=False,
           expect_sync=True,
           obs_noise=False,
           plotfig=False):
    """ Run simulation, generate figures and display/save; simulation
    of gp has dt=0.005 and gm learning rate is κ ~ 0.1, thus one order
    of magnitude less; for generative model used dt=1, therefore running
    at same speed as generative process.
    """
    # Set simulation conditions
    if dw != 0:
        simcond = 'noise'
    elif not(all(np.isclose(ω_offset, np.array([0, 0])))):
        simcond = 'offset'
    elif perturb:
        simcond = 'perturb'

    if not(action):
        simcond = 'noaction'

    # Initialise random number generator
    rng = np.random.RandomState(rng)

    # Save run
    save_run = []
    save_fa = []

    # Run simulation
    for r in range(n_run):
        # Initialise gp, gm
        gp = GP(dt=dt_gp, n_step=n_step, dw=dw, ω_offset=ω_offset, rng=rng)
        gm = GM(dt=dt_gm, expect_sync=expect_sync)

        # Initialise sequence of b_states
        state_seq = gp.state_seq

        # Matrices for saving results - obs are noisy observations for gm
        save_gp = np.concatenate(
            (np.expand_dims(gp.y, axis=1), np.zeros([5, n_step])), axis=1)
        save_obs = np.concatenate(
            (np.expand_dims(gp.y[[0, 3]], axis=1), np.zeros([2, n_step])), axis=1)
        save_u = np.empty([3, 2, n_step])
        save_ε = np.empty([3, 2, n_step])

        for i in range(n_step):
            # New observations
            if obs_noise:
                obs = save_gp[[0, 3], i] + rng.normal(size=2)
            else:
                obs = save_gp[[0, 3], i] + rng.normal(size=2) * [0.0, 0.0]
                # Update gm, gp
            gm.update(obs)
            # pdb.set_trace()
            if ((i == n_step // 13) or (i == n_step * 7 // 13)) and (perturb is True):
                # gp.update(state_seq[i], gm.u[2, :], action=action)  # feed v
                gp.update(state_seq[i], gm.u[0, :], action=action, perturb=True)  # feed x
            else:
                # gp.update(state_seq[i], gm.u[2, :], action=action)  # feed v
                gp.update(state_seq[i], gm.u[0, :], action=action)  # feed x
                # Save - for gp next time step because updated
            save_gp[:, i + 1] = gp.y
            save_obs[:, i] = obs
            save_u[:, :, i] = gm.u
            save_ε[:, :, i] = gm.ε

        # for i in range(n_step):
        #     # New observations
        #     if obs_noise:
        #         obs = save_gp[[0, 3], i] + rng.normal(size=2)
        #     else:
        #         obs = save_gp[[0, 3], i] + rng.normal(size=2) * [0.0, 0.0]
        #     # Update gm, gp
        #     gm.update(obs)
        #     # pdb.set_trace()
        #     # gp.update(state_seq[i], gm.u[2, :], action=action)  # feed v
        #     gp.update(state_seq[i], gm.u[0, :], action=action)  # feed x
        #     # Save - for gp next time step because updated
        #     save_gp[:, i + 1] = gp.y
        #     save_obs[:, i] = obs
        #     save_u[:, :, i] = gm.u
        #     save_ε[:, :, i] = gm.ε

        # Save run
        save_run.append([save_gp, save_obs, save_u, save_ε])

        # Save Free action
        save_fa.append(np.cumsum(gm.vfe))

        # Spectral analysis of wh-rs difference
        if r == n_run - 1:
            diff_wt, coil, coir, freq, sim_t = spectral_fun(
                gp.diff_wh_rs, dt_gp, n_step)

    # Generate figures ############
    # Color variables
    cl_wh = 'darkkhaki'
    cl_rs = 'slategray'

    # Style plots
    plt.style.use('seaborn-v0_8-pastel')
    mpl.rcParams['lines.linewidth'] = 4
    mpl.rcParams['ytick.labelsize'] = 20
    mpl.rcParams['xtick.labelsize'] = 20    
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['legend.fontsize'] = 20    

    # Plot generative process
    fig1, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True, sharey=False)
    axs[0].plot(save_obs[0, :].T)
    axs[0].plot(save_gp[0, :].T, color=cl_wh, label='whisking')
    axs[1].plot(save_obs[1, :].T)
    axs[1].plot(save_gp[3, :].T, color=cl_rs, label='respiration')

    # Set xticks and label
    axs[1].set_xticks(sim_t[::5] / dt_gp)
    axs[1].set_xticklabels(sim_t[::5].astype(int).tolist())
    axs[1].set_xlabel('sec')

    # Set y label
    axs[0].set_ylabel('Amplitude whisking (a.u.)')
    axs[1].set_ylabel('Amplitude respiration (a.u.)')

    # Add vertical lines, legend and title + remove frame
    state_int_csum = np.cumsum(gp.state_interval)
    for idx, _ in enumerate(axs):
        axs[idx].axvline(state_int_csum[0], ls='--', lw='3', c='salmon')
        axs[idx].axvline(state_int_csum[1], ls='--', lw='3', c='salmon')
        # axs[idx].axvline(state_int_csum[2], ls='--', lw='3', c='salmon')
        # axs[idx].axvline(state_int_csum[3], ls='--', lw='3', c='salmon')
        axs[idx].legend(loc="upper right",
                        handlelength=0,
                        bbox_to_anchor=(1.13, 1))
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)

    fig1.suptitle('Generative process')

    # Plot generative model - hidden states x
    fig2, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True, sharey=False)
    axs[0].plot(save_u[0, 0, :].T, color=cl_wh, label='whisking')
    axs[1].plot(save_u[0, 1, :].T, color=cl_rs, label='respiration')

    # Set xticks and label
    axs[1].set_xticks(sim_t[::5] / dt_gp)
    axs[1].set_xticklabels(sim_t[::5].astype(int).tolist())
    axs[1].set_xlabel('sec')

    # Set y label
    axs[0].set_ylabel('a.u.')
    axs[1].set_ylabel('a.u.')

    # Add vertical lines, legend and title + remove frame
    # state_int_csum = np.cumsum(gp.state_interval)
    for idx, _ in enumerate(axs):
        axs[idx].axvline(state_int_csum[0], ls='--', lw='3', c='salmon')
        axs[idx].axvline(state_int_csum[1], ls='--', lw='3', c='salmon')
        # axs[idx].axvline(state_int_csum[2], ls='--', lw='3', c='salmon')
        # axs[idx].axvline(state_int_csum[3], ls='--', lw='3', c='salmon')
        axs[idx].legend(loc="upper right",
                        handlelength=0,
                        bbox_to_anchor=(1.13, 1))
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)

    fig2.suptitle('Hidden states')

    # Plot generative model - hidden state velocities v_prime and pe
    fig3, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True, sharey=False)
    axs[0].plot(-save_u[1, 0, :].T, color=cl_wh, label='whisking')
    axs[1].plot(-save_u[1, 1, :].T, color=cl_rs, label='respiration')

    # Set xticks and label
    axs[1].set_xticks(sim_t[::5] / dt_gp)
    axs[1].set_xticklabels(sim_t[::5].astype(int).tolist())
    axs[1].set_xlabel('sec')

    # Set y label
    axs[0].set_ylabel('a.u.')
    axs[1].set_ylabel('a.u.')

    # Add vertical lines, legend and title + remove frame
    # state_int_csum = np.cumsum(gp.state_interval)
    for idx, _ in enumerate(axs):
        axs[idx].axvline(state_int_csum[0], ls='--', lw='3', c='salmon')
        axs[idx].axvline(state_int_csum[1], ls='--', lw='3', c='salmon')
        # axs[idx].axvline(state_int_csum[2], ls='--', lw='3', c='salmon')
        # axs[idx].axvline(state_int_csum[3], ls='--', lw='3', c='salmon')
        axs[idx].legend(loc="upper right",
                        handlelength=0,
                        bbox_to_anchor=(1.13, 1))
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)

    fig3.suptitle('Hidden state velocities')

    # Plot generative model - hidden causes v
    fig4, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True, sharey=False)
    axs[0].plot(save_u[2, 0, :].T, color=cl_wh, label='whisking')
    axs[1].plot(save_u[2, 1, :].T, color=cl_rs, label='respiration')

    # Set xticks and label
    axs[1].set_xticks(sim_t[::5] / dt_gp)
    axs[1].set_xticklabels(sim_t[::5].astype(int).tolist())
    axs[1].set_xlabel('sec')

    # Set y label
    axs[0].set_ylabel('a.u.')
    axs[1].set_ylabel('a.u.')


    # Add vertical lines, legend and title + remove frame
    # state_int_csum = np.cumsum(gp.state_interval)
    for idx, _ in enumerate(axs):
        axs[idx].axvline(state_int_csum[0], ls='--', lw='3', c='salmon')
        axs[idx].axvline(state_int_csum[1], ls='--', lw='3', c='salmon')
        # axs[idx].axvline(state_int_csum[2], ls='--', lw='3', c='salmon')
        # axs[idx].axvline(state_int_csum[3], ls='--', lw='3', c='salmon')
        axs[idx].legend(loc="upper right",
                        handlelength=0,
                        bbox_to_anchor=(1.13, 1))
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)

    fig4.suptitle('Hidden causes')

    # Plot prediction errors
    fig5 = plt.figure(figsize=(12, 12))
    subfig = fig5.subfigures(2, 1)
    axs_0 = subfig[0].subplots(2, 1, sharex=True, sharey=False)
    axs_0[0].plot(save_ε[0, 0, :].T, color=cl_wh, label='whisking')
    axs_0[1].plot(save_ε[0, 1, :].T, color=cl_rs, label='respiration')

    axs_1 = subfig[1].subplots(2, 1, sharex=True, sharey=False)
    axs_1[0].plot(save_ε[1, 0, :].T, color=cl_wh, label='whisking')
    axs_1[1].plot(save_ε[1, 1, :].T, color=cl_rs, label='respiration')

    # Set xticks and label
    axs_0[1].set_xticks(sim_t[::5] / dt_gp)
    axs_0[1].set_xticklabels(sim_t[::5].astype(int).tolist())
    axs_0[1].set_xlabel('sec')
    axs_1[1].set_xticks(sim_t[::5] / dt_gp)
    axs_1[1].set_xticklabels(sim_t[::5].astype(int).tolist())
    axs_1[1].set_xlabel('sec')

    # Set y label
    axs_0[0].set_ylabel('a.u.')
    axs_0[1].set_ylabel('a.u.')
    axs_1[0].set_ylabel('a.u.')
    axs_1[1].set_ylabel('a.u.')

    # Add vertical lines, legend and title + remove frame
    # state_int_csum = np.cumsum(gp.state_interval)
    for idx, _ in enumerate(axs):
        axs_0[idx].axvline(state_int_csum[0], ls='--', lw='3', c='salmon')
        axs_1[idx].axvline(state_int_csum[0], ls='--', lw='3', c='salmon')
        axs_0[idx].axvline(state_int_csum[1], ls='--', lw='3', c='salmon')
        axs_1[idx].axvline(state_int_csum[1], ls='--', lw='3', c='salmon')
        # axs_0[idx].axvline(state_int_csum[2], ls='--', lw='3', c='salmon')
        # axs_1[idx].axvline(state_int_csum[2], ls='--', lw='3', c='salmon')
        # axs_0[idx].axvline(state_int_csum[3], ls='--', lw='3', c='salmon')
        # axs_1[idx].axvline(state_int_csum[3], ls='--', lw='3', c='salmon')
        axs_0[idx].legend(loc="upper right",
                          handlelength=0,
                          bbox_to_anchor=(1.13, 1))
        axs_0[idx].spines['top'].set_visible(False)
        axs_0[idx].spines['right'].set_visible(False)
        axs_1[idx].legend(loc="upper right",
                          handlelength=0,
                          bbox_to_anchor=(1.13, 1))
        axs_1[idx].spines['top'].set_visible(False)
        axs_1[idx].spines['right'].set_visible(False)

    subfig[0].suptitle('Prediction errors y')
    subfig[1].suptitle('Prediction errors x')

    # Plot together (compare)
    fig6, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True, sharey=True)
    axs[0].plot(save_u[0, 0, :].T, label='hidden state')
    axs[0].plot(save_u[2, 0, :].T, label='hidden cause')
    # axs[0].plot(save_gp[0, :-1].T, color=cl_wh, label='wh state')
    axs[0].plot(save_obs[0, :].T, color=cl_wh, label='whisking')
    axs[1].plot(save_u[0, 1, :].T, label='hidden state')
    axs[1].plot(save_u[2, 1, :].T, label='hidden cause')
    # axs[1].plot(save_gp[3, :-1].T, color=cl_rs, label='respiration')
    axs[1].plot(save_obs[1, :].T, color=cl_rs, label='respiration')

    # Add vertical lines, legend and title + remove frame
    # state_int_csum = np.cumsum(gp.state_interval)
    for idx, _ in enumerate(axs):
        axs[idx].axvline(state_int_csum[0], ls='--', lw='3', c='salmon')
        axs[idx].axvline(state_int_csum[1], ls='--', lw='3', c='salmon')
        # axs[idx].axvline(state_int_csum[2], ls='--', lw='3', c='salmon')
        # axs[idx].axvline(state_int_csum[3], ls='--', lw='3', c='salmon')
        axs[idx].legend(loc="upper right",
                        handlelength=0,
                        bbox_to_anchor=(1.13, 1))
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)

    # Set xtics and label
    axs[1].set_xticks(sim_t[::5] / dt_gp)
    axs[1].set_xticklabels(sim_t[::5].astype(int).tolist())
    axs[1].set_xlabel('sec')

    # Set y label
    axs[0].set_ylabel('a.u.')
    axs[1].set_ylabel('a.u.')


    # Add vertical lines, legend and title + remove frame
    for idx, _ in enumerate(axs):
        axs[idx].axvline(state_int_csum[0], ls='--', lw='3', c='salmon')
        axs[idx].axvline(state_int_csum[1], ls='--', lw='3', c='salmon')
        # axs[idx].axvline(state_int_csum[2], ls='--', lw='3', c='salmon')
        # axs[idx].axvline(state_int_csum[3], ls='--', lw='3', c='salmon')
        # if idx == 0 | idx == 2:
        axs[idx].legend(loc="upper right", bbox_to_anchor=(1.13, 1))
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)

    fig6.suptitle('Compare')

    # Plot VFA and PE
    df = pd.DataFrame(save_fa).T
    df = df.melt(ignore_index=False)

    fig7, ax7 = plt.subplots(figsize=(12, 12))
    sns.lineplot(data=df, x=df.index, y='value', hue='variable', ax=ax7)
    # sns.lineplot(data=df, x=df.index, y='value', ax=ax7)
    # Save Free Action
    # free_action = save_fa[0]

    # fig7, ax7 = plt.subplots(figsize=(12, 12))
    # ax7.plot(free_action)

    # # # Add horizontal line
    # ax7.axhline(np.cumsum(gm.vfe)[-1], color='black', linestyle='--')
    # ax7.set_yticks(
    #     list(plt.yticks()[0]) + [free_action[-1]],
    #     list(plt.yticks()[0]) + [(free_action[-1])])

    # Set xticks and label
    ax7.set_xticks(sim_t[::5] / dt_gp)
    ax7.set_xticklabels(sim_t[::5].astype(int).tolist())
    ax7.set_xlabel('sec')


    # Set y label
    ax7.set_ylabel('Free action')

    # Add vertical lines and title + remove frame
    ax7.axvline(state_int_csum[0], ls='--', lw='3', c='salmon')
    ax7.axvline(state_int_csum[1], ls='--', lw='3', c='salmon')
    # ax7.axvline(state_int_csum[2], ls='--', lw='3', c='salmon')
    # ax7.axvline(state_int_csum[3], ls='--', lw='3', c='salmon')
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)

    fig7.suptitle('Variational action')

    # Plot difference
    fig8, ax8 = plt.subplots(3, 2, figsize=(12, 12), sharex=True, sharey=False, gridspec_kw={'width_ratios': (5, .1), 'wspace': 0})
    # Plot
    ax8[0, 0].plot(save_gp[0, 1:], color=cl_wh, label='whisking')
    ax8[0, 0].plot(save_gp[3, 1:], color=cl_rs, label='respiration')
    ax8[1, 0].plot(np.array(gp.diff_wh_rs), label='whisking-respiration')
    spectrogram = ax8[2, 0].imshow(np.abs(diff_wt), aspect='auto', origin='lower')
    # ax8[2].pcolormesh(np.arange(n_step), freq, np.abs(diff_wt))

    # Set xticks and label
    ax8[2, 0].set_xticks(sim_t[::5] / dt_gp)
    ax8[2, 0].set_xticklabels(sim_t[::5].astype(int).tolist())
    ax8[2, 0].set_xlabel('sec')

    # Set yticks and label for heatmap
    ax8[2, 0].set_yticks(ax8[2, 0].get_yticks()[1:])
    ax8[2, 0].set_yticklabels((ax8[2, 0].get_yticks() / 10).astype(int))

    # Set ylim for difference plot
    ax8[1, 0].set_ylim([-2, 2])

    # Show axis, add vertical lines, ylabel, and title + remove frame
    for idx, __ in enumerate(ax8):
        ax8[idx, 0].axvline(state_int_csum[0], ls='--', lw='5', c='salmon')
        ax8[idx, 0].axvline(state_int_csum[1], ls='--', lw='5', c='salmon')
        # ax8[idx].axvline(state_int_csum[2], ls='--', lw='3', c='salmon')
        # ax8[idx].axvline(state_int_csum[3], ls='--', lw='3', c='salmon')
        ax8[idx, 0].spines['top'].set_visible(False)
        ax8[idx, 0].spines['right'].set_visible(False)
        if idx != 2:
            ax8[idx, 0].set_ylabel('a.u.')
        else:
            ax8[idx, 0].set_ylabel('Hz')
            # Display ticks when sharing axis
        ax8[idx, 0].xaxis.set_tick_params(labelbottom=True)

    # Set labels and heatmap
    # ax8[0, 0].legend(bbox_to_anchor='top right')
    ax8[0, 0].legend(loc='upper right')
    cbar = plt.colorbar(mappable=spectrogram, ax=ax8[2, 1], fraction=0.4)
    cbar.set_label('power')

    # Hide unused axes
    for ax in range(3):
        ax8[ax, 1].set_axis_off()

    # # Set y label
    # ax8[1].set_ylabel('a.u.')
    fig8.suptitle('Difference whisking - respiration')

    # Plot state space
    # pdb.set_trace()
    col_interval = ['violet','tomato','mediumseagreen','y']
    time_sim = 0
    fig9, ax9 = plt.subplots(figsize=(12, 12))
    for idx, interval in enumerate(gp.state_interval):
        t_slice = np.arange(time_sim, time_sim + interval)
        ax9.plot(save_gp[0, t_slice], save_gp[3, t_slice], c=col_interval[idx])
        time_sim += interval

    # Save figures
    # os.path.join(os.getcwd() + 'fig' + )
    # Plot or save figures
    if plotfig:
        plt.show()
        # pdb.set_trace()
    else:
        # Check cwd
        # pdb.set_trace()
        if not ('state_estimation' in os.getcwd()):
            sys.path.append(os.path.join(os.getcwd(), 'state_estimation'))

        # Name of save_path
        save_path = os.path.join(os.getcwd(), 'save_figures', simcond)

        # Check if save_path exists
        if not(os.path.lexists(save_path)):
            os.makedirs(save_path)

        # Save figures
        if np.allclose(gm.θ_f, np.eye(2)):
            fig5.savefig(save_path + '/pe_nosync.svg')
            fig6.savefig(save_path + '/gpgm_nosync.svg')
            fig7.savefig(save_path + '/fa_nosync.svg')
            fig8.savefig(save_path + '/diff_nosync.svg')
            fig9.savefig(save_path + '/ss_nosync.svg')
        else:
            fig5.savefig(save_path + '/pe_sync.svg')
            fig6.savefig(save_path + '/gpgm_sync.svg')
            fig7.savefig(save_path + '/fa_sync.svg')
            fig8.savefig(save_path + '/diff_sync.svg')
            fig9.savefig(save_path + '/ss_sync.svg')

        plt.show()



if __name__ == "__main__":
    """ Run simulation display and save results; simulation of gp has dt=0.005
    and gm learning rate is κ ~ 0.1, thus one order of magnitude less;
    for generative model used dt=1, therefore running at same speed as
    generative process; use cov_module.cov_target() function to estimate
    covariance matrix in targer (synchronised) scenario.
    """
    # Simulation parameters
    n_step = 1500               # Simulation length
    dt_gp = 0.005               # Time constant GP
    dt_gm = 1                   # Time constant GM (= GP)
    rng = 100                   # rng seed
    n_run = 50                  # number or simulation runs
    plotfig = True              # if False: save figures

    # Parameters for behaviour and cerebellar state estimation 
    dw = 0                      # Noise gp dynamics
    # dw = 6                      # Noise gp dynamics
    ω_offset = np.array([.1, .1]) * 2 * np.pi  # Offset angular frequency respiration
    # ω_offset = np.array([.0, .0])  # Offset angular frequency respiration
    perturb = True              # whether or not perturb oscillator phase
    # perturb = False              # whether or not perturb oscillator phase
    expect_sync = True       # Control expectations in pp connectivity
    action = True             # Action
    obs_noise = False        # no observation noise

    # Run simulations
    runsim(n_step=n_step, dt_gp=dt_gp, dt_gm=dt_gm, dw=dw, ω_offset=ω_offset, perturb=perturb, expect_sync=expect_sync, action=action, n_run=n_run, rng=rng, obs_noise=obs_noise, plotfig=plotfig)
