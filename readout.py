from scipy.integrate import ode, quad
from scipy.stats import norm
import numpy as np
from numpy import random

class readout_simulation:

    def __init__(self, b_in, freq_mm, chi, kappa, T1e, T1g, alpha_0, t_sim, dt):

        # Takes a function of t representing the baseband signal being sent to the 
        # cavity. Its magnitude is proportional to the square root of photon flux.
        self.b_in = b_in

        # Frequency mismatch between the input signal and the cavity frequency
        self.freq_mm = freq_mm

        # Dispersive shift on cavity frequency as determined by the transmon
        # state (Hz)
        self.chi = chi

        # Coupling between resonator and transmission line (Hz)
        self.kappa = kappa

        # Characteristic time of e -> g transmon transition 
        self.T1e = T1e

        # Characteristic time of g -> e transmon transition 
        self.T1g = T1g

        # Initial state of the cavity (coherent state amplitude)
        self.alpha_0 = alpha_0

        # Total simulation time (s)
        self.t_sim = t_sim
        
        # Simulation timestep (s)
        self.dt = dt

        # Theoretical expected values of measured phase theta of the reflected wave
        # when the transmon is in the ground or excited state (degrees). Assumes b_in is a 
        # single-frequency pulse. The calculation is performed using the reflection coefficient. 
        self.exp_theta = {'g': np.angle((-1j*chi - kappa)/(-1j*chi + kappa), deg = True),
                          'e': np.angle((1j*chi - kappa)/(1j*chi + kappa), deg = True)}

    def resonator_EOM(self, t, alpha, q_state): 
        '''
        Returns the derivative of alpha at a given t
        '''

        return (1j*(-1)**q_state*self.chi/2 - self.kappa/2)*alpha - 1j*self.kappa**0.5*self.b_in(t)

    def simulate_cavity_dynamics(self, q_state):
        '''
        This function runs a single readout including the simulation of system's dynamics, 
        digital signal processing and theta estimation. However, it assumes an initial value of the qubit
        and also allows for a finite relaxation rate as defined by T1g and T1e.
        '''

        Ndt = int(self.t_sim/self.dt) + 1

        halt_simulation = False
        alpha_t = self.alpha_0
        flip_prob_per_step = self.dt/(self.T1e if q_state else self.T1g)
        t_rng = [0.]
        cavity_alpha = [self.alpha_0]

        while not halt_simulation:

            rg = ode(self.resonator_EOM)
            rg.set_integrator('zvode', method = 'bdf').set_initial_value(cavity_alpha[-1], t_rng[-1]).set_f_params(q_state)
            alpha_list = []

            while rg.successful() and rg.t < self.t_sim:
                t_rng.append(rg.t + self.dt)
                cavity_alpha.append(rg.integrate(rg.t + self.dt))
                if random.random() < flip_prob_per_step:
                    q_state = 0 if q_state else 1
                    flip_prob_per_step = self.dt/(self.T1e if q_state else self.T1g)
                    break
        
            # Checks if simulation is complete
            if rg.t >= self.t_sim:
                halt_simulation = True

        b_out = [-1j*self.kappa**0.5*cavity_alpha[i] + self.b_in(t_rng[i]) for i in range(len(t_rng))]

        return b_out, cavity_alpha, t_rng

    def digital_signal_processing(self, b_out, t_rng, t_wait, t_acq, dt_acq): 
        '''
        Simulates the acquisition of b_out and remove the frequency mismatch.
        Inputs:
        ::t_wait:: Waiting time before starting acquisition (s)
        ::t_acq:: Total acquisition time (s)
        ::dt_acq:: Acquisition interval (s). It should be a multiple of dt
        Outputs:
        ::acq_b_out:: Dictionary containing the digitalized output signal for each acquisition. 
                      The ground (excited) state is stored in key = 'g' ('e').
        ::dsp_b_out:: Dictionary containing the digitalized output signal for each acquisition
                       step after the frequency mismatch is removed. The ground (excited) state
                       is stored in key = 'g' ('e').
        ::acq_time:: Acquisition timestamps (s).
        '''

        Nt_acq = round(t_acq/dt_acq)    # number of samples
        dindex = round(dt_acq/self.dt)

        # Collecting samples from the output signal
        start_index = t_rng.index(sorted(t_rng, key = lambda x: (t_wait - x)**2)[0])
        end_index = t_rng.index(sorted(t_rng, key = lambda x: ((t_wait + t_acq) - x)**2)[0])

        acq_b_out = []
        acq_time = []
        for n in range(Nt_acq):
            sampling_index = start_index + n*dindex
            acq_b_out.append(b_out[sampling_index])
            acq_time.append(t_rng[sampling_index])

        # Manually removing the frequency mismatch
        dsp_b_out = [x*np.exp(-1j*2*np.pi*self.freq_mm*dt_acq*n) 
                            for n, x in enumerate(acq_b_out)]

        # Adding quantum noise to the acquired data
        noisy_proc_b_out = {}

        # Variance of quadrature component in coherent states
        quad_var = 1/4

        # Get random noise
        bidimensional_noise_array = random.multivariate_normal([0, 0], [[quad_var, 0], [0, quad_var]], size = len(acq_time))
        complex_noise_array = [x[0] + 1j*x[1] for x in bidimensional_noise_array]

        noisy_dsp_b_out = [complex_noise_array[i] + dsp_b_out[i] for i in range(len(dsp_b_out))]

        return acq_b_out, dsp_b_out, noisy_dsp_b_out, acq_time


    def theta_estimation(self, noisy_dsp_b_out, dt_acq): 
        '''
        Extracts theta estimation for the whole readout for ground and excited state.
        ::noisy_proc_b_out:: noisy post-processed data measured from b_out as defined in digital_signal_processing.
        ::dt_acq:: Acquisition interval (s). Same as defined for digital_signal_processing.
        Outputs:
        ::theta_out:: Dictionary containing the each measurement of theta. 
                      The ground (excited) state is stored in key = 'g' ('e').
        ::acq_time:: Timestamps for each theta measurement (s).
        '''

        # The phase estimation is made by integrating the values of theta in time
        def integrate_theta(noisy_theta, N = -1):
            if N == -1:
                int_theta = dt_acq*sum(noisy_theta)
            else:
                int_theta = dt_acq*sum(noisy_theta[0:N])
            return int_theta

        acq_time = [i*dt_acq for i in range(len(noisy_dsp_b_out))]

        theta_out = [float(np.angle(x, deg = True)) for x in noisy_dsp_b_out]

        int_theta_out = [integrate_theta(theta_out, N = i) for i in range(len(theta_out) + 1)]

        theta_estimate = [int_theta_out[i]/dt_acq/i for i in range(1, len(int_theta_out))]

        est_q_state = [0 if abs(self.exp_theta['g'] - th) < abs(self.exp_theta['e'] - th) else 1
                       for th in theta_estimate]

        return theta_out, int_theta_out, theta_estimate, est_q_state, acq_time

        
    def state_estimation(self, noisy_dsp_b_out, dt_acq): 
        '''
        Extracts state estimation for the whole readout for ground and excited state.
        The difference between theta_estimation and state_estimation is that the latter
        does state differentiation based on the signal IQ values projected onto the separatrix
        instead of its angle.
        Inputs:
        ::noisy_proc_b_out:: noisy post-processed data measured from b_out as defined in digital_signal_processing.
        ::dt_acq:: Acquisition interval (s). Same as defined for digital_signal_processing.
        Outputs:
        ::signal_out:: Dictionary containing the each measurement of output signal. 
                       The ground (excited) state is stored in key = 'g' ('e').
        ::acq_time:: Timestamps for each signal measurement (s).
        '''

        # Integrates signal in time
        def integrate_signal(noisy_signal, N = -1):
            if N == -1:
                int_signal = dt_acq*sum(noisy_signal)
            else:
                int_signal = dt_acq*sum(noisy_signal[0:N])
            return int_signal

        acq_time = [i*dt_acq for i in range(len(noisy_dsp_b_out))]

        # Define theoretically determined separatrix
        theta_e, theta_g = self.exp_theta['e'], self.exp_theta['g']
        exp_e = np.exp(1j*np.pi*theta_e/180)
        exp_g = np.exp(1j*np.pi*theta_g/180)
        separatrix = 1j*(exp_e + exp_g)/np.absolute(exp_e + exp_g)
        separatrix /= np.absolute(separatrix)
        proj_exp_g = (exp_g.real*separatrix.real + exp_g.imag*separatrix.imag)
        proj_exp_e = (exp_e.real*separatrix.real + exp_e.imag*separatrix.imag)

        # Project noisy post-processed signal onto separatrix to get maximum state distinction
        signal_out = [(x.real*separatrix.real + x.imag*separatrix.imag)[0] for x in noisy_dsp_b_out]

        # Integrate signal
        int_signal_out = [integrate_signal(signal_out, N = i) for i in range(len(signal_out) + 1)]

        signal_estimate = [int_signal_out[i]/dt_acq/i for i in range(1, len(int_signal_out))]

        est_q_state = [0 if abs(proj_exp_g - sn) < abs(proj_exp_e - sn) else 1
                       for sn in signal_estimate]

        return signal_out, int_signal_out, signal_estimate, est_q_state, acq_time

    def do_readout(self, q_state, t_wait, t_acq, dt_acq):
        '''
        '''

        # Simulate the system's dynamics
        b_out, cavity_alpha, t_rng = self.simulate_cavity_dynamics(q_state)

        # Does the acquisition and DSP of the received signal
        acq_b_out, dsp_b_out, noisy_dsp_b_out, acq_time = self.digital_signal_processing(b_out, t_rng, t_wait, t_acq, dt_acq)

        # Estimates the value of theta
        theta_out, int_theta_out, theta_estimate, est_q_state, acq_time = self.theta_estimation(noisy_dsp_b_out, dt_acq)

        return theta_out, int_theta_out, theta_estimate, est_q_state, acq_time

    def do_multiple_readouts(self, q_state, N, t_wait, t_acq, dt_acq): 
        '''
        This function runs readout simulations N times to allow a statistical evaluation of the process.
        It calculates the evolution of fidelity as a function of the number of acquisition steps.
        '''

        Nt_acq = round(t_acq/dt_acq)
        # Separate the data of each acquisition step into a separated array for gaussian fit 
        acq_steps_data =  [[] for _ in range(Nt_acq + 1)]
        est_q_state_steps =  [[] for _ in range(Nt_acq)] 
        
        # Save the integrated signal measurement  
        int_signal_data = []

        # Do readout N times to account for the noise and qubit flips
        for n in range(N):
            # Run readout
            b_out, cavity_alpha, t_rng = self.simulate_cavity_dynamics(q_state)
            acq_b_out, dsp_b_out, noisy_dsp_b_out, acq_time = self.digital_signal_processing(b_out, t_rng, t_wait, t_acq, dt_acq)
            signal_out, int_signal_out, signal_estimate, est_q_state, acq_time = self.state_estimation(noisy_dsp_b_out, dt_acq)
            
            # Save the data
            int_signal_data.append(int_signal_out)
            for k in range(len(int_signal_out)):
                acq_steps_data[k].append(int_signal_out[k])
            for k in range(len(est_q_state)):
                est_q_state_steps[k].append(est_q_state[k])

        # Make a gaussian fit for each step in acq_steps_data
        gaussian_fit = []
        for n in range(1, len(acq_steps_data)):
            gaussian_fit.append(norm.fit(acq_steps_data[n]))

        # Estimate the error rate by acquisition step by counting readout failures. 
        error_rate = []
        for est_q_state in est_q_state_steps:
            error_rate.append(1 - est_q_state.count(q_state)/len(est_q_state))

        return int_signal_data, acq_steps_data, gaussian_fit, error_rate, acq_time

