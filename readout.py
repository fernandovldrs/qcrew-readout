from scipy.integrate import ode, quad
from scipy.stats import norm
import numpy as np
from numpy import random

class readout_simulation:

    def __init__(self, b_in, freq_mm, chi, kappa, alpha_0, t_sim, dt):

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

        # Initial state of the cavity (coherent state amplitude)
        self.alpha_0 = alpha_0

        # Total simulation time (s)
        self.t_sim = t_sim
        
        # Simulation timestep (s)
        self.dt = dt
        
        # Arrays of output signal and cavity state at each simulation step, stored in t_rng
        self.b_out, self.cavity_alpha, self.t_rng = self.simulate_cavity_dynamics()

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

    def simulate_cavity_dynamics(self):
        '''
        This function takes the design parameters of the system and simulate the dynamics 
        of the cavity state alpha as a function of time.
        Outputs:
        ::b_out:: Dictionary containing output signal for when the transmon is in the ground
                state (key = 'g') and in the excited state (key = 'e').
        ::cavity_alpha:: Dictionary similar to b_out containing the state-dependent cavity state.
        ::t_rng:: Simulation timestap (s) corresponding to b_out and cavity_alpha values.
        '''

        Ndt = int(self.t_sim/self.dt) + 1
        t_rng = [x for x in np.linspace(0, self.t_sim, Ndt)]    # Time in microseconds

        b_out = {}
        cavity_alpha = {}
        for q_state in [0, 1]:
            out_key = 'e' if q_state else 'g'
                
            rg = ode(self.resonator_EOM)
            rg.set_integrator('zvode', method = 'bdf').set_initial_value(self.alpha_0, 0).set_f_params(q_state)
            alpha_val = []
            while rg.successful() and rg.t < self.t_sim:
                alpha_val.append(rg.integrate(rg.t + self.dt))

            cavity_alpha[out_key] = alpha_val
            b_out[out_key] = [-1j*self.kappa**0.5*alpha_val[x] + self.b_in(self.dt*x) for x in range(len(alpha_val))]

        self.b_out = b_out
        self.cavity_alpha = cavity_alpha
        self.t_rng = t_rng

        return b_out, cavity_alpha, t_rng

    def b_signal_proc(self, t_wait, t_acq, dt_acq): 
        '''
        Simulates the acquisition of b_out and remove the frequency mismatch.
        Inputs:
        ::t_wait:: Waiting time before starting acquisition (s)
        ::t_acq:: Total acquisition time (s)
        ::dt_acq:: Acquisition interval (s). It should be a multiple of dt
        Outputs:
        ::acq_b_out:: Dictionary containing the digitalized output signal for each acquisition. 
                      The ground (excited) state is stored in key = 'g' ('e').
        ::proc_b_out:: Dictionary containing the digitalized output signal for each acquisition
                       step after the frequency mismatch is removed. The ground (excited) state
                       is stored in key = 'g' ('e').
        ::acq_time:: Acquisition timestamps (s).
        '''

        Nt_acq = round(t_acq/dt_acq)    # number of samples
        t_rng = self.t_rng

        dindex = round(dt_acq/self.dt)

        # Collecting samples from the output signal
        start_index = t_rng.index(sorted(t_rng, key = lambda x: (t_wait - x)**2)[0])
        end_index = t_rng.index(sorted(t_rng, key = lambda x: ((t_wait + t_acq) - x)**2)[0])

        acq_b_out = {'g': [], 'e': []}
        acq_time = []
        for n in range(Nt_acq):
            sampling_index = start_index + n*dindex
            acq_b_out['e'].append(self.b_out['e'][sampling_index])
            acq_b_out['g'].append(self.b_out['g'][sampling_index])
            acq_time.append(t_rng[sampling_index])

        proc_b_out = {'g': [x*np.exp(-1j*2*np.pi*self.freq_mm*dt_acq*n) 
                            for n, x in enumerate(acq_b_out['g'])],
                      'e': [x*np.exp(-1j*2*np.pi*self.freq_mm*dt_acq*n) 
                            for n, x in enumerate(acq_b_out['e'])]}

        noisy_proc_b_out = {}

        # Standard deviation of quadrature components in coherent states
        quad_var = 1/4

        # Get random noise
        noise_array = random.multivariate_normal([0, 0], [[quad_var, 0], [0, quad_var]], size = 2*len(acq_time))
        noise_e = [x[0] + 1j*x[1] for x in noise_array[:len(acq_time)]]
        noise_g = [x[0] + 1j*x[1] for x in noise_array[len(acq_time):]]

        noisy_proc_b_out = {'g': [noise_g[i] + proc_b_out['g'][i] for i in range(len(proc_b_out['g']))],
                            'e': [noise_e[i] + proc_b_out['e'][i] for i in range(len(proc_b_out['e']))]}

        return acq_b_out, proc_b_out, noisy_proc_b_out, acq_time


    def theta_estimation(self, noisy_proc_b_out, dt_acq): 
        '''
        Extracts theta estimation for the whole readout for ground and excited state.
        ::noisy_proc_b_out:: noisy post-processed data measured from b_out as defined in b_signal_proc.
        ::dt_acq:: Acquisition interval (s). Same as defined for b_signal_proc.
        Outputs:
        ::theta_out:: Dictionary containing the each measurement of theta. 
                      The ground (excited) state is stored in key = 'g' ('e').
        ::acq_time:: Timestamps for each theta measurement (s).
        '''

        theta_out = {'g': [float(np.angle(x, deg = True)) for x in noisy_proc_b_out['g']],
                     'e': [float(np.angle(x, deg = True)) for x in noisy_proc_b_out['e']]}

        # The phase estimation is made by integrating the values of theta in time
        def integrate_theta(noisy_theta, N = -1):
            
            if N == -1:
                int_theta = dt_acq*sum(noisy_theta)
            else:
                int_theta = dt_acq*sum(noisy_theta[0:N])
            return int_theta

        int_theta_out = {'g': [integrate_theta(theta_out['g'], N = i) for i in range(len(theta_out['g']))],
                         'e': [integrate_theta(theta_out['e'], N = i) for i in range(len(theta_out['e']))]}

        theta_estimate = {'g': int_theta_out['g'][-1]/(dt_acq*len(int_theta_out['g'])),
                          'e': int_theta_out['e'][-1]/(dt_acq*len(int_theta_out['e']))}

        acq_time = [i*dt_acq for i in range(len(noisy_proc_b_out['g']))]

        return theta_out, int_theta_out, theta_estimate, acq_time

    #def do_readout

    def do_multiple_readouts(self, N, t_wait, t_acq, dt_acq): 
        '''
        This function runs readout simulations N times to allow a statistical evaluation of the process.
        It calculates the evolution of fidelity as a function of the number of acquisition steps.
        '''

        Nt_acq = round(t_acq/dt_acq)
        # Separate the data of each acquisition step into a separated array for gaussian fit 
        acq_steps_data =  {'g': [[] for _ in range(Nt_acq)], 'e': [[] for _ in range(Nt_acq)]}
        
        # Save the integrated theta meas. data for each of the N runs in a dictionary indexed by transmon state
        int_theta_data = {'g': [], 'e': []}

        # Do readout N times to account for the noise
        for n in range(N):
            # Run readout
            acq_b_out, proc_b_out, noisy_proc_b_out, acq_time = self.b_signal_proc(t_wait, t_acq, dt_acq)
            theta_out, int_theta_out, theta_estimate, acq_time = self.theta_estimation(noisy_proc_b_out, dt_acq)
            
            # Save the data
            int_theta_data['g'].append(int_theta_out['g'])
            int_theta_data['e'].append(int_theta_out['e'])
            for k in range(len(int_theta_out['e'])):
                acq_steps_data['g'][k].append(int_theta_out['g'][k])
                acq_steps_data['e'][k].append(int_theta_out['e'][k])

        acq_N = len(acq_time)

        # Make a gaussian fit for each step in acq_steps_data
        gaussian_fit = {'g':[], 'e':[]}
        for n in range(len(acq_steps_data['e'])):
            gaussian_fit['g'].append(norm.fit(acq_steps_data['g'][n]))
            gaussian_fit['e'].append(norm.fit(acq_steps_data['e'][n]))
            
        # Define the theoretical boundaries for state discrimination
        theta_bound = (self.exp_theta['g'] + self.exp_theta['e'])/2   # Theoretical phase discrimination boundary
        theta_t_bound = [theta_bound*dt_acq*i for i in range(acq_N)]   # theta_bound integrated in acquisition time 

        # Calculate the fidelity matrix of each acquisition step. 
        fidelity_matrices = []
        for n in range(1, len(gaussian_fit['g'])):
            mu_e, std_e = gaussian_fit['e'][n]
            mu_g, std_g = gaussian_fit['g'][n]
            gaussian_e = lambda x: norm.pdf(x, mu_e, std_e)
            gaussian_g = lambda x: norm.pdf(x, mu_g, std_g)
            
            if self.exp_theta['e'] > self.exp_theta['g']:
                err_e = quad(gaussian_e, -100*std_e, theta_t_bound[n])[0]
                err_g = quad(gaussian_g, theta_t_bound[n], 100*std_g)[0]
                    
            if self.exp_theta['e'] < self.exp_theta['g']:
                err_g = quad(gaussian_g, -100*std_g, theta_bound_t[n])[0]
                err_e = quad(gaussian_e, theta_t_bound[n], 100*std_e)[0]
            
            fidelity_matrices.append([[1-err_e, err_e],[err_g, 1-err_g]])

        return int_theta_data, acq_steps_data, gaussian_fit, fidelity_matrices, acq_time

