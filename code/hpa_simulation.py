from scipy.integrate import solve_ivp
import numpy as np
import scipy.stats as st
import pickle
import argparse
from scipy.optimize import curve_fit


DEFAULT_PARS = {  # days
    "gamma_x1": np.log(2)/4*24*60,
    "gamma_x2": np.log(2)/20*24*60,
    "gamma_x3": np.log(2)/80*24*60,
    "gamma_P": 1/30, # according to Nolan et al. and Kataoka et al. and our first hair cortisol paper
    "gamma_A": 1/30, # according to Nolan et al. and Kataoka et al. and our first hair cortisol paper
}

PARS = DEFAULT_PARS.copy()


def HPA_GLANDS(t, y, ut=lambda t: 1, pars=PARS):

    kGR = 4 # the deault is 4
    n = 3

    def MR(x): return 1/x
    def GR(x): return np.divide(1, np.power(np.divide(x, kGR), n) + 1)

    x1, x2, x3, P, A = y
    u = ut(t)

    dx1 = pars["gamma_x1"]*(u*MR(x3)*GR(x3) - x1) # with GR
    # dx1 = pars["gamma_x1"]*(u*MR(x3) - x1) # without GR

    dx2 = pars["gamma_x2"]*(x1*P*GR(x3) - x2)   # with GR
    # dx2 = pars["gamma_x2"]*(x1*P - x2)     # without GR

    dx3 = pars["gamma_x3"]*(x2*A - x3)

    dP = pars["gamma_P"]*P*(x1 - 1)
    dA = pars["gamma_A"]*A*(x2 - 1)

    return [dx1, dx2, dx3, dP, dA]


def HPA_NO_GLANDS(t, y, ut=lambda t: 1, pars=PARS):

    kGR = 4 # the deault is 4
    n = 3

    def MR(x): return 1/x
    def GR(x): return np.divide(1, np.power(np.divide(x, kGR), n) + 1)

    x1, x2, x3 = y
    u = ut(t)

    dx1 = pars["gamma_x1"]*(u*MR(x3)*GR(x3) - x1) # with GR
    # dx1 = pars["gamma_x1"]*(u*MR(x3) - x1) # without GR

    dx2 = pars["gamma_x2"]*(x1*GR(x3) - x2)   # with GR
    # dx2 = pars["gamma_x2"]*(x1 - x2)     # without GR

    dx3 = pars["gamma_x3"]*(x2 - x3)

    return [dx1, dx2, dx3]


def generate_sin_sum_func(freqs, amplitude, base=0, exp=True, random_state=None):

    N_freqs = len(freqs)
    phases = st.uniform.rvs(loc=-np.pi, scale=2*np.pi, size=N_freqs, random_state=random_state)

    if exp:
        return lambda t: np.exp(base + (1/N_freqs)*amplitude*np.sum(np.sin(2*np.pi*t*freqs + phases)))
    else:
        return lambda t: base + (1/N_freqs)*amplitude*np.sum(np.sin(2*np.pi*t*freqs + phases))


def average_bimonthly(signal, time_spacing, time_unit='day'):
    '''
    @params:
        time_spacing - the time interval between 2 data points
        time_unit - 'day' / 'min'

    @returns:
        t_bimonths
        bimonthly_signal
    '''

    if time_unit == 'day':
        bimonths = 60
    elif time_unit == 'min':
        bimonths = 60 * 24 * 60
    else:
        raise ValueError(
            f'receive time_unit = {time_unit}. should be day or min')

    N_bimonths = int(round((len(signal) - 1) * time_spacing / bimonths))
    t_bimonths = np.arange(N_bimonths) * bimonths + int(bimonths/2)

    bimonthly_signal = np.array(
        [np.mean(signal[int(i*bimonths/time_spacing):int((i+1)*bimonths/time_spacing)])
         for i in range(N_bimonths)]
    )

    return t_bimonths, bimonthly_signal


def normalize_by_fitting(data, t, sigma=None, bounds=(-np.inf, np.inf), exp=True):
    '''
    normalizes the data by canceling the exponential component of it

    @params:
        data - [#time X #samples]
        sigma - [#time X #samples], should contain values of standard deviations of errors in data.
                In this case, the optimized function is ``chisq = sum((r / sigma) ** 2)``
        bounds - 2-tuple of array_like, optional
                 Lower and upper bounds on parameters. Defaults to no bounds.
                 Each element of the tuple must be either an array with the length equal
                 to the number of parameters, or a scalar (in which case the bound is
                 taken to be the same for all parameters.) Use ``np.inf`` with an
                 appropriate sign to disable bounds on all or some parameters.

    @return:
        normalized data - data.shape
        fitted params - [#samples X #params]
        params_std
    '''

    _sigma = sigma
    def linear(t, A, alpha): return np.log(A) - alpha*t

    fitted_params = np.zeros((data.shape[1], 2))
    params_std = np.zeros_like(fitted_params)
    data_normalized = np.zeros_like(data)
    for i in range(data.shape[1]):
        y = np.log(data[:, i])
        if sigma is not None:
            _sigma = sigma[:, i]

        popt, pcov = curve_fit(linear, t, y, sigma=_sigma, bounds=bounds)
        data_normalized[:, i] = np.log(data[:, i]) - np.log(popt[0]) + popt[1]*t
        if exp:
            data_normalized[:, i] = np.exp(data_normalized[:, i])
        fitted_params[i] = popt
        params_std[i] = np.sqrt(np.diag(pcov))

    return data_normalized, fitted_params, params_std


def calculate_fft(data, time_samples):
    '''
    data - [#vars X #time_points]
    '''

    data = data.T

    return {
        'amps': np.absolute(np.fft.rfft(data, axis=0)).T,
        'phases': np.angle(np.fft.rfft(data, axis=0)).T,
        'freqs': np.fft.rfftfreq(len(time_samples), d=time_samples[1]-time_samples[0])
    }


def run(freqs, input_baseline, input_amplitude, alpha=None, with_glands=True, random_state=None):
    '''
    @params:
        freqs - frequencies to generate the input from
        input_baseline - the mean of the input noise (before exponentiating)
        input_amplitude - the amplitude of the input (before exponentiating)
        alpha - a decaying coefficeint to simulate hair cortisol experiment, in units [1/day]
        with_glands - True for model with glands. False for model without glands
    '''

    ut = generate_sin_sum_func(freqs=freqs, amplitude=input_amplitude, base=input_baseline, exp=True, random_state=random_state)

    # simulate full cortisol time series ("blood cortisol")
    t = np.arange(0, 4*360)
    sol = solve_ivp(
        fun=HPA_GLANDS if with_glands else HPA_NO_GLANDS,
        t_span=[t[0], t[-1]],
        y0=[1]*5 if with_glands else [1]*3,
        t_eval=t,
        args=(ut, PARS),
    )
    sampled_u = np.array([ut(i) for i in t])

    res = {
        't': t,
        'u': sampled_u,
        'u_fft': calculate_fft(sampled_u, t),
        'sol': sol,
        'sol_fft': calculate_fft(sol.y, t),
    }

    if (not sol.success) or (alpha is None):
        return res

    # last year cortisol:
    last_year_first_day = sol.t[-1] - 360
    last_year_idx = sol.t > last_year_first_day
    last_year_t = sol.t[last_year_idx] - last_year_first_day  # in order to make t [0, 360]
    last_year_cortisol = sol.y[2][last_year_idx]
    res['t_last_year'] = last_year_t

    # inserting decay
    decaying_cortisol = last_year_cortisol * np.exp(-alpha*last_year_t)
    res['alpha'] = alpha
    res['decaying_cortisol'] = decaying_cortisol
    res['decaying_cortisol_fft'] = calculate_fft(decaying_cortisol, last_year_t)

    # averaging bi-monthly
    t_bimonths, bimonthly_cortisol = average_bimonthly(decaying_cortisol, last_year_t[1]-last_year_t[0])
    res['t_bimonths'] = t_bimonths
    res['bimonthly_cortisol'] = bimonthly_cortisol
    res['bimonthly_cortisol_fft'] = calculate_fft(bimonthly_cortisol, t_bimonths)

    ################################################################
    # Up until here we simulated the 6 segments of measured cortisol
    ################################################################

    # Normalizing the simulated measured cortisol as we do with the real data
    normalized_cortisol, _, _ = normalize_by_fitting(bimonthly_cortisol[:, np.newaxis], t_bimonths, bounds=([0, 0.0001], np.inf))
    normalized_cortisol = normalized_cortisol.reshape(len(normalized_cortisol))
    res['normalized_cortisol'] = normalized_cortisol
    res['normalized_cortisol_fft'] = calculate_fft(normalized_cortisol, t_bimonths)

    return res


def step(t, val_f, ts, val_i=1):
    
    if t < ts:
        return val_i
    else:
        return val_f


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
#     parser.add_argument('-I', dest='config_filepath', default='', help='parameters configuration file path')
    parser.add_argument('-A', dest='alpha', default=None, help='the rate of decaying cortisol')
    parser.add_argument('-R', dest='random_state', default=None, help='the random seed for this run')
    parser.add_argument('output_filepath', help='output file path to store the results')
    args = parser.parse_args()

    with_glands = False

    # Input parameters - TODO: incorporate it into a configuration
    freqs = np.arange(1, 361)/(4*360) # 360 frequncies in 4 years (from 1 in 4 years to 360 in 4 years (90 times in a year))
    input_baseline = np.log(20)
    input_amplitude = 28

    alpha = None
    if args.alpha is not None:
        alpha = float(args.alpha)

    random_state = None
    if args.random_state is not None:
        random_state = int(args.random_state)

    res = run(freqs, input_baseline, input_amplitude, alpha, with_glands=with_glands, random_state=random_state)
    with open(args.output_filepath, 'wb') as f:
        pickle.dump(res, f)