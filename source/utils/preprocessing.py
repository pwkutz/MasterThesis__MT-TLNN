import numpy as np 
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances
from glob import glob
import h5py
import json
from scipy.fft import fft, ifft

#import preprocessing
from source.troll_to_modelica import load_troll_data, transform_run

## raw TROLL data should be smoothed, because we are not interested in high frequencies. use after FFT 
def gauss_ma(current_signal, desired_lenght, overlap):
    step = round(len(current_signal)/desired_lenght)
    smoothed = []
    for i in range(1, (desired_lenght-overlap)):
        window = current_signal[round(i*step-step/2):round(i*step+step/2)]
        x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), len(window))
        if sum(window) > 0:
            smoothed.append(np.average(a = window, weights = norm.pdf(x)))
        else:
            smoothed.append(0)
    return np.array(smoothed)

def moving_average(current_signal, window_size):
    ret = np.cumsum(current_signal, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    smoothed = ret[window_size - 1:] / window_size
    return smoothed

# just cut off high frequencies
def filter_high_frequencies(data, sampling_rate, cutoff_frequency):
    # Perform FFT
    spectrum = fft(data)
    # Determine the frequency bin corresponding to the cutoff frequency
    num_samples = len(data)
    freq_bin = int(cutoff_frequency * num_samples / sampling_rate)
    # Zero out high frequency components
    spectrum[freq_bin:] = 0
    # Perform inverse FFT to obtain the filtered signal
    filtered_signal = ifft(spectrum)
    return np.real(filtered_signal)


# DISCLAIMER: This function is copied from https://github.com/nwhitehead/swmixer/blob/master/swmixer.py, 
#             which was released under LGPL. 
def resample_by_interpolation(signal, input_signal_length, output_signal_length):
    scale = output_signal_length / input_signal_length
    # calculate new length of sample
    n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

## arm acceleration should be apprx the same during any run, so it is enough to use one scan run (where no contact is occured)... ?
def arm_acceleration(troll_scan):
    ## using this https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    a_init = [np.mean(troll_scan['a_tcp'][0]), np.mean(troll_scan['a_tcp'][1]), np.mean(troll_scan['a_tcp'][2])]
    norm = np.sqrt(np.dot(a_init, a_init))
    b = [a_init[0]/norm, a_init[1]/norm, a_init[2]/norm]
    a = [0, 0, 1]
    v = np.cross(a, b)
    s = np.sqrt(np.sum(v**2))
    c = np.dot(a, b)
    v_x = np.zeros((3,3))
    #v_x[0, 0] = 
    v_x[0, 1] = -v[2]
    v_x[0, 2] = v[1]
    v_x[1, 0] = v[2]
    #v_x[1, 1] = 
    v_x[1, 2] = -v[0]
    v_x[2, 0] = -v[1]
    v_x[2, 1] = v[0]
    #v_x[2, 2] = 
    R = np.identity(3) + v_x + (1/(1+c)) * np.dot(v_x, v_x)
    # rotate tcp acceleration signal to match the forces signal
    a_tcp_rotated = np.dot(troll_scan['a_tcp'].transpose(), R)
    return a_tcp_rotated

def derivatives( X_test, gp_surrogate, y_surrogate):
    surrogate_derivatives = []
    #X_test = scm_test_X[0][:, :12]
    #real_pred = gp_surrogate.predict(X_test)
    for i in range(12):
        # analytical der
        K = gp_surrogate.kern.K(gp_surrogate.X, X_test)
        K_train = gp_surrogate.kern.K(gp_surrogate.X)
        l = gp_surrogate.kern.lengthscale[0]
        r = -euclidean_distances(gp_surrogate.X[:, i].reshape(-1, 1), X_test[:, i].reshape(-1, 1))/l**2
        dK = K * r
        K_train += 1e-6 * np.eye(K_train.shape[0])
        L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_surrogate))
        analytical_pred = np.dot(dK.T, alpha)
        surrogate_derivatives.append(analytical_pred)
        ## alternative analytical der 
        '''
        input_dim = 12
        active_dimensions = np.arange(0,input_dim)
        k1 = GPy.kern.RBF(input_dim)
        gp_mf = GPy.models.GPRegression(X=scm_X_train[:, :12], Y=scm_y_train, kernel=k1)
        gp_mf.optimize(max_iters = 500)

        real_pred = gp_mf.predict(scm_X_test[:, :12])

        K = gp_mf.kern.K(gp_mf.X, gp_mf.X)
        K_star = gp_mf.kern.K(gp_mf.X, scm_X_test[:, :12])

        cholezky_from_K = np.linalg.cholesky(K + 1e-6*np.eye(K.shape[0]))
        alpha = np.linalg.solve(cholezky_from_K, gp_mf.Y)
        custom_pred = np.dot(K, alpha)
        '''
        # numerical der
        '''
        X_test_plus_dx = scm_test_X[0][:, :12]
        X_test_plus_dx[:, i] += 1e-6

        pred_plus = gp_surrogate.predict(X_test_plus_dx)
        #pred_minus = gp_surrogate.predict(X_test_minus_dx)
        der = (pred_plus[0]-real_pred[0])/1e-6'''
    return np.transpose(np.array(surrogate_derivatives))

def define_troll_gravity_vec(nrow_troll, use_only_flat_runs = True):
    if use_only_flat_runs:
        gravity_vec = np.concatenate([np.zeros(nrow_troll)[:, None], np.zeros(nrow_troll)[:, None], -1*np.ones(nrow_troll)[:, None]], axis=1)
    else:
        ######### TODO ###########
        ### find a gravity angle using... soil profile/sinkage? Or this possible only with the upload of the surface? then only through Modelica? Or somehow run it once and store somewhere?
        gravity_vec = None
        ##########################
    return gravity_vec

def extract_data_from_run(method, troll_id, troll_raw_data_location, modelica_data_from_troll_location, json_w_start_times, json_w_stop_times, include_scans = False):
    # determine from which second the troll data are usable (after one full turn of the wheel)
    # for the next runs (with bumps, where v=0.15) haven't calculated starttime exactly - only approx
    # slippage-starttime relationaship is exponential
    with open(json_w_start_times) as data_file:
        analyazable_starttime = json.load(data_file)

    # time duration of each run can vary quite high, from 5 to 80 seconds
    with open(json_w_stop_times) as data_file:
        troll_runs_stop_time = json.load(data_file)

    # remeber to use specifi starting time-points (due to different slippage rate - we should analyze wheel's movement only after one full cicle of the wheel has been traversed)
    start_time = float([x for x in analyazable_starttime if troll_id in analyazable_starttime[x]][0])
    stop_time = float(troll_runs_stop_time[str(troll_id)])
    current_surface_scan = None
    if method == "troll":
        troll_run_name = glob(troll_raw_data_location+"*"+str(troll_id)+"*")[0]
        troll_run = load_troll_data(troll_run_name, mode="analysis")
        troll_dataset = transform_run(troll_run)
        start_indx = np.where(troll_dataset[:, 0] > start_time)[0][0]
        # filter out effect of a robotic arm acceleration
        a_tcp_rotated = arm_acceleration(troll_run)
        # compensate troll forces with the robotic arm acceleration
        current_y = troll_dataset[:, 14:16] + 20.97*a_tcp_rotated[:, :2] # compensation parameter 20.97 was found empirically. only traction force and y-force(or no? anyway, we don't care about it), don't compensate normal force and torques
        current_y = np.concatenate([current_y, troll_dataset[:, 16:20]], axis=1) # attached originaltorques and normal force
        current_y = current_y[start_indx:] 
        ## data extraction from TROLL is a bit different, but some Modelica code depends on it, so don't change 
        troll_gravity_vec = define_troll_gravity_vec(current_y.shape[0], use_only_flat_runs = True) 
        # convert troll X columns structure to the TerRA\SCM columns structure
        current_X = np.concatenate([troll_dataset[start_indx:, 1:3], troll_dataset[start_indx:, 5:7], troll_dataset[start_indx:, 11:13], troll_gravity_vec], axis = 1) 
        current_surface_scan = None # I don't read the surface from the raw TROLL surface scans. Just use ones from SCM,  anyway 
    else:
        filename = modelica_data_from_troll_location
        data = h5py.File(filename, 'r')
        data = data[str(troll_id)][method]
        start_indx = np.where(np.array([data["input"][x][0] for x in range(int(len(data['input'])))]) > start_time)[0][0]

        current_X = np.array([data["input"][x][1][:12] for x in range(int(len(data['input'])))])[start_indx:]
        if include_scans:
            current_surface_scan = np.array([data["input"][x][1][12:] for x in range(int(len(data['input'])))])[start_indx:]
        current_y = np.array([data["output"][x][1][:6] for x in range(int(len(data['output'])))])[start_indx:]
    return current_X, current_y, start_time, stop_time, current_surface_scan