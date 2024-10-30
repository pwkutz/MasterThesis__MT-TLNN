import pandas as pd 
import numpy as np
import h5py
import json
from tqdm import tqdm
from source.abstract_dataset import AbstrTerMechDataset
#import matplotlib.pylab as plt
from source.utils.preprocessing import moving_average, extract_data_from_run, filter_high_frequencies, resample_by_interpolation #arm_acceleration, gauss_ma,
from sklearn.metrics import mean_squared_error

# setting for tqdm
green = "\033[32;20;53m"
bar_format = f"{green}{{l_bar}}{{bar:50}} [{{elapsed}}]{{r_bar}}"

np.random.seed(96)

class DatasetTroll:
    def __init__(self, train_share = 1, use_only_flat_runs = False, selected_runs = None, troll_raw_data_location = None, modelica_data_from_troll_location = None, points_per_second = 10, cutoff_frequency = 5, smoothind_window_size = 10, deactivate_terra=True, include_scans = False, timestep = None, num_used_runs = None, json_w_runs_id = 'parameters/runs_id.json'): 
        self.troll_raw_data_location = troll_raw_data_location
        self.modelica_data_from_troll_location = modelica_data_from_troll_location
        self.train_share = train_share
        self.deactivate_terra = deactivate_terra
        self.include_scans = include_scans
        self.smoothind_window_size = smoothind_window_size

        with open(json_w_runs_id) as data_file:
            self.runs = json.load(data_file)
        self.flat_runs = np.array(self.runs["flat_runs"])
        self.bump_runs = np.array(self.runs["bump_runs"])
        self.points_per_second = points_per_second
        self.timestep = timestep
        #self.use_only_flat_runs = use_only_flat_runs
        self.cutoff_frequency = cutoff_frequency

        # by default we are using ALL flat runs
        if use_only_flat_runs and selected_runs == None:
            self.used_runs = self.flat_runs
        elif selected_runs != None:
            self.used_runs = selected_runs
        else:
            self.used_runs = np.concatenate([self.flat_runs, self.bump_runs])

        ### how many runs we want to use?
        if num_used_runs is not None:
            self.used_runs = np.random.choice(self.used_runs, num_used_runs, replace=False)

        # determine from which second the troll data are usable (after one full turn of the wheel)
        # for the next runs (with bumps, where v=0.15) haven't calculated starttime exactly - only approx
        # slippage-starttime relationaship is exponential
        # with open(json_w_start_times) as data_file:
        #    self.analyazable_starttime = json.load(data_file)

        # time duration of each run can vary quite high, from 5 to 80 seconds
        #with open(json_w_stop_times) as data_file:
        #    self.troll_runs_stop_time = json.load(data_file)

        terra, scm, troll = self.create_datasets(points_per_second)
        self.dataset = {"terra": terra,
                        "scm" : scm,
                        "troll": troll}

    def info():
        print("OUTDATED: Data generated from TROLL runs, conducted from X.12.2022 till 22.06.2023 with 52 (44 are actually used) flat runs and 13+N runs with surface distortions (bumps and pits). \n TerRA and SCM datasets were created on TROLL trajectories, using TrollInput from ContactDynamics library. Modified version of TrollInput which was used here could be found on dev-vlad branch. \n Numerical solver: Rkfix4 \n Parameters used in Modelica in order to adjust generated TerRA and SCM data to TROLL data: \n z_offset = 0.0115 for SCM amd 0.0065 for TerRA. \n scm_step = 0.03 \n overall modelica solver step == 0.03 (for SCM) and 0.001 for TerRA \n To check optimization or optimize data generation hyperparameters look in 'optimization.ipynb' \n To validate the data, call 'data_validation'")
   
    def return_sim_mse(self, method):
        # show average MSE for SCM or TerRA
        error = []
        for test_id in range(int(len(self.used_runs)*(1-self.train_share))):
            error.append(mean_squared_error(self.dataset[method].train_test_dataset['test']['y'][test_id][:, 0][:, None], 
                                            self.dataset['troll'].train_test_dataset['test']['y'][test_id][:, 0][:, None]))
        return error

    def smooth_simulations(self, unsmoothed_data, points_per_second, smoothind_window_size, start_time, stop_time, method):
        smoothed_data = []
        #make smooting over columns
        ## TODO change to frequency cut, doesn't work with Gaussian MA (might be working with MA)
        for i in range(unsmoothed_data.shape[1]):
            filtered_data = filter_high_frequencies(unsmoothed_data[:, i], points_per_second, self.cutoff_frequency)
            ma_smoothed = moving_average(filtered_data, window_size = smoothind_window_size)
            #### downsample smoothed version
            downsampled = resample_by_interpolation(ma_smoothed, len(ma_smoothed), int(stop_time-start_time)*points_per_second)
            smoothed_data.append(downsampled)
        smoothed_data = np.array(smoothed_data)
        smoothed_data = smoothed_data.transpose()
        return smoothed_data

    def create_train_test_dataset_for_one_fidelity_level(self, method, transfered_ids, points_per_second, smoothind_window_size):
        X = []
        y = []
        surfscans = []
        for troll_id in tqdm(transfered_ids, bar_format=bar_format, desc=method):
            current_X, current_y, current_strart_time, current_stop_time, current_surface_scan = extract_data_from_run(method = method, troll_id = troll_id, troll_raw_data_location = self.troll_raw_data_location, modelica_data_from_troll_location = self.modelica_data_from_troll_location, json_w_start_times = 'parameters/analyazable_starttime.json', json_w_stop_times = "parameters/runs_stop_time.json", include_scans = self.include_scans)
            
            smoothed_X = self.smooth_simulations(current_X, points_per_second, smoothind_window_size, current_strart_time, current_stop_time, method) 
            smoothed_y = self.smooth_simulations(current_y, points_per_second, smoothind_window_size, current_strart_time, current_stop_time, method)
            if method == "troll" or not self.include_scans:
                smoothed_current_surface_scan = current_surface_scan # they are None's anyway... TODO: find a better way to deal with surface absence in TROLL (include actuall hi-fi surface?)
            else:
                smoothed_current_surface_scan = self.smooth_simulations(current_surface_scan, points_per_second, smoothind_window_size, current_strart_time, current_stop_time, method)
            X.append(smoothed_X) 
            y.append(smoothed_y)
            '''
            #check how chosen smoothing changed the data (traction force)
            if method == "troll":
                plt.plot(np.linspace(0, 30, len(current_y[:, 0])),
                        current_y[:, 0], label='original')
                plt.plot(np.linspace(0, 30, len(smoothed_y[:, 0])), 
                        smoothed_y[:, 0], label='smoothed')
                plt.title("TROLL ID: "+str(troll_id))
                plt.legend()
                plt.show()
            '''
            surfscans.append(smoothed_current_surface_scan)
        return X, y, surfscans
        
    def get_train_test_runs_indeces(self):
        train_indeces = np.random.choice(range(len(self.used_runs)), size=round(len(self.used_runs)*self.train_share), replace=False) 
        train_ids = self.used_runs[train_indeces]
        test_ids = np.array([x for x in self.used_runs if x not in train_ids])
        return train_ids, test_ids

    def get_train_test_data(self, method, points_per_second, train_ids, test_ids, smoothind_window_size):
        print("create train")
        X_train, y_train, scans_train = self.create_train_test_dataset_for_one_fidelity_level(method, train_ids, points_per_second, smoothind_window_size)
        #X_train, y_train= np.concatenate(X_train), np.concatenate(y_train)
        if scans_train[0] is not None:
            scans_train = np.concatenate(scans_train)
        #print("created X train with shape: ", X_train.shape)
        #print("created y train with shape: ", y_train.shape)
        # don't concatenate test datasets
        print("create test")
        X_test, y_test, scans_test = self.create_train_test_dataset_for_one_fidelity_level(method, test_ids, points_per_second, smoothind_window_size)
        return X_train, y_train, scans_train, X_test, y_test, scans_test
    
    def create_datasets(self, points_per_second):
        train_ids, test_ids = self.get_train_test_runs_indeces()
        if self.deactivate_terra:
            terra = None
        else:
            terra = AbstrTerMechDataset(self.get_train_test_data("terra", points_per_second, train_ids, test_ids, self.smoothind_window_size), "terra", train_ids, test_ids)
        scm = AbstrTerMechDataset(self.get_train_test_data("scm", points_per_second, train_ids, test_ids, self.smoothind_window_size), "scm", train_ids, test_ids)
        troll = AbstrTerMechDataset(self.get_train_test_data("troll", points_per_second, train_ids, test_ids, self.smoothind_window_size*8), "troll", train_ids, test_ids)
        return terra, scm, troll

    def store_one_method_to_hdf(self, abstract_terramech_dataset, method, path_to_file):
        dataset = h5py.File(path_to_file, 'a')
        print("Storing " + method + " data to " + path_to_file)
        dataset.create_dataset('/' + method + '/X_train', data=abstract_terramech_dataset.X_train)
        dataset.create_dataset('/' + method + '/y_train', data=abstract_terramech_dataset.y_train)
        dataset.create_dataset('/' + method + '/X_test', data=abstract_terramech_dataset.X_test)
        dataset.create_dataset('/' + method + '/y_test', data=abstract_terramech_dataset.y_test)
        dataset.close()

    def save_all_to_hdf(self, path_to_file):
        self.store_one_method_to_hdf(self.terra, "terra", path_to_file)
        self.store_one_method_to_hdf(self.scm, "scm", path_to_file)
        self.store_one_method_to_hdf(self.troll, "troll", path_to_file)

    def save_to_csv(self):
        '''
        saving to csv format surface scans and tabular data, both train and test
        '''
        hf_surface = self.dataset["scm"].train_test_dataset['train']['surface']
        hf_y = self.dataset["troll"].train_test_dataset['train']['y']
        pd.DataFrame(np.concatenate([hf_surface, hf_y], axis=1)).to_csv("HF_surface_scans.csv", header=False, index=False)

        hf_X = self.dataset["scm"].train_test_dataset['train']['X']
        lf_y_from_hf = self.dataset["scm"].train_test_dataset['train']['y']
        hf_y = self.dataset["troll"].train_test_dataset['train']['y']
        pd.DataFrame(np.concatenate([hf_X, lf_y_from_hf, hf_y], axis=1)).to_csv("HF_w_LF_tabular_data.csv", header=False, index=False)

        alltestruns = []
        for indx in range(len(self.get_train_test_runs_indeces()[1])):
            rund_indx = self.get_train_test_runs_indeces()[1][indx]
            run_length = self.dataset["scm"].train_test_dataset['test']['X'][indx].shape[0]
            hf_X = self.dataset["scm"].train_test_dataset['test']['X'][indx]
            lf_y_from_hf = self.dataset["scm"].train_test_dataset['test']['y'][indx]
            hf_y = self.dataset["troll"].train_test_dataset['test']['y'][indx]
            alltestruns.append(np.concatenate([hf_X, lf_y_from_hf, hf_y, np.repeat(rund_indx, run_length)[:, None]], axis=1))

        pd.DataFrame(np.concatenate(alltestruns, axis=0)).to_csv("HF_w_LF_test_tabular_data.csv", header=False, index=False)