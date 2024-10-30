import pandas as pd
import numpy as np
import h5py
import math
from matplotlib import pyplot as plt
from math import ceil
from source.utils.preprocessing import gauss_ma, moving_average, filter_high_frequencies
from tqdm import tqdm
import logging
from source.utils.preprocessing import extract_data_from_run

class Validator:
    def __init__(self, train_test_dataset, troll_raw_data_location, modelica_data_from_troll_location, json_w_start_times = 'parameters/analyazable_starttime.json', json_w_stop_times = "parameters/runs_stop_time.json"):
        self.train_test_dataset = train_test_dataset
        #self.runs = dataset[0].runs_indeces
        self.troll_raw_data_location = troll_raw_data_location
        self.modelica_data_from_troll_location = modelica_data_from_troll_location
        self.json_w_start_times = json_w_start_times
        self.json_w_stop_times = json_w_stop_times

    def validate_dataset(self, method, train_or_test):
        print("Input variables")
        pd.DataFrame(self.train_test_dataset.dataset[method].train_test_dataset[train_or_test]["X"]).describe()
        print("Output variables")
        pd.DataFrame(self.train_test_dataset.dataset[method].train_test_dataset[train_or_test]["y"]).describe()

    def plot_against_time(self, feature_id, run_ids_list, X_or_y_datset = True, smoothind_window_size = 20): # if X_or_y_datset = True, use X dataset, if False - use y dataset
        
        #######################
        ### TODO: Remove the smoothing, dataset is already smoothed in dataset creation 
        #######################
        
        num_of_ids = min(57, len(run_ids_list)) # max = 57 runs
        #num_columns = ceil(num_of_ids/3) #int(np.sqrt(num_of_ids))
        fig, ax = plt.subplots(nrows=math.ceil(num_of_ids/3), ncols=3, figsize=(20, (num_of_ids//3)*7))#, sharey = True, sharex = True)
        #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))#, sharey = True, sharex = True)
        random_choice = np.random.choice(run_ids_list, num_of_ids, replace=False) 
        for i in range(num_of_ids):
            run_id = random_choice[i]
            #ax_ = ax
            if num_of_ids > 3:
                ax_ = ax[i//3, i%3]
            else:
                ax_ = ax[i%3]
            current_X_terra, current_y_terra, current_strart_time_terra, current_stop_time_terra, _ = extract_data_from_run("terra", run_id, self.troll_raw_data_location, self.modelica_data_from_troll_location, self.json_w_start_times, self.json_w_stop_times)
            current_X_scm, current_y_scm, current_strart_time_scm, current_stop_time_scm, _ = extract_data_from_run("scm", run_id, self.troll_raw_data_location, self.modelica_data_from_troll_location, self.json_w_start_times, self.json_w_stop_times)
            current_X_troll, current_y_troll, current_strart_time_troll, current_stop_time_troll, _ = extract_data_from_run("troll", run_id, self.troll_raw_data_location, self.modelica_data_from_troll_location, self.json_w_start_times, self.json_w_stop_times)
            
            if X_or_y_datset:
                data_terra = current_X_terra
                data_scm = current_X_scm
                data_troll = current_X_troll
            else:
                data_terra = current_y_terra
                data_scm = current_y_scm
                data_troll = current_y_troll

            #feature_of_interest_terra = data_terra[:, feature_id]
            #x_terra = np.linspace(current_strart_time_terra, current_stop_time_terra, len(feature_of_interest_terra))
            #feature_of_interest_terra_filtered = filter_high_frequencies(feature_of_interest_terra, 5, self.train_test_dataset.cutoff_frequency)
            #feature_of_interest_terra_smoothed = moving_average(feature_of_interest_terra_filtered, window_size = smoothind_window_size)
            #x_terra_smoothed = np.linspace(current_strart_time_terra, current_stop_time_terra, len(feature_of_interest_terra_smoothed))
            #ax_.plot(x_terra, feature_of_interest_terra, label = "original TerRA",c = "red")
            #ax_.plot(x_terra_smoothed, feature_of_interest_terra_smoothed, label = "TerRA w/o noise",c = "orange")

            feature_of_interest_troll = data_troll[:, feature_id]
            x_troll = np.linspace(current_strart_time_troll, current_stop_time_troll, len(np.array(feature_of_interest_troll).ravel()))
            #feature_of_interest_troll_filtered = filter_high_frequencies(feature_of_interest_troll, 5, self.train_test_dataset.cutoff_frequency)
            #feature_of_interest_troll_smoothed = moving_average(feature_of_interest_troll_filtered, window_size = smoothind_window_size*20)
            x_troll_smoothed = np.linspace(current_strart_time_troll, current_stop_time_troll, len(feature_of_interest_troll_smoothed))
            ax_.plot(x_troll, np.array(feature_of_interest_troll).ravel(), label = "original TROLL",  c = "green")
            ax_.plot(x_troll_smoothed, np.array(feature_of_interest_troll_smoothed).ravel(), label = "TROLL w/o noise",  c = "yellow")

            feature_of_interest_scm = data_scm[:, feature_id]
            x_scm = np.linspace(current_strart_time_scm, current_stop_time_scm, len(feature_of_interest_scm))
            feature_of_interest_scm_filtered = filter_high_frequencies(feature_of_interest_scm, 5, self.train_test_dataset.cutoff_frequency)
            feature_of_interest_scm_smoothed = moving_average(feature_of_interest_scm_filtered, window_size = smoothind_window_size)
            x_scm_smoothed = np.linspace(current_strart_time_scm, current_stop_time_scm, len(feature_of_interest_scm_smoothed))
            ax_.plot(x_scm, np.array(feature_of_interest_scm).ravel(), label = "original SCM",  c = "blue")
            ax_.plot(x_scm_smoothed, np.array(feature_of_interest_scm_smoothed).ravel(), label = "SCM w/o noise",  c = "purple")
            
            ax_.set_xlabel("time, [s]")
            ax_.set_ylabel("traction force, [N]")
            ax_.legend(prop={'size': 10})
            ax_.set_title("TROLL run with ID: " + str(run_id))
            #else:
            #    feature_of_interest_terra = current_y_terra[:, feature_id]
            #    x_terra = np.linspace(current_strart_time_terra, current_stop_time_terra, len(np.array(feature_of_interest_terra).ravel()))
            #    ax_.plot(x_terra, np.array(feature_of_interest_terra).ravel(), label = "TerRA",c = "red")
            #    feature_of_interest_scm = current_y_scm[:, feature_id]
            #    x_scm = np.linspace(current_strart_time_scm, current_stop_time_scm, len(np.array(feature_of_interest_scm).ravel()))
            #    ax_.plot(x_scm, np.array(feature_of_interest_scm).ravel(), label = "SCM",  c = "blue")
            #    feature_of_interest_troll = current_y_troll[:, feature_id]
            #    x_troll = np.linspace(current_strart_time_troll, current_stop_time_troll, len(np.array(feature_of_interest_troll).ravel()))
            #    ax_.plot(x_troll, np.array(feature_of_interest_troll).ravel(), label = "TROLL",  c = "green")
            #    ax_.plot(x_troll, np.array(feature_of_interest_troll_smoothed).ravel(), label = "TROLL w/o noise",  c = "yellow")
            #    ax_.set_xlabel("time, [s]")
            #    ax_.set_ylabel("feature id: " + str(feature_id) + " , []")
            #    ax_.legend()
            #    ax_.set_title("TROLL run with ID: " + str(run_id))
        plt.show()

    def plot_normal_against_sinkage_in_train(self):
        plt.figure(figsize=(10, 15))
        #x_troll = self.train_test_dataset_from_troll.dataset['scm'].train_test_dataset['train']['X'][:, 2]
        y_troll = self.train_test_dataset.dataset['troll'].train_test_dataset['train']['y'][:, 2]
        x_scm = self.train_test_dataset.dataset['scm'].train_test_dataset['train']['X'][:, 2]
        y_scm = self.train_test_dataset.dataset['scm'].train_test_dataset['train']['y'][:, 2]
        plt.scatter(x_scm, y_troll, label = "TROLL", c = "green") # we use x from scm only. from hi-fi troll only outputs
        plt.scatter(x_scm, y_scm, label = "SCM", c = "blue")
        plt.legend()

    def validate_runs(self, run_ids_list):
        self.plot_against_time()
        self.plot_against_time()
        self.plot_against_time()

class SynthValidator:
    def __init__(self, num_of_exp, modelica_data_from_troll_location = "D:/Vlad/data/all_runs_synth.h5", smoothind_window_size = 20, points_per_second = 5, cutoff_frequency = 5, terra_exist = False, activate_surface = False):
        data = h5py.File(modelica_data_from_troll_location, 'r')
        self.run_length = data['0']['scm']['input'].shape[0] ## to specific, could be a bug here
        self.smoothind_window_size = smoothind_window_size
        self.points_per_second = points_per_second
        self.cutoff_frequency = cutoff_frequency
        scm_X = []
        scm_surface = []
        scm_y = []
        terra_X = []
        terra_surface = []
        terra_y = []
        
        for i in tqdm(range(num_of_exp), desc = "reading runs..."):
            scm_run_length = np.min([int(len(data[str(i)]["scm"]["output"])), int(len(data[str(i)]["scm"]["output"]))])
            scm_X.append(np.array([data[str(i)]["scm"]["input"][x][1][:12] for x in range(scm_run_length)]))
            if activate_surface:
                scm_surface.append(np.array([data[str(i)]["scm"]["input"][x][1][12:] for x in range(scm_run_length)]))
            scm_y.append(np.array([data[str(i)]["scm"]["output"][x][1][:6] for x in range(scm_run_length)]))
            if terra_exist:
                    terra_run_length = np.min([int(len(data[str(i)]["terra"]["output"])), int(len(data[str(i)]["terra"]["output"]))])
                    terra_X.append(np.array([data[str(i)]["terra"]["input"][x][1][:12] for x in range(terra_run_length)]))
                    if activate_surface:
                        terra_surface.append(np.array([data[str(i)]["terra"]["input"][x][1][12:] for x in range(terra_run_length)]))
                    terra_y.append(np.array([data[str(i)]["terra"]["output"][x][1][:6] for x in range(terra_run_length)]))
            
        print("concatenating runs...")
        self.scm_X = np.concatenate([self.smooth_signals(run) for run in scm_X], axis=0)
        if activate_surface:  
            self.scm_surface = np.concatenate([self.smooth_signals(run) for run in scm_surface], axis=0)
        else:
            self.scm_surface = None
        self.scm_y = np.concatenate([self.smooth_signals(run) for run in scm_y], axis=0)

        if terra_exist:
            self.terra_X = np.concatenate([self.smooth_signals(run) for run in terra_X], axis=0)
            if activate_surface:
                self.terra_surface = np.concatenate([self.smooth_signals(run) for run in terra_surface], axis=0)
            else:
                self.terra_surface = None
            self.terra_y = np.concatenate([self.smooth_signals(run) for run in terra_y], axis=0)
        else:
            self.terra_X, self.terra_surface, self.terra_y = None, None, None
        
        self.dataset = {"terra":{"X": self.terra_X, "surface": self.terra_surface, "y": self.terra_y},
                        "scm":{"X": self.scm_X, "surface": self.scm_surface, "y": self.scm_y}}

        # variables that will be used for plotting
        self.force_x_scm = self.scm_y[:, 0]
        self.force_z_scm = self.scm_y[:, 2]
        if terra_exist:
            self.force_x_terra = self.terra_y[:, 0]
            self.force_z_terra = self.terra_y[:, 2]

    def smooth_signals(self, dataset):
        one_run_all_signals = []
        for i in range(dataset.shape[1]):
            filtered = filter_high_frequencies(dataset[:, i], self.points_per_second, self.cutoff_frequency)
            smoothed = moving_average(np.array(filtered), self.smoothind_window_size)
            one_run_all_signals.append(smoothed)
        one_run_all_signals = np.stack(one_run_all_signals, axis = 1)
        return one_run_all_signals
    
#################################### TODO: rewrite to one function ##################################
    def sinkage_against_time(self):
        plt.figure(figsize=(12,7))
        for i in range(10):
            plt.plot(range(self.run_length), self.scm_X[(i*self.run_length):((i+1)*self.run_length), 2], c = 'blue', alpha=0.2)
            plt.plot(range(self.run_length), self.terra_X[(i*self.run_length):((i+1)*self.run_length), 2], c = 'red', alpha=0.2)
        plt.xlabel("timestamp, [s]")
        plt.ylabel("sinkage, [m]")
        plt.legend(["SCM", "TerRA"])

    def traction_against_time(self):
        plt.figure(figsize=(15,10))
        for i in range(10):
            plt.plot(range(self.run_length), self.force_x_scm[(i*self.run_length):((i+1)*self.run_length)], c = 'blue', alpha=0.2)
            plt.plot(range(self.run_length), self.force_x_terra[(i*self.run_length):((i+1)*self.run_length)], c = 'red', alpha=0.2)
        plt.xlabel("timestamp, [s]")
        plt.ylabel("traction force, [N]")
        plt.legend(["SCM", "TerRA"])

    def normal_against_time(self):        
        plt.figure(figsize=(15,10))
        for i in range(10):
            plt.plot(range(self.run_length), self.force_x_scm[(i*self.run_length):((i+1)*self.run_length)], c = 'blue', alpha=0.2)
            plt.plot(range(self.run_length), self.force_z_terra[(i*self.run_length):((i+1)*self.run_length)], c = 'red', alpha=0.2)
        plt.xlabel("timestamp, [s]")
        plt.ylabel("normal force, [N]")
        plt.ylim(-5, 100)
        plt.legend(["SCM", "TerRA"])
#########################################################################################################

    def plot_normal_against_sinkage_in_train(self):
        plt.figure(figsize=(10, 10))
        plt.scatter(self.scm_X[:, 2], self.force_z_scm, label = "SCM data for surrogate creation", c = "green") # we use x from scm only. from hi-fi troll only outputs
        plt.xlabel("p2, sinkage, [m]")
        plt.ylabel("normal force, [N]")
        plt.legend()

    def export_surface():
        # only for synthetic lower-fidelity data
        '''
        lf_surface = synthval.dataset["scm"]['surface']
        lf_y = synthval.dataset["scm"]['y']
        pd.DataFrame(np.concatenate([lf_surface, lf_y], axis=1)).to_csv("LF_surface_scans_200runs_drop_diff_mass.csv", header=False, index=False)
        '''