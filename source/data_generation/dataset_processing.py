from source.utils.validation import SynthValidator
import numpy as np
import matplotlib.pyplot as plt
from source import dataset_from_troll

class normalizer:
    def __init__(self, X: np.ndarray):
        self.max = np.max(X, axis=0)
        self.min = np.min(X, axis=0)
        self.selected_columns = self.max - self.min != 0

    def normalize(self, X):
        X_scaled = ((X-self.min)/(self.max-self.min))[:, self.selected_columns].copy()
        return X_scaled
    
    def save_weights(self, filename):
        with open("D:/Vlad/mufintroll/parameters/" + filename, 'w') as file:
            file.write(' '.join(map(str, np.arange(len(self.selected_columns))[self.selected_columns])) + '\n')
            file.write(' '.join(map(str, self.max[self.selected_columns])) + '\n')
            file.write(' '.join(map(str, self.min[self.selected_columns])) + '\n')


class Processed_dataset:
    '''
    class represents a down-sampled, normalized dataset without non-informative columns; should be used for fitting the MF models
    '''
    def __init__(self, dataset_from_troll: dataset_from_troll.DatasetTroll, sythetic_datasets: list, method = 'scm'):
        list_of_unnorm_X = []
        list_of_y = []
        for sythetic_dataset in sythetic_datasets:
            synthetic_data = sythetic_dataset.dataset.copy()
            # lower fidelity
            list_of_unnorm_X.append(synthetic_data[method]['X'].copy())
            list_of_y.append(synthetic_data[method]['y'].copy())
        
        self.X_lf_unnormalized = np.concatenate(list_of_unnorm_X, axis = 0)
        y_lf = np.concatenate(list_of_y, axis = 0)
        ## delete outliers
        #no_outlier_mask = (self.X_lf_unnormalized[:, 2] < 0.13) & (y_lf[:, 0] > -40) & (y_lf[:, 2] < 100) & (self.X_lf_unnormalized[:, 0] > -0.1)
        #no_outlier_mask = (self.X_lf_unnormalized[:, 2] < 0.13) # & (self.X_lf_unnormalized[:, 0] > -0.1)
        self.X_lf_unnormalized = self.X_lf_unnormalized#[no_outlier_mask, :].copy()
        self.y_lf = y_lf#[no_outlier_mask, :]

        # use X input from the modelica simulation representing the lower fidelity level, because whose are inputs which will be used for running the simulation; X input from TROLL and SCM\TerRA might be non-aligned or not perfectly alligned, so using TROLL input might lead to mistakes
        # TODO: resolve alignment issue
        # higher fidelity
        self.X_hf = dataset_from_troll.dataset[method].train_test_dataset['train']['X'].copy()
        self.y_hf = dataset_from_troll.dataset['troll'].train_test_dataset['train']['y'].copy()
        self.X_hf_test = dataset_from_troll.dataset[method].train_test_dataset['test']['X'].copy()
        self.y_hf_test = dataset_from_troll.dataset['troll'].train_test_dataset['test']['y'].copy()
        self.hf_norm = normalizer(self.X_hf)
        self.lf_norm = normalizer(self.X_lf_unnormalized)
        #### keep number of columns the same in both LF and HF datasets...
        self.hf_norm.selected_columns *= self.lf_norm.selected_columns
        self.lf_norm.selected_columns = self.hf_norm.selected_columns


    def create_dataset(self, lf_n_points: int, hf_n_points: int, compare_distr: bool):
        #print("selected columns: ", self.lf_norm.selected_columns)
        #X_lf_normalized = self.lf_norm.normalize(self.X_lf_unnormalized)
        self.X_lf_train, self.y_lf_train = self.downsample(self.lf_norm.normalize(self.X_lf_unnormalized.copy()), self.y_lf, lf_n_points)
        self.X_hf_train, self.y_hf_train = self.downsample(self.hf_norm.normalize(self.X_hf.copy()), self.y_hf, hf_n_points)

        if compare_distr:
            interesting_dims = [0, 2, 7]
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
            X_unnorm_down_lf, y_unnorm_down_lf = self.downsample(self.X_lf_unnormalized.copy(), self.y_lf, lf_n_points)
            for i in range(3):
                ax[i].hist(self.X_lf_unnormalized[:, interesting_dims[i]], alpha=0.5, density = True, label = "original")
                ax[i].hist(X_unnorm_down_lf[:, interesting_dims[i]], alpha=0.5, density = True, label = "downsampled")
                ax[i].set_title("Input feature: " + str(interesting_dims[i]))
                ax[i].legend(prop={'size': 10})
            plt.show()
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            for i in range(2):
                ax[i].hist(self.y_lf[:, interesting_dims[i]], alpha=0.5, density = True, label = "original")
                ax[i].hist(y_unnorm_down_lf[:, interesting_dims[i]], alpha=0.5, density = True, label = "downsampled")
                ax[i].set_title("Output feature: " + str(interesting_dims[i]))
                ax[i].legend(prop={'size': 10})
            plt.show()

        self.X_test = [self.hf_norm.normalize(test_run) for test_run in self.X_hf_test]
        self.y_test = self.y_hf_test

    #def delete_outliers(self, M):
        

    def downsample(self, X, y, n_points):
        rand_indx = np.random.choice(a = X.shape[0], size = n_points, replace=False)
        X_downsampled = X[rand_indx]
        y_downsampled = y[rand_indx]
        return X_downsampled, y_downsampled
    
    def save_weights(self, filename):
        with open("D:/Vlad/mufintroll/parameters/" + filename, 'w') as file:
            file.write(' '.join(map(str, np.arange(len(self.hf_norm.selected_columns))[self.hf_norm.selected_columns])) + '\n')
            for norm in [self.lf_norm, self.hf_norm]:
                file.write(' '.join(map(str, norm.max[self.hf_norm.selected_columns])) + '\n')
                file.write(' '.join(map(str, norm.min[self.hf_norm.selected_columns])) + '\n')

    def check_distribution(self, origina_data, downsampled):
        # check if the distribution of features holds in both undersampled and original distributions
        for i in range(6):
            plt.figure(figsize=(10,5))
            plt.hist(origina_data[:, i], bins=40, density=True, label = "original", range = (-20, 20))
            plt.hist(downsampled[:, i], bins=40, density=True, alpha = 0.6, label = "downsampled", range = (-20, 20))
            plt.legend()
            #plt.xlabel("traction force")
            #plt.xlim((-20, 20))
            plt.show()