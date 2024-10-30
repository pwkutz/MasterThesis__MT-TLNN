# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import glorot_uniform
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import KFold
from sourceMA.data_loader import resource_path
from tensorflow.keras.models import load_model
from sourceMA import data_loader
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sourceMA.data_loader import de_normalize


class AE:

    def __init__(self, config):

        self.model = None
        self.AENet(input_shape=(None, config["input_shape"]["AE"][0]),
                         classes=1)
        self.hist = None
        self.predictions = None
        self.training = False

    def AENet(self, input_shape=(None, 1), classes=1):

        # Define the input as a tensor with shape input_shape
        X_input = layers.Input(input_shape)

        X = layers.Dense(8, activation="relu", name="decoder")(X_input)
        X = layers.Dense(1, activation="sigmoid", name="encoder")(X)

        self.model = Model(inputs=X_input, outputs=X, name='TransferNet')

    def plot_stats__nn(self):

        '''plot structure and dimensions of NN'''

        self.model.summary()
        tf.keras.utils.plot_model(self.model, show_shapes=True)

    def compile(self):

        self.model.compile(loss='mse',
                           optimizer='Adam',
                           metrics=['mse'])

    def launch_AE(self, features, target, config):

        # plot_stats__nn()
        self.training = True
        self.compile()

        cp_callback = keras.callbacks.EarlyStopping(monitor='val_mse',  # Early Stopping
                                                    patience=config["patience"])

        self.hist = self.model.fit(features,  # Features
                                   target,  # force x component
                                   epochs=config["epoch"],  # Number of epochs
                                   verbose=1,
                                   batch_size=config["batch_size"],
                                   validation_split=config["val_split"],
                                   callbacks=[cp_callback])

    def launch_TransferNet__PerformanceCheck(self, config, train_features, train_target):

        self.compile()

        cp_callback = keras.callbacks.EarlyStopping(monitor='val_mse',  # Early Stopping
                                                    patience=config["patience"])

        # batch_size = config["batch_size"] = train_features.shape[0]

        self.hist = self.model.fit(train_features,  # Features
                                   train_target,  # force x component
                                   epochs=config["epoch"],  # Number of epochs
                                   verbose=1,
                                   batch_size=config["batch_size"],
                                   validation_split=config["val_split"],
                                   callbacks=[cp_callback])

    def freeze_BaseModel(self, config):

        '''
        freeze all layers except the last two layers.
        All other layers except last 2 layers: training with LF data
        Last two layers: training with HF data
        '''

        for layer in self.model.layers[0:-config["keep_Nth_last_layer"]]: layer.trainable = False
        for layer in self.model.layers[-config["keep_Nth_last_layer"]:]: layer.trainable = True

    def load_model(self, config, type):

        ''' load trained parameters out of .keras file '''

        self.model = load_model(data_loader.resource_path(config["checkpoint_path"][type]))

    def predicts(self, test_features):

        ''' propagate test set through trained NN + obtains MDACNN values for input samples'''

        predictions = self.model.predict(test_features)  # prediction
        self.predictions = [k for i in predictions.tolist() for k in i]

    def plot(self, features, target):

        '''plot either the results of propagating the test or train dataset'''

        X, X_min, X_max, Y_min, Y_max = DefineParams(features, target)

        self.PlotPrediction(target, X)

        if self.hist is not None: self.PlotTrainingProgress()

        test_loss, test_accuracy = self.evaluates(features, target)

        print(f" Accuracy (model.evaluate): {test_accuracy}")

    def analyse(self, train_features, train_target, test_features, test_target):

        '''
        define data intervals (biggest, smallest sample value and biggest and smallest output value)
        + plot LF, HF and MDACNN output results
        '''

        self.plot(test_features, test_target)
        if self.training:
            self.predicts(train_features)
            self.plot(train_features, train_target)

    def PlotPrediction(self, test_target, X):

        '''
        plot the predictions made by the MDACNN
        The predictions are the output of the MDACNN for input features (which get forwarded through NN).
        '''

        plt.style.use("seaborn-v0_8")

        plt.plot(X, test_target, color='b', marker='.', label='Ground Truth')  # HF features + LF evaluation
        plt.plot(X, self.predictions, color='r', marker='.', label='CNN')

        plt.title("CNN using Transfer Learning")
        plt.xlabel("Samples")
        plt.ylabel("Value")
        plt.legend(["Ground Truth", "CNN"])

        plt.grid(True)
        # plt.xlim(X_min, X_max)  # show only in range of all X values
        # plt.ylim(Y_min, Y_max)  # show only range of all Y values

        y_hat = test_target
        y_pred = self.predictions
        accuracy = mean_squared_error(y_hat, y_pred)

        # textstr = 'Accuracy=%.2f' % accuracy
        print("... Accuracy (MSE):", accuracy)
        # plt.text(X_min - (np.sqrt(X_min ** 2 + X_max ** 2) / 8), Y_min + (np.sqrt(Y_min ** 2 + Y_max ** 2) / 2), textstr, fontsize=14)
        plt.subplots_adjust(left=0.25)
        plt.show()

    def plot__accuracy(self):

        '''
        Accuracy documents overfitting.
        Overfitting ... no generalization ... learning training samples instead of pattern
        '''

        plt.plot(self.hist.history['mse'], label="train")
        plt.plot(self.hist.history['val_mse'], label="val")
        plt.title("Accuracy")
        plt.legend()
        plt.show()

    def plot__loss(self):

        '''
        Loss documents underfitting.
        Underfitting ... not learning from training dataset. No learning samples and pattern.
        '''

        plt.plot(self.hist.history['loss'], label="train")
        plt.plot(self.hist.history['val_loss'], label="val")
        plt.title("Loss")
        plt.legend()
        plt.show()

    def PlotTrainingProgress(self):

        ''' plot the history of loss and accuracy during the training epochs'''

        self.plot__accuracy()
        self.plot__loss()

    def evaluates(self, test_features, test_target):

        '''propagates test set through trained NN + obtains test loss and test accuracy'''

        self.model.compile(loss='mse',
                           optimizer='adam',
                           metrics=['mse'])

        test_loss, test_accuracy = self.model.evaluate(test_features, test_target)
        return test_loss, test_accuracy


def DefineParams(test_features, test_target):
    '''
    predefine all parameters needed to plot the ...
    prediction (MDACNN outputs)
    + the learning process (course of the error during the training process)
    '''

    # test.feature |--> datatable with 26 columns of order [LF,LF(LF), HF, LF(HF)]
    # test.target |--> ground truth value for yHF. HF-Function results. Result which should be achieved by MDACNN
    # test.feat_HF__target_LF |--> LF(HF) aka. most right column in data frame
    # self.predictions |--> Y_P for test.feature by MDACNN
    # all samples are sorted

    Y_min = min(test_target)
    Y_max = max(test_target)
    X_max = max(list(range(test_features.shape[0])))  # Batch-Dim of Tensor |--> biggest Batch index
    X_min = min(list(range(test_features.shape[0])))
    X = list(range(test_features.shape[0]))  # iterative list over all test samples

    return X, X_min, X_max, Y_min, Y_max


def TransferLearning(model, config,
                     train_features__HF, train_target__HF,
                     train_features__LF, train_target__LF):
    '''
    Transfer Learning:
    1) train on LF data
    2) re-train on HF data
    '''

    model.launch_TransferNet(train_features__LF,
                             train_target__LF,
                             config)

    model.freeze_BaseModel(config)

    model.launch_TransferNet(train_features__HF,
                             train_target__HF,
                             config)


def TransferLearning__PerformanceCheck(CNN_model, config,
                                       train_features, train_target,
                                       test_features, test_target):
    CNN_model.launch_TransferNet__PerformanceCheck(config, transform_to_scans(train_features), train_target)

    CNN_model.freeze_BaseModel(config)

    CNN_model.launch_TransferNet__PerformanceCheck(config, transform_to_scans(test_features), test_target)


def transform_to_scans(x_data):
    surface_scans = []
    for i in range(x_data.shape[0]):
        surface_scans.append(np.reshape(np.array(x_data.iloc[i, :]), (64, 64, 1)))
    surface_scans = np.array(surface_scans)

    return surface_scans


def safety_check(config):
    if (config["mode"]["TestAndTrain"] is True) & (config["mode"]["PerformanceCheck"] is True):
        print("Mode Error: 'TestAndRun' and 'PerformanceCheck' are on")
        sys.exit()
    elif (config["mode"]["TestAndTrain"] is not True) & (config["mode"]["PerformanceCheck"] is not True):
        print("Mode Error: 'TestAndRun' and 'PerformanceCheck' are off")
        sys.exit()


def select__feat_target(split, features, target):
    '''
    splits divide dataset into Train and Test Datasets (per fold)
    retrieve for each fold Train and Test Dataset
    Validation Dataset gets taken from Train Dataset (10 %)
    '''

    features = features.iloc[split].reset_index(drop=True)
    target = target[split]

    return features, target


def cross_validation(config, features__HF, target__HF, features__LF, target__LF):
    # Merge inputs and targets
    # inputs = np.concatenate((input_train, input_test), axis=0)
    # targets = np.concatenate((target_train, target_test), axis=0)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=config["num_folds"], shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    for kfold, (train__split, test__split) in enumerate(kfold.split(features__HF, target__HF)):
        # Define the model architecture
        CNN_model = CNN(config)  # define MDACNN

        train_features, train_target = select__feat_target(train__split, features__HF, target__HF)
        test_features, test_target = select__feat_target(test__split, features__HF, target__HF)

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        TransferLearning__PerformanceCheck(CNN_model, config,
                                           train_features, train_target,
                                           test_features, test_target)

        scores = CNN_model.model.evaluate(transform_to_scans(test_features), test_target, verbose=0)
        print(f'Score for fold {fold_no}: '
              f'{CNN_model.model.metrics_names[0]} of {scores[0]}; {CNN_model.model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1])  # * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1

        CNN_model.model.save(resource_path(f'Weights__NN/wg_{kfold}.keras'))

    idx_max = acc_per_fold.index(max(acc_per_fold))
    file = f'Weights__NN/wg_{idx_max}.keras'

    return np.mean(acc_per_fold), np.mean(loss_per_fold), file

