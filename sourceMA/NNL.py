# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import glorot_uniform
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import KFold
from sourceMA.data_loader import resource_path
from tensorflow.keras.models import load_model
from sourceMA import data_loader
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sourceMA.data_loader import de_normalize

class CNN:

    def __init__(self, config):

        self.model = None
        self.TransferNet(input_shape=(config["input_shape"]["CNN"][0],
                                      config["input_shape"]["CNN"][1],
                                      config["input_shape"]["CNN"][2]),
                         classes=1)
        self.hist = None
        self.predictions = None
        self.training = False

    def TransferNet(self, input_shape=(64, 64, 1), classes=1):

        # Define the input as a tensor with shape input_shape
        X_input = layers.Input(input_shape)


        X = layers.Conv2D(8, (3, 3), strides=(1, 1), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
        X = layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid", name="AveragePool1")(X)
        X = layers.Conv2D(24, (5, 5), strides=(2, 2), name='conv2')(X)
        X = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", name="AveragePool2")(X)

        X = layers.Flatten()(X)

        X = layers.Dense(4, name="FC1")(X)
        X = layers.Dense(1024, activation="leaky_relu", name="FC2")(X)
        # last 2 layers: train with HF data during transfer learning
        X = layers.Dense(1024, activation="leaky_relu", name="FC3")(X)

        X = layers.Dense(1, name="FC4")(X)  # output layer

        self.model = Model(inputs=X_input, outputs=X, name='CNN')

    def plot_stats__nn(self):

        '''plot structure and dimensions of NN'''

        self.model.summary()
        tf.keras.utils.plot_model(self.model, show_shapes=True)

    def compile(self):

        self.model.compile(loss='mse',
                      optimizer='Adagrad',
                      metrics=['mae'])

    def launch_TransferNet(self, train_features, train_target, config):


        # plot_stats__nn()
        self.training = True
        self.compile()

        cp_callback = keras.callbacks.EarlyStopping(monitor='val_mae',  # Early Stopping
                                                    patience=config["patience"])

        self.hist= self.model.fit(train_features,  # Features
                                train_target,  # force x component
                                epochs=config["epoch"],  # Number of epochs
                                verbose=1,
                                batch_size=config["batch_size"],
                                validation_split=config["val_split"],
                                callbacks=[cp_callback])



    def launch_TransferNet__PerformanceCheck(self, config, train_features, train_target):

            self.compile()

            cp_callback = keras.callbacks.EarlyStopping(monitor='val_mae',  # Early Stopping
                                                        patience=config["patience"])

            #batch_size = config["batch_size"] = train_features.shape[0]

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

        X, X_min, X_max, Y_min, Y_max= DefineParams(features, target)

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


        plt.plot(X, test_target, color='b', marker='.', label='Ground Truth') # HF features + LF evaluation
        plt.plot(X, self.predictions, color='r', marker='.', label='CNN')

        plt.title("CNN using Transfer Learning")
        plt.xlabel("Samples")
        plt.ylabel("Value")
        plt.legend(["Ground Truth", "CNN"])

        plt.grid(True)
        #plt.xlim(X_min, X_max)  # show only in range of all X values
        #plt.ylim(Y_min, Y_max)  # show only range of all Y values

        y_hat = test_target
        y_pred = self.predictions
        accuracy = mean_squared_error(y_hat, y_pred)

        #textstr = 'Accuracy=%.2f' % accuracy
        print("... Accuracy (MSE):", accuracy)
        # plt.text(X_min - (np.sqrt(X_min ** 2 + X_max ** 2) / 8), Y_min + (np.sqrt(Y_min ** 2 + Y_max ** 2) / 2), textstr, fontsize=14)
        plt.subplots_adjust(left=0.25)
        plt.show()


    def plot__accuracy(self):

        '''
        Accuracy documents overfitting.
        Overfitting ... no generalization ... learning training samples instead of pattern
        '''

        plt.plot(self.hist.history['mae'], label="train")
        plt.plot(self.hist.history['val_mae'], label="val")
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
                          metrics=['mae'])

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
    X = list(range(test_features.shape[0])) # iterative list over all test samples

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
        surface_scans.append(np.reshape(np.array(x_data.iloc[i, :]), (64,64,1)))
    surface_scans = np.array(surface_scans)

    return surface_scans


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
    #inputs = np.concatenate((input_train, input_test), axis=0)
    #targets = np.concatenate((target_train, target_test), axis=0)

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
        acc_per_fold.append(scores[1])# * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1

        CNN_model.model.save(resource_path(f'Weights__NN/wg_{kfold}.keras'))

    idx_max = acc_per_fold.index(max(acc_per_fold))
    file = f'Weights__NN/wg_{idx_max}.keras'

    return np.mean(acc_per_fold), np.mean(loss_per_fold), file





##################################################################################
##################################################################################


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # retrieve filters
    F1, F2, F3 = filters
    # Save input values 
    X_shortcut = X
    # First component of main path
    X = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid", name=conv_name_base+'2a', 
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)
    #Second component of main path
    X = layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same", name=conv_name_base+'2b', 
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)
    # Third component of main path
    X = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid", name=conv_name_base+'2c', 
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    # add shortcut
    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    # Save input values 
    X_shortcut = X
    # First component of main path
    X = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding="valid", name=conv_name_base+'2a', 
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)
    #Second component of main path
    X = layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same", name=conv_name_base+'2b', 
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)
    # Third component of main path
    X = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid", name=conv_name_base+'2c', 
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    ### SHORTCUT
    X_shortcut = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid", name=conv_name_base+'1', 
               kernel_initializer=glorot_uniform(seed=0))(X)
    X_shortcut = layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
    # final step
    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)
    
    return X

def create_relative_velocity_scans(df):
    surface_scans = []
    for i in range(df.shape[0]):
        surface_scans.append(np.reshape(np.array(df.iloc[i, 18:]), (64,64,1)))
    surface_scans = np.array(surface_scans)

    ## create relative velocities map under the wheel
    step = 0.005
    length = surface_scans.shape[0]
    v_rel = np.zeros((length,64,64,3))

    x = abs(np.linspace(np.full((1,64),-0.16)[0], np.full((1,64),0.16)[0], 64))
    y = x.transpose()

    for num_of_scan in range(length):
        z = surface_scans[num_of_scan].reshape((64, 64)) # choose height map for each 

        r = np.stack([x,y,z])
        w = np.array(df.iloc[num_of_scan, 12:15])
        v = np.array(df.iloc[num_of_scan, 9:12])
        tmp = v.reshape((1,3)) + np.cross(w, r.reshape(3,-1).transpose())

        v_rel[num_of_scan] = tmp.reshape((64, 64, 3))
    return v_rel



def call_test_run():
    sys.path.append("D:/Vlad/Dymola_2020/Modelica/Library/python_interface/dymola.egg")
    from dymola.dymola_interface import DymolaInterface
    from dymola.dymola_exception import DymolaException

    path = "D:/Vlad/Dymola_2020/bin64/Dymola.exe"
    lib = ["D:/Vlad/library/ContactDynamics/package.mo", "D:/Vlad/work/scripts/run_bekker_model.mo", "D:/Vlad/work/terra.mo", 
           "D:/Vlad/library/SR_Utilities/package.mo", "D:/Vlad/library/Visualization2/Modelica/Visualization2/package.mo",
           "D:/Vlad/library/Visualization2Overlays/package.mo", "D:/Vlad/library/MyPackage/package.mo"]
    work = "D:/Vlad/work/synthetic_surfaces/"

    def init_dymola(path,work,libs):
        dymola = DymolaInterface(path)
        for lib in libs:
            dymola.openModel(lib)
        dymola.cd(work)    
        return dymola

    dymola = init_dymola(path,work,lib)
    ## read the values from the created file "terra_output" and create the dataset
    size = dymola.readTrajectorySize("D:/Vlad/work/TerraWithoutVis.mat")
    names = dymola.readTrajectoryNames("D:/Vlad/work/TerraWithoutVis.mat")
    ## define relevant features
    features = []

    #features.append("wheel.mLTMFrame.output_values[1]")
    for i in range(6):
        #ftr = "wheel.mLTMFrame.input_values[" + str(i+1) + "]"
        ftr = "wheel.datagathering.output_values[" + str(i+1) + "]"
        features.append(ftr)

    for i in range(2108):
        #ftr = "wheel.mLTMFrame.input_values[" + str(i+1) + "]"
        ftr = "wheel.datagathering.input_values[" + str(i+1) + "]"
        features.append(ftr)

    dataset = dymola.readTrajectory("D:/Vlad/work/TerraWithoutVis.mat", features, size)
    df_1 = pd.DataFrame(dataset).transpose()
    ## define relevant features
    features = []

    for i in range(2108, 4108):
        #ftr = "wheel.mLTMFrame.input_values[" + str(i+1) + "]"
        ftr = "wheel.datagathering.input_values[" + str(i+1) + "]"
        features.append(ftr)

    dataset = dymola.readTrajectory("D:/Vlad/work/TerraWithoutVis.mat", features, size)
    df_2 = pd.DataFrame(dataset).transpose()
    df_final = pd.concat([df_1, df_2], axis = 1)
    
    return(df_final)

def generate_arrays_from_file(path, batchsize = 64):
    inputs = []
    targets = []
    batchcount = 0
    step = 0.005

    x_image = np.linspace(np.full((1,64),-0.16)[0], np.full((1,64),0.16)[0], 64)
    y_image = x_image.transpose()
    
    while True:
        with open(path) as f:
            for line in f:
                l = line.split(',')
                l = [float(i) for i in l]
                #create velocity maps
                z = np.reshape(np.array(l[18:]), (64, 64)) # choose height map for each 
                r = np.stack([x_image,y_image,z])
                
                w = np.array(l[12:15])
                v = np.array(l[9:12])
                tmp = v.reshape((1,3)) + np.cross(w, r.reshape(3,-1).transpose())
                
                x = tmp.reshape((64, 64, 3))
                
                y = l[0] # focre x component
                inputs.append(x)
                targets.append(y)
                batchcount += 1
                if batchcount > batchsize:
                    X = np.array(inputs)
                    y = np.array(targets, dtype='float32')
                    yield (X, y)
                    inputs = []
                    targets = []
                    batchcount = 0


# +
# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model

# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

class BNN():
    def __init__(self, train_size, hidden_units, input_size):
        self.train_size = train_size
        self.hidden_units = hidden_units
        self.input_size = input_size


    # -
    def create_bnn_model(self):
        inputs = layers.Input((self.input_size,))
        features = layers.BatchNormalization()(inputs)

        # Create hidden layers with weight uncertainty using the DenseVariational layer.
        for units in self.hidden_units:
            features = tfp.layers.DenseVariational(
                units=units,
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=1/self.train_size,
                activation="sigmoid",
            )(features)

        # The output is deterministic: a single point estimate.
        outputs = layers.Dense(units=1)(features)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        #return model

    def predict(self, test_x, iterations=100):
        predicted = []
        for _ in range(iterations):
            predicted.append(self.model(np.array(test_x)))
        predicted = np.concatenate(predicted, axis=1)

        prediction_mean = np.mean(predicted, axis=1)
        #prediction_min = np.min(predicted, axis=1).tolist()
        #prediction_max = np.max(predicted, axis=1).tolist()
        #prediction_range = (np.max(predicted, axis=1) - np.min(predicted, axis=1)).tolist()
        prediction_var = np.var(predicted, axis=1)
        return prediction_mean, prediction_var

    def create_probablistic_bnn_model(self, input_dim):
        inputs = layers.Input((input_dim,))
        features = layers.BatchNormalization()(inputs)

        # Create hidden layers with weight uncertainty using the DenseVariational layer.
        for units in self.hidden_units:
            features = tfp.layers.DenseVariational(
                units=units,
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=1 / self.train_size,
                activation="sigmoid",
            )(features)

        # Create a probabilisticå output (Normal distribution), and use the `Dense` layer
        # to produce the parameters of the distribution.
        # We set units=2 to learn both the mean and the variance of the Normal distribution.
        distribution_params = layers.Dense(units=2)(features)
        outputs = tfp.layers.IndependentNormal(1)(distribution_params)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        #return model

    def negative_loglikelihood(self, targets, estimated_distribution):
        return -estimated_distribution.log_prob(targets)

    def compile(self, optimizer_, loss_, metrics_):
        self.model.compile(        
            optimizer=optimizer_,
            loss= loss_,
            metrics=metrics_)

    def fit(self, X, y, num_epochs, X_test, y_test):
        self.model.fit(X, y, epochs=num_epochs, 
                    batch_size = 64, validation_data=(X_test, y_test), verbose = 0)

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = np.array(self.y[idx * self.batch_size:(idx + 1) * self.batch_size])
        return batch_x, batch_y
