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


def transform_to_scans(x_data):
    surface_scans = []
    for i in range(x_data.shape[0]):
        surface_scans.append(np.reshape(np.array(x_data.iloc[i, :]), (64,64,1)))
    surface_scans = np.array(surface_scans)
    
    return surface_scans

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

def ResNet50(input_shape = (64, 64, 1), classes = 6):
    # Define the input as a tensor with shape input_shape
    X_input = layers.Input(input_shape)
    # Zero-Padding
    X = layers.ZeroPadding2D((3, 3))(X_input)
    # Stage 1
    X = layers.Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    # AVGPOOL
    X = layers.AveragePooling2D(pool_size=(2,2), padding='same')(X)
    # Output layer
    X = layers.Flatten()(X)
    X = layers.Dense(32)(X)
    X = layers.Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

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

def launch_resnet(train_features, train_target, test_features, test_target, number_of_input_channels):
    model_res = ResNet50(input_shape = (64, 64, number_of_input_channels), classes = 1)
    model_res.compile(loss='mse', 
                      optimizer='Adagrad', 
                      metrics=['mae'])

    history_res = model_res.fit(train_features, # Features
                          train_target, # force x component
                          epochs=8, # Number of epochs
                          verbose=1,
                          batch_size=32,
                          validation_data=(test_features, test_target))
    return history_res

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


# example of usage
'''
from source.auxiliary import nn
import pandas as pd

train = pd.read_csv("HF_train_surface_scans.csv", header=None) 

train_features = train.iloc[:, :4096]
train_outputs = train.iloc[:, 4096:]
train_target = train_outputs.iloc[:, 0]

test = pd.read_csv("HF_test_surface_scans.csv", header=None)

run_4350_indx = test_outputs.iloc[:, -1] == 4350
test_features = test.iloc[:, :4096][run_4350_indx]
test_outputs = test.iloc[:, 4096:][run_4350_indx]
test_target = test_outputs.iloc[:, 0]

nn.launch_resnet(nn.transform_to_scans(train_features), train_target, nn.transform_to_scans(test_features), test_target, number_of_input_channels = 1)
'''


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
