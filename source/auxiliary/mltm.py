import numpy as np
import pandas as pd
from math import pi, tan, sin, cos, exp, acos, asin, atan2, sqrt, tanh
from numpy import sign
#from sklearn.ensemble import RandomForestRegressor
#from tensorflow.keras.models import load_model
#import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
import nn

'''
with open('over_fitted_rf.pkl', 'rb') as file:
    pickle_model = pickle.load(file)
with open('over_fitted_rf_force_y.pkl', 'rb') as file:
    pickle_model2 = pickle.load(file)
with open('over_fitted_rf_force_z.pkl', 'rb') as file:
    pickle_model3 = pickle.load(file)
with open('over_fitted_rf_torque_x.pkl', 'rb') as file:
    pickle_model4 = pickle.load(file)
with open('over_fitted_rf_torque_z.pkl', 'rb') as file:
    pickle_model6 = pickle.load(file)   
'''


with open('over_fitted_rf_torque_y.pkl', 'rb') as file:
    pickle_model5 = pickle.load(file)
'''
with open('GPy_SCM.dump', 'rb') as file:
    pickle_model = pickle.load(file)
'''
lf_model = pickle.load(open("low_fidelity_GP.dump","rb"))
model_terra = load_model('C://Users//fedi_vl//Documents//work//new_terra.h5')
mf_model = pickle.load(open("medium_fidelity_GP.dump","rb"))

    
#touched_ground = False
m = 4 # kg
g = 9.81 # m/s^2

k = 1 # elasticity coeffitient
c = m*g*k/0.04 
d = 2*sqrt(m*c)

mu = 0.1 # friction coef

wheel_radius = np.array([0, 0, 0.125])


'''
device_name = "/cpu:0"
with tf.device(device_name):
    print("starting models load")
    model_fx = load_model('C://Users//fedi_vl//Documents//work//resnet_fx.h5')
    print('models are loaded')
'''


def calc_forces(input_values):
    #global k
    #global model_fx
    #global device_name
    global pickle_model,pickle_model5 #,pickle_model3,pickle_model4,pickle_model2,pickle_model6
    
    r_ref = np.array( input_values[0:3])
    v_ref = np.array( input_values[3:6])
    w_ref = np.array( input_values[6:9])
    n_gra = np.array( input_values[9:12])
    h = np.array(input_values[12:4108])
    
    #force = np.zeros(3)
    #torque = np.zeros(3)    
    
    #touched a ground:
    penetration = 0.125 - r_ref[2]
    d_penetration = -v_ref[2]
    
    if r_ref[2] <= 0.125:
        force_2 = c*penetration + d_penetration*d # normal force
    else:
        force_2 = 0
    
    #v_contact = v_ref + np.cross(r_ref, w_ref)

    #force[0] = -mu*abs(force[2])*tanh(v_contact[0]*1000)
    #force[1] = -mu*abs(force[2])*tanh(v_contact[1]*1000)
    # stop the wheel when the velocity = 0
    # load models
    #print(force)
    
    '''
    with tf.device(device_name):
        
        #scan = tf.convert_to_tensor(np.reshape(np.array(h), (1,64,64,1)), dtype = float)
        scan = np.reshape(np.array(h), (1,64,64,1))
        pred = model_fx.predict(scan)[0][0]
        '''
    #multi-fidelity prediction
    # sample f_1 at xtest   
    nsamples = 1
    mu1, C1 = lf_model.predict(np.array(input_values[:12])[np.newaxis], full_cov=True)
    X_trans = nn.transform_to_scans(pd.DataFrame(np.array(input_values[12:4108])[np.newaxis]))
    nn1 = model_terra.predict(X_trans)
    # Z = (nsamples, len(X_test)) -> Each row is a distribution based on mean(mu1) and variance(C1)
    Z = np.random.multivariate_normal(mu1.flatten(),C1,nsamples)

    # push samples through f_2
    tmp_m = np.zeros((nsamples,1))
    tmp_v = np.zeros((nsamples,1))
    for i in range(0,nsamples):
        # mu, v = (len(X_test), 1)
        temp_mu, temp_v = mf_model.predict(np.hstack((np.array(input_values[:12])[np.newaxis], Z[i,:][:,None], nn1)))
        tmp_m[i,:] = temp_mu.flatten()
        tmp_v[i,:] = temp_v.flatten()

    # get posterior mean and variance
    # mean, var = (nsamples, 1)
    nonLinear_mean = np.mean(tmp_m, axis = 0)[:,None]
    #with tf.device(device_name):    
    #force_x = pickle_model.predict(np.array(input_values[:12])[:, np.newaxis].transpose())
    #force_y = pickle_model2.predict([input_values[:12]])
    #force_z = pickle_model3.predict([input_values[:12]])
    
    #torque_x = pickle_model4.predict([input_values[:12]])
    torque_y = pickle_model5.predict([input_values[:12]])
    #torque_z = pickle_model6.predict([input_values[:12]])
    #force = np.array([model_fx.predict(scan)[0][0], 0, force_2])
    
    force = np.array([nonLinear_mean[0][0], 0, force_2])
    torque = np.array([0, torque_y[0], 0])

    return np.concatenate((force,torque))
    

if __name__ == "__main__":
    pass
    #print(calc_forces(np.zeros(4108)))