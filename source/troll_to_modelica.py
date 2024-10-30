import numpy as np
import scipy.io
import sys
import mat73
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

green = "\033[32;20;53m"
bar_format = f"{green}{{l_bar}}{{bar:50}} [{{elapsed}}]{{r_bar}}"

def load_troll_data(filename, mode = None):
    data_dict = mat73.loadmat(filename)
    return reduce_data(data_dict['TrollDataRecording'], mode)

def reduce_data(data_dict, mode = None):
    reduced_data = {}
    #plt.figure(figsize=(15, 7))
    #minor_ticks = np.arange(0, 101, 5)
    #plt.yticks(minor_ticks)
    #plt.grid(alpha=1)
    #plt.axhline(y=12, color='r', linestyle='-')
    #plt.axhline(y=4, color='r', linestyle='-')
    #plt.plot(data_dict['data']['status']['statusInternal'])
    if mode == "calibration":
        # SWE is done by PROGRAMM 1
        # for calibration we are interested in the initialization period also
        start = np.where(data_dict['data']['status']['statusInternal'] == 2)[0][0]
        stop = np.where(data_dict['data']['status']['statusInternal'] == 4)[0][-1]
        test_area = range(start, stop, 1)
    elif mode == 'analysis':
        # SWE is done by PROGRAMM 1
        # for analysis we are interested only in the contact period
        test_area = (data_dict['data']['status']['statusInternal'] == 12) | (data_dict['data']['status']['statusInternal'] == 4) 
    elif mode == "scan":
        ## Scanign is done by PROGRAMM 5
        test_area = (data_dict['data']['status']['statusInternal'] == 5) | (data_dict['data']['status']['statusInternal'] == 6) 
    else:
        test_area = range()
    reduced_data['time'] = data_dict['data']['time'][test_area] 
    if len(reduced_data['time']) == 0:
        return None
    reduced_data['time'] = reduced_data['time'] -  reduced_data['time'][0]
    reduced_data['force_contact'] = data_dict['data']['fts']['force']['force_contact'][:,test_area] 
    reduced_data['force_setpoint'] = data_dict['data']['fts']['force']['force_setpoint'][:,test_area] 
    reduced_data['torque_contact'] = data_dict['data']['fts']['torque']['torque_contact'][:,test_area] 
    reduced_data['torque_setpoint'] = data_dict['data']['fts']['torque']['torque_setpoint'][:,test_area] 
    reduced_data['omega_m'] = data_dict['data']['driveUnit']['omega_m'][test_area] 
    reduced_data['omega_s'] = data_dict['data']['driveUnit']['omega_s'][test_area] 
    reduced_data['r_TCP'] = data_dict['data']['robot']['pos']['r_TCP'][:,test_area] 
    reduced_data['v_delta'] = data_dict['data']['robot']['pos']['v_delta'][:,test_area] 
    reduced_data['ABC_TCP'] = data_dict['data']['robot']['ori']['ABC_TCP'][:,test_area]
    reduced_data['w_delta'] = data_dict['data']['robot']['ori']['w_delta'][:,test_area]
    reduced_data['a_tcp'] = data_dict['data']['imu']['accelerations'][:,test_area]
    #reduced_data['experiment'] = data_dict['experiment']['swe']
    #reduced_data['id'] = data_dict['experiment']['id']
    return reduced_data

#Create dataset 
def transform_run(data):
    time = np.transpose(np.array([data['time']]))
    torque = np.transpose(data['torque_contact'])
    forces = np.transpose(data['force_contact'])
    r = np.transpose(data['r_TCP']) # relative coordinates
    v_delta = np.transpose(data['v_delta']) # horizontal velocity
    omega_m = np.transpose(np.array([data['omega_m']])) # rotation of the motor

    abc = np.transpose(data['ABC_TCP']) # something related to the steering angle? or normal force?
    w = np.transpose(np.array(data['w_delta'])) # angular velocity
    # concat columns
    ### ATTENTION: don't touch the sequence of these values, because some Modelica code depends on that!
    ## col indx: [0], [1,2,3], [4,5,6], [7,8,9], [10,11,12], [13], [14,15,16], [17,18,19]
    X = np.concatenate((time, r, omega_m, v_delta, abc, w, forces, torque), axis = 1)
    return X

def create_modelica_compatible(troll_dir = "path/to/troll_experiments"): #"H:/NOT_SAVED_TO_BACKUP/TROLL_data/experiments_31.01.23/"):
    for name in tqdm(os.listdir(troll_dir), bar_format=bar_format):
        run = load_troll_data(troll_dir + name, mode="analysis")
        if run == None:
            continue
        #plt.plot(run['r_TCP'][0], run['r_TCP'][2])
        #plt.show()
        for_dymula = transform_run(run)
        name = 'modelica_compatible_' + str(name[4:10]) + '.mat'
        scipy.io.savemat('C:/Users/fedi_vl/Documents/work/modelica_compatible/' + name, mdict={'myrun': for_dymula})
