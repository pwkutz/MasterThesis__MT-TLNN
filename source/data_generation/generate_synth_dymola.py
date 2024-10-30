import sys
import numpy as np
sys.path.append("D:/Vlad/Dymola_2022x_win/Dymola 2022x/Modelica/Library/python_interface/dymola.egg") # append path to the modelica's interface
from dymola.dymola_interface import DymolaInterface
from dymola.dymola_exception import DymolaException
#import pandas as pd
from tqdm import tqdm
from time import time
import scipy.io
import h5py
import logging
import math

    ###############################################################################################################################
    ###### I don't need no more TerRA's replications over the fixed SCM trajectories, because I need only the same training  ######
    ###### data points for the highest-fidelity data... So these poitns I can reproduce over TROLL trajectories              ######
    ###### and for TerRA or SCM surrogates I need only synthetic TerRA and SCM data                                          ######
    ###############################################################################################################################

class SynthDataGen():
    def __init__(self, allruns_filename):
        ## append paths to all needed modelica scripts
        self.path = "D:/Vlad/Dymola_2022x_win/Dymola 2022x/bin64/Dymola.exe"
        self.libs = ["D:/Vlad/library/ContactDynamics/package.mo", "D:/Vlad/scripts/run_bekker_model.mo", "D:/Vlad/work/terra.mo", 
            "D:/Vlad/library/SR_Utilities/package.mo", "D:/Vlad/library/Visualization2/Modelica/Visualization2/package.mo",
            "D:/Vlad/library/Visualization2Overlays/package.mo", "D:/Vlad/library/Mufi/package.mo", "D:/Vlad/library/HDF5/package.mo"]
        self.work = "D:/Vlad/work/work_python"
        self.allruns_filename = allruns_filename
        self.init_dymola()
        self.allruns = h5py.File(self.allruns_filename, 'a') # create the final h5file, where all runs will be recorded

    def init_dymola(self):
        self.dymola = DymolaInterface(self.path)
        for lib in self.libs:
            self.dymola.openModel(lib)
        self.dymola.cd(self.work)
    
    def create_table_values(self, value_vector, run_length, table_name):
        ## create initial variables for the timtable in dymola simulation
        table = []
        # timetable which controls the velocity of the wheel in each of the 15 seconds of simulation
        for i in range(1, run_length+1): 
            time1 = table_name + 'Table.table[' + str(i) + ', 1]'
            time2 = table_name + 'Table.table[' + str(i) + ', 2]'
            table.append(time1)
            table.append(time2)
        ## and values for it
        table_values = np.transpose(np.concatenate([np.linspace(0, run_length-1, run_length)[np.newaxis], 
                                                        value_vector[np.newaxis]], axis = 0)).flatten()
        return table, table_values
    
    def run(self, support_vector, velocity_vector, steering_vector, indx, run_length, method, wheel_mass):
        # elevation controls the slope of the surface beneath the wheel    
        elevation_variables_name = []
        for i in range(1, len(support_vector) + 3):
            elevation_variables_name.append('elevationMap.surface.support_h[' + str(i) + ']')
        velocity_table_name, velocity_table_values = self.create_table_values(velocity_vector, run_length, "velocity")
        steering_table_name, steering_table_values = self.create_table_values(steering_vector, run_length, "steering")
        initial_variables_name = elevation_variables_name + velocity_table_name + steering_table_name + ['oneWheelBody.m']
        initial_variables_values = [0, 0] + list(support_vector) + list(velocity_table_values) + list(steering_table_values) + [wheel_mass]
        name = "Mufi.Experiments." + method + "_datagathering_steering"
        output_name = 'scm_tmp.mat'
        result, _ = self.dymola.simulateMultiResultsModel(name, 
                                                    stopTime=run_length, 
                                                    numberOfIntervals=10000,
                                                    method="Rkfix4",
                                                    fixedstepsize=0.001,
                                                    initialNames=initial_variables_name,
                                                    initialValues=[initial_variables_values],
                                                    resultFile=output_name)
        if not result:
            print(self.dymola.getLastErrorLog())
        filename = self.work + '/out_' + method + '_fixed.h5'
        runfile = h5py.File(filename, 'r')
        # append run to the final file
        # 0,1,2 - r_ref; 3,4,5 - v_ref, 6,7,8 - w_ref, 9,10,11 - n_gravity
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/input', data=runfile["input"])
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/output', data=runfile["output"])
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/aux', data=runfile["aux"])
        runfile.close()

    def sliding(self, support_vector, indx, run_length, method, wheel_mass):
        # elevation controls the slope of the surface beneath the wheel
        # TODO: remake this
        elevation_variables_name = ['elevationMap.surface.support_h[1]',
                                    'elevationMap.surface.support_h[2]',
                                    'elevationMap.surface.support_h[3]',
                                    'elevationMap.surface.support_h[4]',
                                    'elevationMap.surface.support_h[5]',
                                    'elevationMap.surface.support_h[6]',
                                    'elevationMap.surface.support_h[7]',
                                    'elevationMap.surface.support_h[8]']
        initial_variables_name = elevation_variables_name + ['oneWheelBody.m']
        initial_variables_values = list(support_vector) + [wheel_mass]
        name = "Mufi.Experiments." + method + "_datagathering_sliding"
        output_name = 'scm_tmp.mat'
        result, _ = self.dymola.simulateMultiResultsModel(name, 
                                                    stopTime=run_length, 
                                                    numberOfIntervals=10000,
                                                    method="Rkfix4",
                                                    fixedstepsize=0.001,
                                                    initialNames=initial_variables_name,
                                                    initialValues=[initial_variables_values],
                                                    resultFile=output_name)
        if not result:
            print(self.dymola.getLastErrorLog())
        filename = self.work + '/out_' + method + '_fixed.h5'
        runfile = h5py.File(filename, 'r')
        # append run to the final file
        # 0,1,2 - r_ref; 3,4,5 - v_ref, 6,7,8 - w_ref, 9,10,11 - n_gravity
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/input', data=runfile["input"])
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/output', data=runfile["output"])
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/aux', data=runfile["aux"])
        runfile.close()

    def uphill(self, support_vector, indx, run_length, method, target_speed):
        # elevation controls the slope of the surface beneath the wheel
        # TODO: remake this
        elevation_variables_name = ['elevationMap.surface.support_h[1]',
                                    'elevationMap.surface.support_h[2]']
        initial_variables_name = elevation_variables_name + ["velocityRamp.height"] 
        initial_variables_values = support_vector + [target_speed]
        name = "Mufi.Experiments." + method + "_datagathering_uphill"
        output_name = 'tmp.mat'
        result, _ = self.dymola.simulateMultiResultsModel(name, 
                                                    stopTime=run_length, 
                                                    numberOfIntervals=10000,
                                                    method="Rkfix4",
                                                    fixedstepsize=0.001,
                                                    initialNames=initial_variables_name,
                                                    initialValues=[initial_variables_values],
                                                    resultFile=output_name)
        if not result:
            print(self.dymola.getLastErrorLog())
        filename = self.work + '/out_' + method + '_uphill.h5'
        runfile = h5py.File(filename, 'r')
        # append run to the final file
        # 0,1,2 - r_ref; 3,4,5 - v_ref, 6,7,8 - w_ref, 9,10,11 - n_gravity
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/input', data=runfile["input"])
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/output', data=runfile["output"])
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/aux', data=runfile["aux"])
        runfile.close()

    def stop(self, support_vector, velocity_vector, steering_vector, indx, run_length, method, wheel_mass):
        # elevation controls the slope of the surface beneath the wheel
        elevation_variables_name = []
        for i in range(1, len(support_vector) + 3):
            elevation_variables_name.append('elevationMap.surface.support_h[' + str(i) + ']')
        velocity_table_name, velocity_table_values = self.create_table_values(velocity_vector, run_length, "velocity")
        steering_table_name, steering_table_values = self.create_table_values(steering_vector, run_length, "steering")
        initial_variables_name = elevation_variables_name + velocity_table_name + steering_table_name + ['oneWheelBody.m']
        initial_variables_values = [0, 0] + list(support_vector) + list(velocity_table_values) + list(steering_table_values) + [wheel_mass]
        name = "Mufi.Experiments." + method + "_datagathering_stopping"
        output_name = 'scm_tmp.mat'
        result, _ = self.dymola.simulateMultiResultsModel(name, 
                                                    stopTime=run_length, 
                                                    numberOfIntervals=10000,
                                                    method="Rkfix4",
                                                    fixedstepsize=0.001,
                                                    initialNames=initial_variables_name,
                                                    initialValues=[initial_variables_values],
                                                    resultFile=output_name)
        if not result:
            print(self.dymola.getLastErrorLog())
        filename = self.work + '/out_' + method + '_fixed.h5'
        runfile = h5py.File(filename, 'r')
        # append run to the final file
        # 0,1,2 - r_ref; 3,4,5 - v_ref, 6,7,8 - w_ref, 9,10,11 - n_gravity
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/input', data=runfile["input"])
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/output', data=runfile["output"])
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/aux', data=runfile["aux"])
        runfile.close()

    def impact(self, velocity_vector, height_vector, indx, run_length, method):
        #velocity_table_name, velocity_table_values = self.create_table_values(velocity_vector, run_length, "velocity")
        #vertical_table_name, vertical_table_values = self.create_table_values(height_vector, run_length, "vertical")
        initial_variables_name = velocity_table_name + vertical_table_name 
        initial_variables_values = list(velocity_table_values) + list(vertical_table_values)
        name = "Mufi.Experiments.ExampleV_" + method
        output_name = 'impact_tmp.mat'
        result, _ = self.dymola.simulateMultiResultsModel(name, 
                                                    stopTime=run_length, 
                                                    numberOfIntervals=10000,
                                                    method="Rkfix4",
                                                    fixedstepsize=0.001,
                                                    initialNames=initial_variables_name,
                                                    initialValues=[initial_variables_values],
                                                    resultFile=output_name)
        if not result:
            print(self.dymola.getLastErrorLog())
        filename = self.work + '/out_' + method + 'impact.h5'
        runfile = h5py.File(filename, 'r')
        # append run to the final file
        # 0,1,2 - r_ref; 3,4,5 - v_ref, 6,7,8 - w_ref, 9,10,11 - n_gravity
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/input', data=runfile["input"])
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/output', data=runfile["output"])
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/aux', data=runfile["aux"])
        runfile.close()

    def bulldoze(self, indx, run_length, method, target_speed):
        initial_variables_name = ["velocityRamp.height"] 
        initial_variables_values = [target_speed]
        name = "Mufi.Experiments.ExampleV_" + method + "_table"
        output_name = 'blldoze_tmp.mat'
        result, _ = self.dymola.simulateMultiResultsModel(name, 
                                                    stopTime=run_length, 
                                                    numberOfIntervals=10000,
                                                    method="Rkfix4",
                                                    fixedstepsize=0.001,
                                                    initialNames=initial_variables_name,
                                                    initialValues=[initial_variables_values],
                                                    resultFile=output_name)
        if not result:
            print(self.dymola.getLastErrorLog())
        filename = self.work + '/out_bulldoze.h5'
        runfile = h5py.File(filename, 'r')
        # append run to the final file
        # 0,1,2 - r_ref; 3,4,5 - v_ref, 6,7,8 - w_ref, 9,10,11 - n_gravity
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/input', data=runfile["input"])
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/output', data=runfile["output"])
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/aux', data=runfile["aux"])
        runfile.close()

    def sinusoidal_force(self, sinus_amplitude, indx, run_length, method):
        initial_variables_name = ["sine.amplitude"]
        initial_variables_values = [sinus_amplitude]
        name = "Mufi.Experiments." + str(method) + "_datagathering_with_sinusoidal_force"
        output_name = 'sinus_force_tmp.mat'
        result, _ = self.dymola.simulateMultiResultsModel(name, 
                                                    stopTime=run_length, 
                                                    numberOfIntervals=10000,
                                                    method="Rkfix4",
                                                    fixedstepsize=0.001,
                                                    initialNames=initial_variables_name,
                                                    initialValues=[initial_variables_values],
                                                    resultFile=output_name)
        if not result:
            print(self.dymola.getLastErrorLog())
        filename = self.work + '/out_sinus_force.h5'
        runfile = h5py.File(filename, 'r')
        # append run to the final file
        # 0,1,2 - r_ref; 3,4,5 - v_ref, 6,7,8 - w_ref, 9,10,11 - n_gravity
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/input', data=runfile["input"])
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/output', data=runfile["output"])
        self.allruns.create_dataset('/' + str(indx) + '/' + method + '/aux', data=runfile["aux"])
        runfile.close()


    def create_fixed_trajectory(self):
        ## this function is needed in case we want to recreate TerRA runs over fixed SCM trajectories
        # Transform TerRA\SCM Modelica outputs to the TROLL-like data, to use it like a fixed trajectory for another simulation
        #read .mat file from the scm simulation
        scm_filename = self.work + '/scm_fixed_trajectory.h5'
        scm_sim = h5py.File(scm_filename)
        data_out=np.array([[np.array(scm_sim['r'])[x][0] for x in range(scm_sim['r'].shape[0])],#time
                    [np.array(scm_sim['r'])[x][1][0] for x in range(scm_sim['r'].shape[0])], # r_x
                    [np.array(scm_sim['r'])[x][1][1] for x in range(scm_sim['r'].shape[0])], # r_y
                    [np.array(scm_sim['r'])[x][1][2] for x in range(scm_sim['r'].shape[0])], # r_z
                    [np.array(scm_sim['w'])[x][1] for x in range(scm_sim['w'].shape[0])], # w_y 
                    [np.array(scm_sim['f'])[x][1][0] for x in range(scm_sim['f'].shape[0])], # f_x
                    [np.array(scm_sim['f'])[x][1][1] for x in range(scm_sim['f'].shape[0])], # f_y
                    [np.array(scm_sim['f'])[x][1][2] for x in range(scm_sim['f'].shape[0])], # f_z
                    [np.array(scm_sim['t'])[x][1][0] for x in range(scm_sim['t'].shape[0])], # t_x
                    [np.array(scm_sim['t'])[x][1][1] for x in range(scm_sim['t'].shape[0])], # t_y
                    [np.array(scm_sim['t'])[x][1][2] for x in range(scm_sim['t'].shape[0])]]) # t_z
        mdic={'scmData':np.transpose(data_out)}
        scipy.io.savemat(self.work + '/scmData_fixed.mat', mdic) 
        scm_sim.close()

    def generate(self, num_of_exp, run_type):
        support_vector_size = 6 #int(sys.argv[2]) - 2 # length of the "support vector" of the surface
        # setting for tqdm
        green = "\033[32;20;53m"
        bar_format = f"{green}{{l_bar}}{{bar:50}} [{{elapsed}}]{{r_bar}}"
        amplitude_distr = np.linspace(10, 40, num_of_exp)
        for indx in tqdm(range(num_of_exp), bar_format=bar_format, desc="synthetic data generation"):
            if run_type == "run":
                support_vector = np.cumsum(np.random.normal(loc = 0.01, scale = 0.06, size=support_vector_size))
                run_length = 20
                velocity_vector = np.random.normal(loc = 1, scale = 2, size=20) #np.random.randint(-1, 3, size = run_length) 
                steering_vector = np.random.normal(loc = 0, scale = 0.5, size=20) #np.random.randint(0, 7, size = run_length)/10
                self.allruns['/' + str(indx) +'/supvec'] = support_vector
                # play with the wheel mass in order to show the variance in normal force and how it correlates with the sinkage
                wheel_mass = np.random.choice([1, 2, 3, 4, 6, 8, 12], 1, p=[0.02, 0.08, 0.15, 0.5, 0.15, 0.08, 0.02])[0]
                try:
                    self.run(support_vector, velocity_vector, steering_vector, indx, run_length, "scm", wheel_mass) # first run scm
                    #self.create_fixed_trajectory() # create fixed scm trajectory for the terra
                    self.run(support_vector, velocity_vector, steering_vector, indx, run_length, "terra", wheel_mass) # run terra on the fixed trajectory from the scm
                except:
                    logging.warning("Restarting connection with Dymola...") # sometimes connection with dymola dies ...
                    self.init_dymola()
                    self.run(support_vector, velocity_vector, steering_vector, indx, run_length, "scm", wheel_mass)
                    #self.create_fixed_trajectory()
                    self.run(support_vector, velocity_vector, steering_vector, indx, run_length, "terra", wheel_mass)    

            elif run_type == "break":
                first_rand = np.random.random(1)
                if first_rand > 0.6:
                    support_vector = np.cumsum(np.repeat(np.random.uniform(-0.25, -0.15, size=1), support_vector_size + 2))
                elif first_rand < 0.4:
                    support_vector = np.cumsum(np.repeat(np.random.uniform(0.1, 0.15, size=1), support_vector_size + 2))
                else:
                    support_vector = np.random.uniform(0, 0.01, size = support_vector_size + 2)
                run_length = 8 # 5 sec of acceleration then 2 sec for breaking. arbitrary

                if np.random.random(1) > 0.2:
                    velocity_vector = np.append(np.append([0], np.cumsum(np.random.normal(loc = 2, scale = 1, size=run_length - 3))), [0, 0])
                else:
                    velocity_vector = np.repeat(0, run_length)
                steering_vector = np.random.normal(loc = 0, scale = 0.2, size=run_length) 
                self.allruns['/' + str(indx) +'/supvec'] = support_vector
                # play with the wheel mass in order to show the variance in normal force and how it correlates with the sinkage
                wheel_mass = np.random.choice([1, 2, 3, 4, 6, 8, 12], 1, p=[0.02, 0.08, 0.15, 0.5, 0.15, 0.08, 0.02])[0]
                try:
                    self.stop(support_vector, velocity_vector, steering_vector, indx, run_length, "scm", wheel_mass) # first run scm
                    self.stop(support_vector, velocity_vector, steering_vector, indx, run_length, "terra", wheel_mass) # run terra on the fixed trajectory from the scm
                except:
                    logging.warning("Restarting connection with Dymola...") # sometimes connection with dymola dies ...
                    self.init_dymola()
                    self.stop(support_vector, velocity_vector, steering_vector, indx, run_length, "scm", wheel_mass)
                    self.stop(support_vector, velocity_vector, steering_vector, indx, run_length, "terra", wheel_mass)                
            
            elif run_type == "slide":
                #support_vector = np.cumsum(np.repeat(np.random.uniform(-0.25, -0.15, size=1), support_vector_size + 2))
                support_vector = np.repeat(0, support_vector_size + 2)
                print(support_vector)
                run_length = 5
                self.allruns['/' + str(indx) +'/supvec'] = support_vector
                # play with the wheel mass in order to show the variance in normal force and how it correlates with the sinkage
                wheel_mass = np.random.choice([1, 2, 3, 4, 10, 20, 30], 1, p=[0.02, 0.08, 0.15, 0.5, 0.15, 0.08, 0.02])[0]
                try:
                    self.sliding(support_vector, indx, run_length, "scm", wheel_mass) # first run scm
                    #self.create_fixed_trajectory() # create fixed scm trajectory for the terra
                    self.sliding(support_vector, indx, run_length, "terra", wheel_mass) # run terra on the fixed trajectory from the scm
                except:
                    logging.warning("Restarting connection with Dymola...") # sometimes connection with dymola dies ...
                    self.init_dymola()
                    self.sliding(support_vector, indx, run_length, "scm", wheel_mass)
                    #self.create_fixed_trajectory()
                    self.sliding(support_vector, indx, run_length, "terra", wheel_mass)   

            elif run_type == "bulldoze":
                #support_vector = np.cumsum(np.repeat(np.random.uniform(-0.25, -0.15, size=1), support_vector_size + 2))
                #support_vector = np.repeat(0, support_vector_size + 2)
                #print(support_vector)
                run_length = 10
                # play with the wheel mass in order to show the variance in normal force and how it correlates with the sinkage
                target_speed = np.random.uniform(1, 5)
                try:
                    self.bulldoze(indx, run_length, "scm", target_speed) 
                    self.bulldoze(indx, run_length, "terra", target_speed) 
                except:
                    logging.warning("Restarting connection with Dymola...") # sometimes connection with dymola dies ...
                    self.init_dymola()
                    self.bulldoze(indx, run_length, "scm", target_speed)
                    self.bulldoze(indx, run_length, "terra", target_speed)   
    
            elif run_type == "uphill":
                run_length = 10
                support_vector = np.random.uniform(0.5, 2.5)
                target_speed = np.random.uniform(1, 5)
                try: 
                    self.uphill([-support_vector, support_vector], indx, run_length, "scm", target_speed)
                    #self.uphill([-support_vector, support_vector], indx, run_length, "terra", target_speed) 
                except:
                    logging.warning("Restarting connection with Dymola...") # sometimes connection with dymola dies ...
                    self.init_dymola()
                    self.uphill([-support_vector, support_vector], indx, run_length, "scm", target_speed)
                    #self.uphill([-support_vector, support_vector], indx, run_length, "terra", target_speed)   

            elif run_type == "impact":
                run_length = 20
                velocity_vector = np.concatenate((np.repeat(0, 5), np.linspace(np.random.uniform(0.1, 0.5), np.random.uniform(0.1, 0.5), run_length - 5)))
                boundaries = [-0.1, -0.5]
                break_points = np.random.uniform(boundaries[0], boundaries[1], 4)
                height_vector = np.concatenate((np.linspace(0, break_points[0], math.floor(run_length/5), endpoint=False), 
                                                np.linspace(break_points[0], break_points[1], math.floor(run_length/5), endpoint=False),
                                                np.linspace(break_points[1], break_points[2], math.floor(run_length/5), endpoint=False),
                                                np.linspace(break_points[2], break_points[3], run_length - 3*math.floor(run_length/5), endpoint=False)))
                #self.allruns['/' + str(indx) +'/supvec'] = support_vector
                #wheel_mass = np.random.choice([1, 2, 3, 4, 6, 8, 12], 1, p=[0.02, 0.08, 0.15, 0.5, 0.15, 0.08, 0.02])[0]
                try:
                    self.impact(velocity_vector, height_vector, indx, run_length, "scm") # first run scm
                    #self.create_fixed_trajectory() # create fixed scm trajectory for the terra
                    self.impact(velocity_vector, height_vector, indx, run_length, "terra") # run terra on the fixed trajectory from the scm
                except:
                    logging.warning("Restarting connection with Dymola...") # sometimes connection with dymola dies ...
                    self.init_dymola()
                    self.impact(velocity_vector, height_vector, indx, run_length, "scm")
                    #self.create_fixed_trajectory()
                    self.impact(velocity_vector, height_vector, indx, run_length, "terra")    
            elif run_type == "sinus_force":
                run_length = 20
                sinus_amplitude = amplitude_distr[indx]
                try:
                    self.sinusoidal_force(sinus_amplitude, indx, run_length, "scm") # first run scm
                    #self.sinusoidal_force(sinus_amplitude, indx, run_length, "terra") # run terra on the fixed trajectory from the scm
                except:
                    logging.warning("Restarting connection with Dymola...") # sometimes connection with dymola dies ...
                    self.init_dymola()
                    self.sinusoidal_force(sinus_amplitude, indx, run_length, "scm")
                    #self.sinusoidal_force(sinus_amplitude, indx, run_length, "terra")    
            else:
                print('Wrong runtype! Please indicate either "run" for normal run or "impact" for hard push into the ground.')
        self.allruns.close()
        print("Synthetic data from TerRA and SCM are generated and stored in ", self.allruns_filename)