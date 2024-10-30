import sys
import numpy as np
sys.path.append("D:/Vlad/Dymola_2022x_win/Dymola 2022x/Modelica/Library/python_interface/dymola.egg") # append path to the modelica's interface
from dymola.dymola_interface import DymolaInterface
from dymola.dymola_exception import DymolaException
import pandas as pd
from tqdm import tqdm
from time import time
import h5py
import logging
import scipy.io

## append paths to all needed files
path = "D:/Vlad/Dymola_2022x_win/Dymola 2022x/bin64/Dymola.exe"
lib = ["D:/Vlad/library/ContactDynamics/package.mo", "D:/Vlad/scripts/run_bekker_model.mo", "D:/Vlad/work/terra.mo", 
       "D:/Vlad/library/SR_Utilities/package.mo", "D:/Vlad/library/Visualization2/Modelica/Visualization2/package.mo",
       "D:/Vlad/library/Visualization2Overlays/package.mo", "D:/Vlad/library/MyPackage/package.mo", "D:/Vlad/library/HDF5/package.mo"]
work = "D:/Vlad/work/work_python"

def init_dymola(path,work,libs):
    dymola = DymolaInterface(path)
    for lib in libs:
        dymola.openModel(lib)
    dymola.cd(work)    
    return dymola

def request_run(support_vector):
    name = "MyPackage.Experiments.TrollDataTest"
    output_name = "TrollDataTest" + "name"
    logging.info("launching a modelica simulation...")
    result, values = dymola.simulateMultiResultsModel(name, 
                                                  stopTime=5, 
                                                  numberOfIntervals=500,
                                                  method="Rkfix4",
                                                  #fixedstepsize=0.01,
                                                  initialNames=[f'elevationMap.surface.support_h[{i+1}]' for i in range(13)] + 
                                                       ['wheel_With_MLTM.sampleInterval'],
                                                  initialValues=[supvec + [0.01]],
                                                  resultFile=output_name)                                        
    if not result:
        logging.error(dymola.getLastErrorLog())
        
    logging.info("loading results from the simulation...")
    filename = work + '/Out.h5' # temporary file generated in the modelica's working directory
    data = h5py.File(filename, 'r')
    input_ = []
    output_ = []
    for i in range(data['input'].shape[0]):
        input_.append(data['input'][i][1]) # 0 - index
        output_.append(data['output'][i][1]) 

    input_ = pd.DataFrame(input_) 
    output_ = pd.DataFrame(output_)
    df = pd.concat([output_, input_], axis = 1)
    
    return df

def partsival_to_mat(path):
    dem_output = pd.read_table(path, sep = '\t', skiprows=4)
    time = np.transpose(np.array([dem_output['time']]))
    torque = np.transpose(dem_output[[' MX', ' MY',' MZ']])
    forces = np.transpose(dem_output[[' FX', ' FY',' FZ']])
    r = np.transpose(dem_output[[' posX', ' posY',' posZ']])
    #v = np.transpose(dem_output[[' vX', ' vY',' vZ']])
    w_prev = np.array([dem_output[' wY']])
    w = np.transpose(w_prev)

    # concat columns
    X = np.concatenate((time, r.transpose(), w, forces.transpose(), torque.transpose()), axis = 1)
    name = 'C://Users/fedi_vl/Documents/work/partsival_run.mat' # the same path specified in modelica script
    scipy.io.savemat(name, mdict={'myrun': X})


def det_supvec(input_name):
    supvec = []
    soil = h5py.File(input_name, 'r')

    posx = np.array(soil['Step#0']['position_x'])
    posy = np.array(soil['Step#0']['position_y'])
    posz = np.array(soil['Step#0']['position_z'])
    
    for i in range(-30, 35, 5):
        supvec.append(max(posz[(posx < i/100 + 0.005) & (posx > i/100 - 0.005)]))
    return supvec

def create_heightmap_from_partsival(scenario_name):
    dem_output = pd.read_table(f"H:/NOT_SAVED_TO_BACKUP/wheel-sim/{scenario_name}/results-wheel-sim.out", sep = '\t', skiprows=4)
    f = h5py.File(f'H://NOT_SAVED_TO_BACKUP/wheel-sim/{scenario_name}/soil-wheel-sim.h5part', 'r')
    dem_scans_list = []
    relation_coef = (dem_output.shape[0]/10)/len(f.keys())
    for i in tqdm(range(int(dem_output.shape[0]/10))):
        name = 'Step#' + str(int(i*relation_coef))
        # determine which part of the surface to take (from -0.16 to 0.16 from the centre of the wheel)
        # chose current position x and y
        x_cur = dem_output.loc[i*10, " posX"] # its moving from the 0 to -0.06
        y_cur = dem_output.loc[i*10, " posY"]
        x = list(f[name]['position_x'])
        framing_indx_x = (np.array(x) > x_cur-0.16) & (np.array(x) < x_cur+0.16)
        # in the y-directipn, center of the wheel is always on the edge close to -0.08
        y = list(f[name]['position_y'])
        h = list(f[name]['position_z'])
        # split the point cloud into the small pilars and take only the biggest height in each pillar
        max_x = max(np.array(x)[framing_indx_x])
        min_x = min(np.array(x)[framing_indx_x])
        max_y = max(np.array(y))
        min_y = -max_y

        max_polling_over_point_cloud = []
        # stretch the height map in the respective parts... 
        for x_iter in np.linspace(min_x, max_x, 64):
            for y_iter in np.linspace(min_y, max_y, 32): # only half of the y-plane
                # frame current window over the point cloud
                framing_indx_x_ = (np.array(x) > x_iter-0.005) & (np.array(x) < x_iter+0.005)
                framing_indx_y_ = (np.array(y) > y_iter-0.005) & (np.array(y) < y_iter+0.005)

                max_polling_over_point_cloud.append(max(np.array(h)[framing_indx_x_*framing_indx_y_]))

        half_scan = np.reshape(np.transpose(np.array(max_polling_over_point_cloud)), (64, 32))
        full_scan = np.concatenate((np.fliplr(half_scan), half_scan), axis = 1)
        dem_scans_list.append(full_scan)

    dem_scans_df = pd.DataFrame(np.reshape(dem_scans_list, (int(dem_output.shape[0]/10), 64*64)))
    dem_scans_df.to_csv(f"D://Vlad/partsival_scm_terra_data/partsival_{scenario_name}_surface.csv")

if __name__ == "__main__":
    scenario_name = sys.argv[1] # scenario upward_slippage or steering
    path_to_partsival_output = f'H://NOT_SAVED_TO_BACKUP/wheel-sim/{scenario_name}/results-wheel-sim.out'
    path_to_soil_pointcloud = f'H://NOT_SAVED_TO_BACKUP/wheel-sim/{scenario_name}/soil-wheel-sim.h5part'
    # create heightmaps of the Partsival simulation 
    create_heightmap_from_partsival(scenario_name)
    # transform the Partsival trajectory output file to the .mat Modelica-readable format
    partsival_to_mat(path_to_partsival_output)
    logging.info("Partsival data translated to .mat modelica-readable format")
    # determine the 'support vector' for the Modelica soil's surface 
    supvec = det_supvec(path_to_soil_pointcloud)
    # launch modelica to replicate Partsival's simulation
    dymola = init_dymola(path,work,lib)
    logging.info("Starting transformation to the SCM simulation...")
    run = request_run(supvec)
    df = pd.DataFrame(run)
    csv_name = f"D://Vlad/partsival_scm_terra_data/partsival_{scenario_name}_to_scm.csv"
    logging.info("Writing data to csv...")
    df.to_csv(csv_name)

