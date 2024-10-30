import h5py
import numpy as np
import sys
from tqdm import tqdm

def change_surface(input_name, output_name, num_of_steps):
    soil = h5py.File(input_name, 'r')

    posx = np.array(soil['Step#0']['position_x'])
    posy = np.array(soil['Step#0']['position_y'])
    posz = np.array(soil['Step#0']['position_z'])

    nTimeSteps = len(soil.keys()) ## number of Step#X
    attributeNames = list(soil['Step#0']) #25 attributes - coords, veloc, etc
    nParticles = len(posx) # number of points in the soil? 111k
    
    xOff = -0.75
    zOff = -0.50
    lambda_ = 50 # the more - the sharper hills
    A = 0.07 # from 0.07 to 0.05, so that wheel can trespase it
    #xTest = np.linspace(-0.09, 0.09, 101)
    #zTest = A*np.cos(lambda_/2/np.pi*xTest+xOff) + zOff

    inIDs = []
    for i in range(nParticles):
        x = posx[i]
        y = posy[i]
        z = posz[i]
        #if x < -xOff:
        if z < (A*np.cos(lambda_/2/np.pi*x+xOff) + zOff):
            inIDs.append(i)

    outputSoil = h5py.File(output_name, 'w')
    for step in tqdm(range(num_of_steps)):
        if step > 400:
            step_ = 400
        else:
            step_ = step
        grop = outputSoil.create_group('/Step#' + str(step) + '/') 
        for a in attributeNames:
            data = np.array(soil["Step#"+ str(step_)][a])
            grop.create_dataset('/Step#' + str(step) + '/' + a, data=data[inIDs])

    outputSoil.close()

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    num_of_steps = int(sys.argv[3])
    
    change_surface(input_file_path, output_file_path, num_of_steps)
