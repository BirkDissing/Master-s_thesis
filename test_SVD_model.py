import numpy as np
import math
import pandas as pd
from ase import Atoms
from ase import neighborlist
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.parallel import world
from gpaw import GPAW
from gpaw.occupations import FermiDirac
from ase.md import MDLogger
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from shaved_var import *
from gpaw import setup_paths
setup_paths.insert(0, "/kemi/williamb/opt/gpaw-datasets/gpaw-setups-0.9.20000")

input = 8
pred_step = 1
max_lag = 5
alpha = 3.8e-05
method = "SVD"
n_time_steps = 50000
dt = 0.5*units.fs
T = 400  # K
file = "SVD_regularized"+str(input)+"_"+str(pred_step)+".xyz"
md = True


# Function which predicts the forces on the atoms in a molecule for a certain number of time steps
#_____Inputs______
# file - String with the name of the xyz file containing information on the molecule
# input - Positive integer with the number of data points used as input in training the OLS VAR models
# pred_step - Positive integer which decides the number of time steps predicted by the OLS VAR models
# order - Positive integer which decides the numper of points in the past is used to predict the next point
#
#_____Output______
# Returns a 3D np array with the predicted forces. The array as the following shape [N, 3, p], with N being the number of atoms in the molecule
# and p being the number of time steps predicted
def predict_forces(file=file, input=input, pred_step=pred_step, max_lag=max_lag, alpha=alpha, method=method):

    mol = read(file, index=slice(-input, None))
    
    n_atoms = mol[0].get_global_number_of_atoms()
    predicted_forces = np.zeros((n_atoms, 3, pred_step))
    forces = np.zeros((input, 3*n_atoms))

    #Get the forces in correct format for model
    for i in range(input):
        forces[i,:] = mol[i].get_forces().ravel()
    
    #Fit model
    VAR_model = ShavedVAR(forces).fit(max_lag, alpha, method=method)

    #Predict forces with model
    predicted_forces = VAR_model.forecast(forces[-max_lag:], pred_step)
        
    return predicted_forces


convergence = {
    "energy": 1e-7,
    "density": 1e-7
}

mol = read("./EtOH_opt.traj")
# mol = repartition_masses(mol, factor=4)

calc = GPAW(
    occupations=FermiDirac(0.05),
    xc="PBE",
    mode="lcao",
    basis="PBE.sz",
    convergence=convergence,
    kpts=(1, 1, 1),
    txt="SVD_regularized"+str(input)+"_"+str(pred_step)+".txt"
)
mol.set_calculator(calc)
mol.center(vacuum=10)
mol.set_pbc(True)

# Set the momenta corresponding to T=300K
MaxwellBoltzmannDistribution(mol, temperature_K=T)
 
# We want to run MD with constant energy using the VelocityVerlet algorithm.
dyn = VelocityVerlet(mol, dt)  
# 0.5 fs time step.

traj = TrajectoryWriter("SVD_regularized_traj"+str(input)+"_"+str(pred_step)+".traj", "w", mol)

df = pd.DataFrame(columns=['md','C1(x)','C1(y)','C1(z)','C2(x)','C2(y)','C2(z)','O(x)','O(y)','O(z)','H1(x)','H1(y)','H1(z)','H2(x)','H2(y)','H2(z)','H3(x)','H3(y)','H3(z)','H4(x)','H4(y)','H4(z)','H5(x)','H5(y)','H5(z)','H6(x)','H6(y)','H6(z)'])
def print_forces(a=mol, md=1, forces=np.array([None])):
    # Prints the forces
    if forces.any()==None:
        forces = a.get_forces()
    row = []
    row.append(md)
    for i in range(len(a)):
        for j in range(3):
            row.append(forces[i,j])
    df.loc[len(df)] = row
    df.to_csv("SVD_regularized_df"+str(input)+"_"+str(pred_step)+".csv")


for i in range(n_time_steps):
    if i%input == 0 and i!=0:
        predicted_forces = predict_forces()
        for j in range(pred_step):
            #Get masses for the atoms in the molecule
            masses = mol.get_masses()[:, np.newaxis]

            #Get the forces, momenta, and positions for the current step
            #forces = mol[i].get_forces()
            forces = predicted_forces[i+1,:] 
            forces = forces.reshape(mol.get_global_number_of_atoms(), 3)
            print_forces(md=0, forces=forces)
            
            p = mol.get_momenta()
            r = mol.get_positions()
            
            #Calculate new momenta and positions
            p += 0.5 * dt * forces
            mol.set_positions(r + dt * p / masses)
            
            #Was in ase.step. Unsure if needed
            if mol.constraints:
                p = (mol.get_positions() - r) * masses / dt

            #Momenta needs to be stored before possible calculations of forces
            mol.set_momenta(p, apply_constraint=False)

            #Forces for next step is found either using predicted forces or gpaw calculator
            if j<pred_step-1:
                forces = predicted_forces[i+1,:] 
                forces = forces.reshape(mol.get_global_number_of_atoms(), 3)
            else:
                forces = mol.get_forces(md=True)
        
            #Calculate and set momenta for the next step
            mol.set_momenta(mol.get_momenta() + 0.5 * dt * forces)
            write(file, mol, append = True)
    dyn.run(1)
    print_forces()
    write(file, mol, append = True)
    traj.write()

