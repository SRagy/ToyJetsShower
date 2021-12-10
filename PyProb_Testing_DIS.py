# This is PyProb_Testing.ipynb converted to a script, fixing some errors.
# I've also reduced num_traces so that this script runs quickly.
# (See comments next to num_traces below for original values.)
import pyprob
import numpy as np
import torch

from pyprob.dis import ModelDIS
from showerSim import invMass_ginkgo

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as mpl_cm
plt.ion()

import sklearn as skl
from sklearn.linear_model import LinearRegression

# Example observations
obs_leaves = torch.tensor([[1.3582e+00, 8.4035e-01, 7.1867e-01, 7.7719e-01],
                           [1.0977e+01, 6.8858e+00, 5.8830e+00, 6.1987e+00],
                           [1.0614e+00, 5.3323e-01, 5.0717e-01, 7.5125e-01],
                           [1.1801e+01, 5.8540e+00, 7.7156e+00, 6.7371e+00],
                           [5.6925e+01, 2.9145e+01, 3.6351e+01, 3.2704e+01],
                           [1.0741e+01, 6.5786e+00, 5.9341e+00, 6.0709e+00],
                           [2.4267e+01, 1.5105e+01, 1.2999e+01, 1.3848e+01],
                           [2.3929e+00, 1.4333e+00, 1.3158e+00, 1.3912e+00],
                           [9.7744e+00, 5.2676e+00, 5.9366e+00, 5.6997e+00],
                           [2.9838e+00, 1.5489e+00, 1.8553e+00, 1.7464e+00],
                           [2.5505e-01, 1.3109e-01, 1.4855e-01, 1.6056e-01],
                           [1.7481e+00, 8.9971e-01, 9.9951e-01, 1.0914e+00],
                           [2.2055e-01, 1.0889e-01, 1.4514e-01, 1.1316e-01],
                           [2.0188e+00, 1.0690e+00, 1.2583e+00, 1.1238e+00],
                           [9.8853e-02, 5.6402e-02, 6.0347e-02, 5.3639e-02],
                           [1.0206e+00, 7.0193e-01, 5.2862e-01, 4.9077e-01],
                           [6.6845e+00, 3.3772e+00, 4.1681e+00, 3.9800e+00],
                           [4.4337e+00, 2.2615e+00, 2.8427e+00, 2.5419e+00],
                           [5.4864e+00, 3.4506e+00, 2.8262e+00, 3.1888e+00],
                           [8.9406e+00, 5.4560e+00, 4.6372e+00, 5.3516e+00],
                           [4.1603e+00, 2.6097e+00, 2.4249e+00, 2.1478e+00],
                           [1.1141e+00, 7.0173e-01, 6.8822e-01, 5.2415e-01],
                           [5.4801e+01, 3.2719e+01, 3.0630e+01, 3.1534e+01],
                           [6.4460e+01, 3.8405e+01, 3.6199e+01, 3.7010e+01],
                           [4.4181e+01, 2.6379e+01, 2.4852e+01, 2.5268e+01],
                           [1.3450e+01, 7.9831e+00, 7.5711e+00, 7.7357e+00],
                           [3.3631e+00, 2.3526e+00, 1.7801e+00, 1.5953e+00],
                           [2.7995e+00, 1.8861e+00, 1.4701e+00, 1.4284e+00],
                           [2.7899e+00, 1.7649e+00, 1.3742e+00, 1.6608e+00],
                           [1.3081e+00, 8.7535e-01, 5.6566e-01, 7.8314e-01],
                           [1.8323e+00, 1.2282e+00, 9.1912e-01, 9.7776e-01],
                           [2.6257e-01, 1.6143e-01, 1.4201e-01, 1.5037e-01],
                           [3.1709e-01, 2.2294e-01, 1.5466e-01, 1.6379e-01],
                           [9.3848e-01, 6.4072e-01, 4.8137e-01, 4.7873e-01],
                           [2.7002e+00, 1.7782e+00, 1.2173e+00, 1.6180e+00],
                           [4.3433e+00, 2.6527e+00, 2.3221e+00, 2.5366e+00],
                           [6.4606e+00, 3.2403e+00, 3.9309e+00, 3.9695e+00],
                           [7.8597e+00, 4.1080e+00, 4.7234e+00, 4.7475e+00],
                           [1.7768e+00, 9.7552e-01, 1.1043e+00, 9.9237e-01],
                           [1.5271e+00, 9.6589e-01, 8.6833e-01, 7.9099e-01],
                           [7.3495e+00, 3.7483e+00, 4.5073e+00, 4.4234e+00],
                           [3.5146e-01, 1.8693e-01, 2.1358e-01, 2.0702e-01],
                           [1.2103e+00, 5.6245e-01, 8.0026e-01, 7.1008e-01],
                           [1.2219e+00, 5.0675e-01, 8.2541e-01, 7.4321e-01],
                           [2.4190e+00, 1.1670e+00, 1.4529e+00, 1.5403e+00],
                           [4.9373e+00, 2.4143e+00, 2.8918e+00, 3.1834e+00]], dtype=torch.float64)

# Define initial conditions for the simulator

jetM = 80. # parent mass -> W
jetdir = np.array([1,1,1]) # direction
jetP = 400. # magnitude
jetvec = jetP * jetdir / np.linalg.norm(jetdir)

jet4vec = np.concatenate(([np.sqrt(jetP**2 + jetM**2)], jetvec))

# Define a function that takes (self, jet) and outputs True for the condition we want
def dummy_bernoulli(self, jet):
    return True

def get_subjet_pT(jet, side="left"):
    if side == "left":
        subjet_left_4vec = jet["content"][jet["tree"][0][0]]
        subjet_left_pT = np.sqrt(subjet_left_4vec[1]**2 + subjet_left_4vec[2]**2)
        return subjet_left_pT
    elif side == "right":
        subjet_right_4vec = jet["content"][jet["tree"][0][1]]
        subjet_right_pT= np.sqrt(subjet_right_4vec[1]**2 + subjet_right_4vec[2]**2)
        return subjet_right_pT
    return None

def subjet_pT_cut(self, jet):
    subjet_left_4vec = jet["content"][jet["tree"][0][0]]
    subjet_right_4vec = jet["content"][jet["tree"][0][1]]
    subjet_left_pT = np.sqrt(subjet_left_4vec[1]**2 + subjet_left_4vec[2]**2)
    subjet_right_pT= np.sqrt(subjet_right_4vec[1]**2 + subjet_right_4vec[2]**2)
    #return (275 <= subjet_left_pT <= 400) or (275 <= subjet_right_pT <= 400)
    return (subjet_left_pT <= 40) and (270 <= subjet_right_pT)

# Make instance of the simulator

class SimulatorModelDIS(invMass_ginkgo.SimulatorModel, ModelDIS):
    pass

simulator = SimulatorModelDIS(rate=[3, 1.5], # exponential dsitribution rate
                              jet_p=jet4vec,  # parent particle 4-vector
                              pt_cut=10.,  # minimum pT for resulting jet
                              Delta_0=torch.tensor(jetM**2),  # parent particle mass squared -> needs tensor
                              M_hard=jetM,  # parent particle mass
                              minLeaves=30,  # minimum number of jet constituents
                              maxLeaves=40,  # maximum " "
                              bool_func=dummy_bernoulli,
                              suppress_output=True,
                              obs_leaves=obs_leaves)

jet = simulator()  # Make sure the forward pass works

# Generate traces for the prior distribution

simulator.train(
    importance_sample_size=100, #SMALL SIZE FOR TESTING!
    proposal_mixture_components=3,
    observe_embeddings={'bool_func': {'dim': 1, 'depth': 1}}

) 
