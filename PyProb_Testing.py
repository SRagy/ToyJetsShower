# This is PyProb_Testing.ipynb converted to a script, fixing some errors.
# I've also reduced num_traces so that this script runs quickly.
# (See comments next to num_traces below for original values.)
import pyprob
import numpy as np
import torch

from showerSim import invMass_ginkgo

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as mpl_cm
plt.ion()

import sklearn as skl
from sklearn.linear_model import LinearRegression

# Define initial conditions for the simulator

jetM = 80. # parent mass -> W
jetdir = np.array([1,1,1]) # direction
jetP = 400. # magnitude
jetvec = jetP * jetdir / np.linalg.norm(jetdir)

jet4vec = np.concatenate(([np.sqrt(jetP**2 + jetM**2)], jetvec))

# Define a function that takes (self, jet) and outputs True for the condition we want

# Condition on the number of leaves
def num_leaves_cut(self, jet):
    return len(jet["leaves"]) >= 27

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

simulator = invMass_ginkgo.SimulatorModel(rate=[3, 1.5], # exponential dsitribution rate
                                     jet_p=jet4vec,  # parent particle 4-vector
                                     pt_cut=10.,  # minimum pT for resulting jet
                                     Delta_0=torch.tensor(jetM**2),  # parent particle mass squared -> needs tensor
                                     M_hard=jetM,  # parent particle mass
                                     minLeaves=30,  # minimum number of jet constituents
                                     maxLeaves=40,  # maximum " "
                                     bool_func=num_leaves_cut,
                                     suppress_output=True)

jet = simulator()  # Make sure the forward pass works

# Generate traces for the prior distribution

prior = simulator.prior(num_traces=500) # Was 5000 in Matthew's script

# Train the NN for inference compilation

simulator.learn_inference_network(
    num_traces=500, # Was 5000 in Matthew's script
    proposal_mixture_components=3,
    observe_embeddings={'bool_func': {'dim': 32, 'depth': 3}}
)

# Generate traces for the posterior distribution

posterior = simulator.posterior(num_traces=500, # Was 15*5000 in Matthew's script
                                inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK,
                                observe={"bool_func": 1})

# Investigate latent variables

index = 1

exp_prior_dist = prior.filter(lambda t: t.named_variables.get('L_decay' + str(index) + 'True') is not None)
exp_post_dist = posterior.filter(lambda t: t.named_variables.get('L_decay' + str(index) + 'True') is not None)

exp_prior_dist = exp_prior_dist.map(lambda t: t['L_decay' + str(index) + 'True'])
exp_post_dist = exp_post_dist.map(lambda t: t['L_decay' + str(index) + 'True'])

w = exp_prior_dist.weights.numpy()
print("Prior effective sample size:", w.sum()**2 / (w**2).sum())

w = exp_post_dist.weights.numpy()
print("Posterior effective sample size:", w.sum()**2 / (w**2).sum())

bins = np.linspace(0,1,35)

fig = plt.figure()
fig.set_size_inches(8,8)
ax = fig.add_subplot(111)
bins = np.linspace(0,1,35)
c1,_,_ = ax.hist([x.item() for x in exp_prior_dist.values],
                 weights=exp_prior_dist.weights.numpy(),
                 bins=bins, alpha = 0.5, label='prior (node index '+str(index)+')', density=True);
c2,_,_ = ax.hist([x.item() for x in exp_post_dist.values],
                 weights=exp_post_dist.weights.numpy(),
                 bins=bins, alpha = 0.5, label = 'posterior (node index '+str(index)+')', density=True);
ax.legend()
ax.set_title("Ginkgo + PyProb Test Conditioned on (left_subtree_pT <= 40 AND right_subtree_pT >= 270)")
ax.set_xlabel('Truncated Exponential Samples')
ax.set_ylabel("Normalized Bin Count")

wait = input("Press enter to terminate")
