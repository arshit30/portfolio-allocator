import unittest
import numpy as np

def weight_constraint(weights):
    range=np.arange(0.98,1.02,0.001)
    assert np.round(sum(weights),3) in range

def strategies_inscope(strat):
    strats=['MSR','EW','GMV','ERW']
    assert strat in strats
