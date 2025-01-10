
import pickle, sys, os
import numpy as np
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
from fhe.client.gen_context import gen_contexts
logN = 16
logSlots = 11
maxLevelsRemaining = 3
levelBudget = [3, 3]
dnum = 4
rescaleTech = "FLEXIBLEAUTO"
dcrtBits=59
firstMod=60
approxModDepth=9

gen_contexts(
    logN=logN,
    logSlots=logSlots, # possible slots value of runtime ciphertext #todo: should be a list?
    maxLevelsRemaining=maxLevelsRemaining,
    levelBudget=levelBudget,
    dnum=dnum,
    dcrtBits=dcrtBits,
    firstMod=firstMod,
    approxModDepth=approxModDepth,
    rotate_index=[],
    secretKeyDist="UNIFORM_TERNARY",
    rescaleTech=rescaleTech,
    save_dir="data"
)

