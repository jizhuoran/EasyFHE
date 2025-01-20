import torch.fhe.bootstrapping as bstest
import torch.fhe.resnet.resnet20 as res
import pickle
import numpy as np
# bstest.run_test_cases()
# bstest.BootstrapTest_N65536L26lB44()

# bstest.BootstrapTest_slots_list_example()
# bstest.BootstrapTest_test_case()

# with open('torch/fhe/resnet/weights.pkl', 'rb') as f:
#     weight_map = pickle.load(f)

# for key, _ in weight_map.items():
#     print(key)
#     print(weight_map[key])
#     print(weight_map[key].shape)
#     print()


res.resnet20( )
