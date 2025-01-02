import pickle, sys, os
import numpy as np
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
import torch
import torch.fhe.bootstrapping as BS


#find all context in the directory
all_correct = True
path = "data/"
for context_file in os.listdir(path):
    if context_file.endswith(".pkl") and "context" in context_file:
        groundtruth_file = context_file.replace("context", "groundtruth")
    else:
        continue
    with open(path+context_file, "rb") as file:
        cryptoContext_byte, openfhe_byte = pickle.load(file)
    cryptoContext = BS.Context.Deserialize(cryptoContext_byte)
    openfhe_context = BS.client.OpenFHEContext.Deserialize(openfhe_byte)

    with open(path+groundtruth_file, "rb") as file:
        input, output = pickle.load(file)

    dim1 = [0, 0]
    cryptoContext.BsContext = BS.BsContext(
        cryptoContext,
        cryptoContext.levelBudget,
        dim1,
        cryptoContext.slots,
        0,
        cryptoContext.rescaleTech,
        cryptoContext.secretKeyDist,
    )

    BS.eval_bootstrap_setup(
        cryptoContext, cryptoContext.levelBudget, dim1, cryptoContext.slots, 0
    )

    # note: do not support FLEXIBLEAUTOEXT　currently,
    # noise_deg=1 for "FLEXIBLEAUTO" and "FIXEDMANUAL", noise_deg=2 for "FLEXIBLEAUTOEXT"
    # todo: generalize the setting
    input.cv = [torch.tensor(elem, device="cuda", dtype=torch.uint64) for elem in input.cv]
    result = BS.eval_bootstrap(input, L0=cryptoContext.L, slots=cryptoContext.slots, cryptoContext=cryptoContext)

    res_cv0 = result.cv[0].cpu().numpy().reshape(-1)
    res_cv1 = result.cv[1].cpu().numpy().reshape(-1)
    groundtruth0 = np.array(output.cv[0], dtype=np.uint64).reshape(-1)
    groundtruth1 = np.array(output.cv[1], dtype=np.uint64).reshape(-1)

    x = np.array([0.111111111 * (i & 0xFF) for i in range(cryptoContext.slots)])
    after_boot = openfhe_context.decrypt(result)
    after_boot = after_boot.cpu().numpy().reshape(-1)
    x = np.array(x, dtype=np.float32).reshape(-1)

    max_err = np.max(np.abs(after_boot - x))
    avg_err = np.mean(np.abs(after_boot - x))

    print("Test case:", context_file.split("/")[-1])
    print("Max error:", max_err, "Average error:", avg_err)

    if np.equal(res_cv0, groundtruth0).all() and np.equal(res_cv1, groundtruth1).all():
        print("Test passed!")
    else:
        all_correct = False
        print("Test failed!")

if all_correct:
    print("All test cases passed!")