import itertools, subprocess, tempfile, os


logN_cand = list(range(14, 17))
logSlots_cand = list(range(5, 17, 3))
maxLevelsRemaining_cand = [3, 5, 7]
levelBudget_cand = [[2, 2], [4, 4]]
dnum_cand = [1, 3, 5]
openfhe_context = None

i = 0
for logN, logSlots, maxLevelsRemaining, levelBudget, dnum in itertools.product(logN_cand, logSlots_cand, maxLevelsRemaining_cand, levelBudget_cand, dnum_cand):
    if logSlots >= logN - 2:
        continue
    try:
        print(i, ": ", logN, logSlots, maxLevelsRemaining, levelBudget, dnum)
        code_string = """
import pickle
import torch.fhe.bootstrapping as bstest
logN = {0}
logSlots = {1}
maxLevelsRemaining = {2}
levelBudget = [{3}, {4}]
dnum = {5}
path_ctx = "torch/fhe/data/context_{0}_{1}_{2}_{3}_{4}_{5}.pkl"
path_io = "torch/fhe/data/groundtruth_{0}_{1}_{2}_{3}_{4}_{5}.pkl"
cryptoContext, openfhe_context, cipher, result = bstest.BootstrapTest_N65536L26lB44(logN, logSlots, maxLevelsRemaining, levelBudget, dnum, save_path=path_ctx)
input = [cipher.cv[0].tolist(), cipher.cv[1].tolist()]
output = [result.cv[0].tolist(), result.cv[1].tolist()]
with open(path_io, "wb") as file:
    pickle.dump((input, output), file)
""".format(logN, logSlots, maxLevelsRemaining, levelBudget[0], levelBudget[1], dnum)
        
        # Create a temporary file to store the code
        with open("temp_file.py", "w") as temp_file:
            print(code_string, file=temp_file)

        try:
            # Execute the temporary file as a separate process
            command = "bash -c 'source ~/.bashrc && python3 temp_file.py'"
            process = subprocess.Popen(command, shell=True)
            process.wait()

            print("Process finished with exit code:", process.returncode)
        finally:
            # Clean up the temporary file
            os.remove("temp_file.py")


    except Exception as e:
        print(e)
        continue