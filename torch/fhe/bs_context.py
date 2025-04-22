import math
import torch
from .ciphertext import Plaintext, Cipher, PreEncodeValues

def get_item(item_name, content_map):
    if item_name in content_map:
        return content_map[item_name]
    return None
    
class CKKS_Boot_Params:
    def __init__(self, level_budget, layers_coll, layers_rem, num_rotations, baby_step, giant_step, num_rotations_rem,
                 baby_step_rem, giant_step_rem):
        self.level_budget = level_budget  # the level budget
        self.layers_coll = layers_coll  # the number of layers to collapse in one level
        self.layers_rem = layers_rem  # the number of layers remaining to be collapsed in one level to have exactly the number of levels specified in the level budget
        self.num_rotations = num_rotations  # the number of rotations in one level
        self.baby_step = baby_step  # the baby step in the baby-step giant-step strategy
        self.giant_step = giant_step  # the giant step in the baby-step giant-step strategy
        self.num_rotations_rem = num_rotations_rem  # the number of rotations in the remaining level
        self.baby_step_rem = baby_step_rem  # the baby step in the baby-step giant-step strategy for the remaining level
        self.giant_step_rem = giant_step_rem  # the giant step in the baby-step giant-step strategy for the remaining level
        self.total_elements = 9

class BsContext:
    def __init__(self, content_map):
        self.M = get_item("M", content_map)
        self.C2S_rot_in = get_item("C2S_rot_in", content_map)
        self.C2S_rot_out = get_item("C2S_rot_out", content_map)
        self.S2C_rot_in = get_item("S2C_rot_in", content_map)
        self.S2C_rot_out = get_item("S2C_rot_out", content_map)
        self.coefficients = get_item("coefficients", content_map)
        self.correctionFactor = get_item("correctionFactor", content_map)
        self.dim1 = get_item("dim1", content_map)
        self.k = get_item("k", content_map)
        self.m_U0PreFFT_ = get_item("m_U0PreFFT", content_map)
        self.m_U0hatTPreFFT_ = get_item("m_U0hatTPreFFT", content_map)
        self.slots = get_item("slots", content_map)
        self.paramsDec = get_item("paramsDec", content_map)
        self.paramsEnc = get_item("paramsEnc", content_map)

        if isinstance(self.m_U0PreFFT_[0][0], Plaintext):
            for i in range(len(self.m_U0hatTPreFFT_)):
                for j in range(len(self.m_U0hatTPreFFT_[i])):
                    self.m_U0hatTPreFFT_[i][j].cv = [torch.tensor(self.m_U0hatTPreFFT_[i][j].cv, dtype = torch.uint64)]
                    Cipher._id_counter = max(Cipher._id_counter, self.m_U0hatTPreFFT_[i][j].cipher_id)

            for i in range(len(self.m_U0PreFFT_)):
                for j in range(len(self.m_U0PreFFT_[i])):
                    self.m_U0PreFFT_[i][j].cv = [torch.tensor(self.m_U0PreFFT_[i][j].cv, dtype = torch.uint64)]
                    Cipher._id_counter = max(Cipher._id_counter, self.m_U0PreFFT_[i][j].cipher_id)
        elif isinstance(self.m_U0PreFFT_[0][0], PreEncodeValues):
            for i in range(len(self.m_U0hatTPreFFT_)):
                for j in range(len(self.m_U0hatTPreFFT_[i])):
                    self.m_U0hatTPreFFT_[i][j].encoded_values = torch.tensor(self.m_U0hatTPreFFT_[i][j].encoded_values)

            for i in range(len(self.m_U0PreFFT_)):
                for j in range(len(self.m_U0PreFFT_[i])):
                    self.m_U0PreFFT_[i][j].encoded_values = torch.tensor(self.m_U0PreFFT_[i][j].encoded_values)
        
        self.BS_FFT = {}
        for i in range(len(self.m_U0hatTPreFFT_)):
            for j in range(len(self.m_U0hatTPreFFT_[i])):
                self.BS_FFT["{}_{}_{}".format("C2S", i, j)] = self.m_U0hatTPreFFT_[i][j]

        for i in range(len(self.m_U0PreFFT_)):
            for j in range(len(self.m_U0PreFFT_[i])):
                self.BS_FFT["{}_{}_{}".format("S2C", i, j)] = self.m_U0PreFFT_[i][j]


    # Placeholder function for SelectLayers, which needs to be defined as per the logic in your system.
    def SelectLayers(self, logBsSlots, budget):
        layers = math.ceil(logBsSlots / budget)
        rows = logBsSlots // layers
        rem = logBsSlots % layers

        dim = rows
        if rem != 0:
            dim = rows + 1

        # The above choice ensures dim <= budget
        if dim < budget:
            layers -= 1
            rows = logBsSlots // layers
            rem = logBsSlots - rows * layers
            dim = rows

            if rem != 0:
                dim = rows + 1

            # The above choice ensures dim >= budget
            while dim != budget:
                rows -= 1
                rem = logBsSlots - rows * layers
                dim = rows
                if rem != 0:
                    dim = rows + 1

        return [layers, rows, rem]

    def GetCollapsedFFTParams(self, slots, levelBudget, dim1):
        dims = self.SelectLayers(int(math.log2(slots)), levelBudget)
        layersCollapse = dims[0]
        remCollapse = dims[2]

        flagRem = 1 if remCollapse != 0 else 0

        numRotations = (1 << (layersCollapse + 1)) - 1
        numRotationsRem = (1 << (remCollapse + 1)) - 1

        # Computing the baby-step b and the giant-step g for the collapsed layers for decoding.
        if dim1 == 0 or dim1 > numRotations:
            if numRotations > 7:
                g = 1 << (int(layersCollapse / 2) + 2)
            else:
                g = 1 << (int(layersCollapse / 2) + 1)
        else:
            g = dim1

        b = (numRotations + 1) // g
        bRem = 0
        gRem = 0

        if flagRem:
            if numRotationsRem > 7:
                gRem = 1 << (int(remCollapse / 2) + 2)
            else:
                gRem = 1 << (int(remCollapse / 2) + 1)
            bRem = (numRotationsRem + 1) // gRem

        # If this return statement changes then CKKS_BOOT_PARAMS should be altered as well
        return CKKS_Boot_Params(int(levelBudget), layersCollapse, remCollapse, int(numRotations), b, g,
                                int(numRotationsRem), bRem, gRem)

    def to_cuda(self):
        for key, value in self.BS_FFT.items():
            if isinstance(value, Plaintext):
                self.BS_FFT[key].cv = [self.BS_FFT[key].cv[0].cuda()]
            elif isinstance(value, PreEncodeValues):
                self.BS_FFT[key].encoded_values = self.BS_FFT[key].encoded_values.cuda()
            else:
                raise TypeError("Unsupported type for BS_FFT value: {}".format(type(value)))
    
    # def encode_FFT(self, x):
    #     self.to_cuda()
    #     if isinstance(self.m_U0PreFFT[0][0], Plaintext):
    #         for i in range(len(self.m_U0hatTPreFFT)):
    #             for j in range(len(self.m_U0hatTPreFFT[i])):
    #                 self.m_U0hatTPreFFT[i][j] = encode(
    #         for i in range(len(self.m_U0PreFFT)):
    #             for j in range(len(self.m_U0PreFFT[i])):
    #                 self.m_U0PreFFT[i][j].cv = [self.m_U0PreFFT[i][j].cv[0].cuda()]