import math
import torch


def get_item(item_name, content_map):
    if item_name in content_map:
        return content_map[item_name]
    return None


class CKKS_Boot_Params:
    def __init__(
        self,
        level_budget,
        layers_coll,
        layers_rem,
        num_rotations,
        baby_step,
        giant_step,
        num_rotations_rem,
        baby_step_rem,
        giant_step_rem,
    ):
        self.level_budget = level_budget
        self.layers_coll = layers_coll
        self.layers_rem = layers_rem
        self.num_rotations = num_rotations
        self.baby_step = baby_step
        self.giant_step = giant_step
        self.num_rotations_rem = num_rotations_rem
        self.baby_step_rem = baby_step_rem
        self.giant_step_rem = giant_step_rem


class BsContext:
    def __init__(self, content_map):
        self.M = get_item("M", content_map)
        self.QmuplusPmu_map = get_item("QmuplusPmu_map", content_map)
        self.QplusP_map = get_item("QplusP_map", content_map)
        self.C2S_rot_in = get_item("C2S_rot_in", content_map)
        self.C2S_rot_out = get_item("C2S_rot_out", content_map)
        self.S2C_rot_in = get_item("S2C_rot_in", content_map)
        self.S2C_rot_out = get_item("S2C_rot_out", content_map)
        self.coefficients = get_item("coefficients", content_map)
        self.correctionFactor = get_item("correctionFactor", content_map)
        self.dim1 = get_item("dim1", content_map)
        self.k = get_item("k", content_map)
        self.m_U0PreFFT_dim = get_item("m_U0PreFFT_dim", content_map)
        self.m_U0PreFFT_limbs = get_item("m_U0PreFFT_limbs", content_map)
        self.m_U0PreFFT_mx = get_item("m_U0PreFFT_mx", content_map)
        self.m_U0hatTPreFFT_dim = get_item("m_U0hatTPreFFT_dim", content_map)
        self.m_U0hatTPreFFT_limbs = get_item("m_U0hatTPreFFT_limbs", content_map)
        self.m_U0hatTPreFFT_mx = get_item("m_U0hatTPreFFT_mx", content_map)
        self.m_U0Pre = get_item("m_U0Pre", content_map)
        self.m_U0PreFFT = get_item("m_U0PreFFT", content_map)
        self.m_U0hatTPre = get_item("m_U0hatTPre", content_map)
        self.m_U0hatTPreFFT = get_item("m_U0hatTPreFFT", content_map)
        self.slots = get_item("slots", content_map)
        self.paramsDec = get_item("paramsDec", content_map)
        self.paramsEnc = get_item("paramsEnc", content_map)

        for key, value in self.QplusP_map.items():
            self.QplusP_map[key] = torch.tensor(value, dtype=torch.uint64)
        for key, value in self.QmuplusPmu_map.items():
            self.QmuplusPmu_map[key] = torch.tensor(value, dtype=torch.uint64)

        for i in range(len(self.m_U0hatTPreFFT)):
            for j in range(len(self.m_U0hatTPreFFT[i])):
                self.m_U0hatTPreFFT[i][j].cv = torch.tensor(
                    self.m_U0hatTPreFFT[i][j].cv, dtype=torch.uint64
                )

        for i in range(len(self.m_U0PreFFT)):
            for j in range(len(self.m_U0PreFFT[i])):
                self.m_U0PreFFT[i][j].cv = torch.tensor(
                    self.m_U0PreFFT[i][j].cv, dtype=torch.uint64
                )

    def SelectLayers(self, logBsSlots, budget):
        layers = math.ceil(logBsSlots / budget)
        rows = logBsSlots // layers
        rem = logBsSlots % layers

        dim = rows
        if rem != 0:
            dim = rows + 1
        if dim < budget:
            layers -= 1
            rows = logBsSlots // layers
            rem = logBsSlots - rows * layers
            dim = rows

            if rem != 0:
                dim = rows + 1

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
        return CKKS_Boot_Params(
            int(levelBudget),
            layersCollapse,
            remCollapse,
            int(numRotations),
            b,
            g,
            int(numRotationsRem),
            bRem,
            gRem,
        )

    def to_cuda(self):
        for key, value in self.QplusP_map.items():
            self.QplusP_map[key] = value.cuda()
        for key, value in self.QmuplusPmu_map.items():
            self.QmuplusPmu_map[key] = value.cuda()

        for i in range(len(self.m_U0hatTPreFFT)):
            for j in range(len(self.m_U0hatTPreFFT[i])):
                self.m_U0hatTPreFFT[i][j].cv = self.m_U0hatTPreFFT[i][j].cv.cuda()

        for i in range(len(self.m_U0PreFFT)):
            for j in range(len(self.m_U0PreFFT[i])):
                self.m_U0PreFFT[i][j].cv = self.m_U0PreFFT[i][j].cv.cuda()
