class Config:
    def __init__(
        self,
        AUTO_LOAD_KEYS=True,
        PTX_TWIN=False,
        CHECK_CIPHER=False,
        AUTO_SYNC=False,
        COMPILER=False,
        COMPARE_WITH_OPENFHE=False,
        TIME_OPS=False,
        COUNT_OPS=False,
        ENCODE_BS_FFT=True,
        MAX_RNS_LIMBS_BY_ROT_EVK={},
        SAVE_MIDDLE=False,
        SAVE_END=False
    ):
        self.AUTO_LOAD_KEYS = AUTO_LOAD_KEYS
        self.PTX_TWIN = PTX_TWIN
        self.CHECK_CIPHER = CHECK_CIPHER
        self.AUTO_SYNC = AUTO_SYNC
        self.COMPILER = COMPILER
        self.COMPARE_WITH_OPENFHE = COMPARE_WITH_OPENFHE
        self.TIME_OPS=TIME_OPS
        self.COUNT_OPS=COUNT_OPS
        self.ENCODE_BS_FFT=ENCODE_BS_FFT
        self.MAX_RNS_LIMBS_BY_ROT_EVK = MAX_RNS_LIMBS_BY_ROT_EVK

        self.SAVE_MIDDLE=SAVE_MIDDLE
        self.SAVE_END=SAVE_END

    def label(self):
        val = 0
        if self.COMPARE_WITH_OPENFHE:
            val += 1
        if self.ENCODE_BS_FFT:
            val += 2
        if self.MAX_RNS_LIMBS_BY_ROT_EVK:
            val += 4
            import warnings
            warnings.warn("\033[31mMAX_RNS_LIMBS_BY_ROT_EVK is set and may lead to potential errors.\033[0m")

        return val
