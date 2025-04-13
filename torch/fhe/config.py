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

        self.SAVE_MIDDLE=SAVE_MIDDLE
        self.SAVE_END=SAVE_END
