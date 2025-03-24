class Config:
    def __init__(
        self,
        ON_DEMAND_LOAD=False,
        PTX_TWIN=False,
        CHECK_CIPHER=False,
        autoLoadAndSetConfig=False,
        AUTO_SYNC=False,
        COMPILER=False,
        COMPARE_WITH_OPENFHE=False,
        mode="release"
    ):
        self.ON_DEMAND_LOAD = ON_DEMAND_LOAD
        self.PTX_TWIN = PTX_TWIN
        self.CHECK_CIPHER = CHECK_CIPHER
        self.autoLoadAndSetConfig = autoLoadAndSetConfig
        self.mode = mode
        self.AUTO_SYNC = AUTO_SYNC
        self.COMPILER = COMPILER
        self.COMPARE_WITH_OPENFHE = COMPARE_WITH_OPENFHE
