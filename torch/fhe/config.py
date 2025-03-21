class Config:
    def __init__(
        self,
        COMPARE_WITH_OPENFHE=False,
        ON_DEMAND_LOAD=False,
        PTX_TWIN=False,
        CHECK_LIBM=False,
        autoLoadAndSetConfig=False,
        mode="release"
    ):
        self.COMPARE_WITH_OPENFHE = COMPARE_WITH_OPENFHE
        self.ON_DEMAND_LOAD = ON_DEMAND_LOAD
        self.PTX_TWIN = PTX_TWIN
        self.CHECK_LIBM = CHECK_LIBM
        self.autoLoadAndSetConfig = autoLoadAndSetConfig
        self.mode = mode