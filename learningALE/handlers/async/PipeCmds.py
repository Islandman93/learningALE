from enum import Enum


class PipeCmds(Enum):
    # host commands
    Start = 1
    End = 2
    HostSendingGlobalParameters = 3

    # client commands
    ClientSendingGradientsSteps = 4
    ClientSendingStats = 5