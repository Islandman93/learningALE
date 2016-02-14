from enum import Enum


class PipeCmds(Enum):
    # host commands
    Start = 1
    End = 2
    HostSendingGlobalParameters = 3
    HostSendingGlobalTarget = 4

    # client commands
    ClientSendingGradients = 5
    ClientSendingSteps = 6
