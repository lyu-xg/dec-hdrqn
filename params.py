class CNN_Feature_Extractors(object):
    
    """ ConvNet Hyperparameters """
    def __init__(self):
        self.p        = 'VALID' #Padding
        self.format   = 'NCHW'
        self.outdim   = [32,64]
        self.kernels  = [4,2]
        self.stride   = [2,1]
        self.fc       = 1024
        self.max_in   = 255.0
