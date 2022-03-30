""" Storing and representing attributes of datasets, including processed versions """

class Dataset:
    """ Superclass for storing data and attributes of datasets """

    def __init__(self, name, fpaths=None):
        self.name = name
        self.fpaths = fpaths

    def load(self):
        """ To be overriden by subclasses """
        pass

class Kennedy2020Dataset(Dataset):

    def load(self):
