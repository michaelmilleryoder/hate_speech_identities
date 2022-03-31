""" Storing and representing attributes of datasets, including processed versions """

class Dataset:
    """ Superclass for storing data and attributes of datasets """

    def __init__(self, name, load_args=None):
        """ Args:
                name: dataset name
                load_args: list of arguments for loading the datasets (like file paths)
        """
        self.name = name
        self.load_args = load_args
        self.data = None
