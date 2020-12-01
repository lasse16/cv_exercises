class BatchFileReader(object):
    """Docstring for BatchFileReader. """

    def __init__(self):
        """TODO: to be defined. """
        pass

    def read(self, file_path):
        import pickle
        with open(file_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            return dict
