class BatchFileWriter(object):
    def write(self, obj, path):
        import pickle
        import os
        dirname = os.path.dirname(path)
        os.makedirs(dirname,exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(obj, file)
