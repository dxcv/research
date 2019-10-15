class DataSet(dict):

    def __getattr__(self, item):
        if item in self.keys():
            return self.get(item)
        raise AttributeError(item)

    def __dir__(self):
        return self.keys()

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

