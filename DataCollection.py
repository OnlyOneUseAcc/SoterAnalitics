import numpy as np


class DataCollection:
    def __init__(self, size):
        self.size = size
        self.__data = np.array([np.zeros(4)] * size)
        self.current_count = 0
        self.start = 0

    def get_data(self):
        if self.current_count < self.size:
            raise AttributeError(f"Unable to get data, "
                                 f"collection has only {self.current_count} objects, need {self.size}")
        return np.concatenate([self.__data[self.start:], self.__data[:self.start]])

    def put(self, data_object):
        self.__data[self.start] = data_object
        self.start += 1
        self.current_count += 1

        if self.start == self.size - 1:
            self.start = 0
