import numpy as np


class SafeArray(np.ndarray):
    @staticmethod
    def wrap(arr):
        return arr.view(SafeArray)
        
    def __getitem__( self, index ):
        if type(index) is int:
            nidx = index % self.shape[0]
        elif type(index) is tuple:
            t = []
            for x,size in zip(index,self.shape):
                if type(x) is int:
                    t.append(x % size)
                else:
                    t.append(x)
            nidx = tuple(t)
        else:
            print 'unknown type:', type(index)
            assert False
        return super(SafeArray,self).__getitem__(nidx)
