"""
simple test script to measure run-time across two devices.

note that this won't work if you don't have 2 openCL drivers installed.
"""

from py2opencl import Py2OpenCL, F
import numpy
import numpy.random
import time



for X,Y in ((7,7), (10,10), (22,22), (31,31), (100,100), (223,223), (316,316)):
    print X*Y, 'cells'

    input = numpy.random.uniform( 0, 2 * numpy.pi, size=(X,Y) ).astype(numpy.dtype('float32'))

    iterate = Py2OpenCL( lambda x: F.sqrt( F.sin(x) ), user_dev_selection=['0'] )  # nvidia
    iterate.bind( input )

    for _ in range(3):
        before = time.time()
        for i in range(int(1e6)):
            _ = iterate.apply( input )

        print '0:', time.time() - before


    iterate = Py2OpenCL( lambda x: F.sqrt( F.sin(x) ), user_dev_selection=['1'] ) # i7
    iterate.bind( input )

    for _ in range(3):
        before = time.time()
        for i in range(int(1e6)):
            _ = iterate.apply( input )
        print '1:', time.time() - before
    print
