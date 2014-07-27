"""
simple test of py2opencl
"""

import numpy as np
import time
from ..driver import Py2OpenCL
from .. import F


def main():

    

    lmb = lambda x: -x if x < 0.5 else F.sin(x)
    #arr = (1000 * np.random.rand(1000)).astype(np.int32)
    arr = np.random.rand(10000000).astype(np.float32)

    print '-- float: -> int:', Py2OpenCL( lambda x: int(x) ).map( 1000 * arr )

    print '-- int -> float:', Py2OpenCL( lambda x: float(x) ).map( (1000 * arr).astype('int32') )

    before = time.time()
    a = Py2OpenCL( lmb ).map( arr )
    print "sine - OpenCL: for %d elements, took" % len(a), time.time() - before
    # b = lmb( arr )  # conditionals don't work this way in Numpy
    before = time.time()
    b = np.where( arr < 0.5, -arr, np.sin(arr) )
    print "sine - numpy: for %d elements, took" % len(a), time.time() - before
    print "max delta: %.2e\n" % np.max(a - b)

    before = time.time()
    a = Py2OpenCL( lambda x: F.atanpi(x) ).map( arr )
    print "arctan(x) / pi - openCL: for %d elements, took" % len(a), time.time() - before
    before = time.time()
    b = (lambda x: F.atanpi(x) / np.pi)( arr )
    print "arctan(x) / pi - numpy: for %d elements, took" % len(a), time.time() - before

    for n in (100, 10000, 1000000, 10000000):
        rnd = np.random.rand(n).astype(np.float32)

        before = time.time()
        res_np = Py2OpenCL( lmb ).map( rnd )
        print "Simple tertiary operator case - OpenCL: for %d elements, took" % len(rnd), time.time() - before

        before = time.time()
        py = map( lmb, rnd )
        print "Simple tertiary operator case - Python: for %d elements, took" % len(rnd), time.time() - before

    import math
    two = Py2OpenCL( lambda x, y: x + y )
    for size in (1e4, 1e5, 1e6, 1e7):
        a, b = np.random.rand(int(1e7)).astype(np.float32), np.random.rand(int(1e7)).astype(np.float32)

        before = time.time()
        res = two.map( a, b )
        print "Simple sum - OpenCL (size=1e%s):" % math.log10(size), time.time() - before
        before = time.time()
        r2 = a + b
        print "Simple sum - numpy (size=1e%s):" % math.log10(size), time.time() - before
        print "max delta: %.2e\n" % np.max(r2 - res)

    two = Py2OpenCL( lambda x, y: x * y )
    for size in (1e4, 1e5, 1e6, 1e7):
        a, b = np.random.rand(int(1e7)).astype(np.float32), np.random.rand(int(1e7)).astype(np.float32)

        before = time.time()
        res = two.map( a, b )
        print "Simple multiplication - OpenCL (size=1e%s):" % int(math.log10(size)), time.time() - before
        before = time.time()
        r2 = a * b
        print "Simple multiplication - numpy (size=1e%s):" % int(math.log10(size)), time.time() - before
        print "max delta: %.2e\n" % np.max(r2 - res)

    print
    two = Py2OpenCL( lambda x, y: x ** y )
    for size in (1e4, 1e5, 1e6, 1e7):
        a, b = np.random.rand(int(1e7)).astype(np.float32), np.random.rand(int(1e7)).astype(np.float32)

        before = time.time()
        res = two.map( a, b )
        print "Simple exponents - OpenCL (size=1e%s):" % int(math.log10(size)), time.time() - before
        before = time.time()
        r2 = a ** b
        print "Simple exponents - numpy (size=1e%s):" % int(math.log10(size)), time.time() - before
        print "max delta: %.2e\n" % np.max(r2 - res)


if __name__ == '__main__':
   main()
