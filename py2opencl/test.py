"""
simple test of py2opencl
"""

import numpy as np
import time
from .driver import Py2OpenCL
from . import F


def main():

    lmb = lambda x: -x if x < 0.5 else F.sin(x)
    a = Py2OpenCL( lmb )

    s = Py2OpenCL( lambda x: x ).map( np.random.rand(1000).astype(np.float32) )

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
        print r2 - res

    print
    two = Py2OpenCL( lambda x, y: x * y )
    for size in (1e4, 1e5, 1e6, 1e7):
        a, b = np.random.rand(int(1e7)).astype(np.float32), np.random.rand(int(1e7)).astype(np.float32)

        before = time.time()
        res = two.map( a, b )
        print "Simple multiplication - OpenCL (size=1e%s):" % int(math.log10(size)), time.time() - before
        before = time.time()
        r2 = a * b
        print "Simple multiplication - numpy (size=1e%s):" % int(math.log10(size)), time.time() - before
        print r2 - res

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
        print r2 - res


if __name__ == '__main__':
   main()
