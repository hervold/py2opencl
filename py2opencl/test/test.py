"""
simple test of py2opencl
"""

import numpy as np
import time
import os.path
from PIL import Image

from ..driver import Py2OpenCL
from ..convert import function_to_kernel
from ..compat import SafeArray
from .. import F

from . import __file__ as test_directory


def avg_img_files( src_path, dest_path ):
    img = Image.open( src_path ).convert('RGB') # 3 uint8's per pixel
    img_arr = np.array(img)
    result = avg_img( img_arr )
    Image.fromarray( result, 'RGB').save( dest_path )



def avg_img( img_arr, purepy=False, user_dev_selection=None ):
    def avg( x, y, z, dest, src ):
        right = src[ x+1, y, z ]
        left = src[ x-1, y, z ]
        up = src[ x, y-1, z ]
        down = src[ x, y+1, z ]
        dest[x,y,z] = (right / 4) + (left / 4) + (up / 4) + (down / 4)

    if purepy:
        dest = np.empty_like( img_arr )

        x,y,z = img_arr.shape
        for i in range( x ):
            for j in range( y ):
                for k in range( z ):
                    avg( i, j, k, dest, SafeArray.wrap(img_arr) )
    else:
        x = Py2OpenCL( avg, user_dev_selection=user_dev_selection )
        x.bind( img_arr )
        print x.kernel
        dest = x.apply()
    return dest


def main():

    raised_expected = False

    s = Py2OpenCL( lambda x: F.sin( x ), prompt=True )
    dev = s.user_dev_selection
    try:
        s.map( (100 * np.random.rand( int(1e3) )).astype('int64') )
    except TypeError:
        # we expect to get this exception
        pass
    else:
        raise Exception("sin shouldn't accept ints")

    img_path = os.path.join( os.path.dirname(test_directory), 'Lenna.png') 
    try:
        img = Image.open( img_path ).convert('RGB') # 3 uint8's per pixel
    except IOError:
        # had some trouble keeping the test image in the package
        print "-- couldn't open test image '%s'; skipping that test" % img_path
    else:
        img_arr = np.array(img)

        before = time.time()
        ocl_result = avg_img( img_arr, user_dev_selection=dev )
        print "openCL img-avg: %.2es" % (time.time() - before)
        before = time.time()
        py_result = avg_img( img_arr, purepy=True )
        print "python img-avg: %.2es" % (time.time() - before)

        Image.fromarray( ocl_result.reshape(img_arr.shape), 'RGB').save('/tmp/oclfoo.png')
        Image.fromarray( py_result, 'RGB').save('/tmp/pyfoo.png')

        #assert (ocl_result == py_result).all(), 'python and openCL computed different image averages: %s' % (ocl_result - py_result)

    arr = np.random.rand( int(1e4) ).astype('float32')

    print 'float: -> int:', Py2OpenCL( lambda x: int(x), user_dev_selection=dev ).map( (1000 * arr).astype('float32') )
    print 'int -> float:', Py2OpenCL( lambda x: float(x), user_dev_selection=dev ).map( (1000 * arr).astype('int32') )

    def f( i, dest, src ):
        x = src[i]
        if x < 0.5:
            dest[i] = -src[i]
        else:
            dest[i] = F.sin(x)

    _a = Py2OpenCL( f, user_dev_selection=dev ).map( arr )
    a = Py2OpenCL( lambda x: -x if x < 0.5 else F.sin(x), user_dev_selection=dev ).map( arr )

    before = time.time()

    assert (_a == a).all(), 'Lambda and function versions computed different results??'

    print "sine - OpenCL: for %d elements, took" % len(a), time.time() - before
    # b = lmb( arr )  # conditionals don't work this way in Numpy
    before = time.time()
    b = np.where( arr < 0.5, -arr, np.sin(arr) )
    print "sine - numpy: for %d elements, took" % len(a), time.time() - before
    print "max delta: %.2e\n" % np.max(a - b)

    before = time.time()
    a = Py2OpenCL( lambda x: F.atanpi(x), user_dev_selection=dev ).map( arr )
    print "arctan(x) / pi - openCL: for %d elements, took %.2es" % (len(a), time.time() - before)
    before = time.time()
    b = (lambda x: F.atanpi(x) / np.pi)( arr )
    print "arctan(x) / pi - numpy: for %d elements, took %.2es" % (len(a), time.time() - before)

    lmb = lambda x: -x if x < 0.5 else F.sin(x)

    for n in (100, 10000, 1000000, 10000000):
        rnd = np.random.rand(n).astype(np.float32)

        before = time.time()
        res_np = Py2OpenCL( lmb, user_dev_selection=dev ).map( rnd )
        print "Simple tertiary operator case - OpenCL: for %d elements, took %.2es" % (len(rnd), time.time() - before)

        before = time.time()
        py = map( lmb, rnd )
        print "Simple tertiary operator case - Python: for %d elements, took %.2es" % (len(rnd), time.time() - before)
        print

    import math
    two = Py2OpenCL( lambda x, y: x + y, user_dev_selection=dev )
    for size in (1e4, 1e5, 1e6, 1e7):
        a, b = np.random.rand(int(1e7)).astype(np.float32), np.random.rand(int(1e7)).astype(np.float32)

        before = time.time()
        res = two.map( a, b )
        print "Simple sum - OpenCL (size=1e%s): %.2es" % (math.log10(size), time.time() - before)
        before = time.time()
        r2 = a + b
        print "Simple sum - numpy (size=1e%s) %.2es" % (math.log10(size), time.time() - before)
        print "  -> max delta: %.2e\n" % np.max(r2 - res)

    two = Py2OpenCL( lambda x, y: x * y, user_dev_selection=dev )
    for size in (1e4, 1e5, 1e6, 1e7):
        a, b = np.random.rand(int(1e7)).astype(np.float32), np.random.rand(int(1e7)).astype(np.float32)

        before = time.time()
        res = two.map( a, b )
        print "Simple multiplication - OpenCL (size=1e%s): %.2es" % (int(math.log10(size)), time.time() - before)
        before = time.time()
        r2 = a * b
        print "Simple multiplication - numpy (size=1e%s): %.2es" % (int(math.log10(size)), time.time() - before)
        print "  -> max delta: %.2e\n" % np.max(r2 - res)

    print
    two = Py2OpenCL( lambda x, y: x ** y, user_dev_selection=dev )
    for size in (1e4, 1e5, 1e6, 1e7):
        a, b = np.random.rand(int(1e7)).astype(np.float32), np.random.rand(int(1e7)).astype(np.float32)

        before = time.time()
        res = two.map( a, b )
        print "Simple exponents - OpenCL (size=1e%s): %.2es" % (int(math.log10(size)), time.time() - before)
        before = time.time()
        r2 = a ** b
        print "Simple exponents - numpy (size=1e%s): %.2es" % (int(math.log10(size)), time.time() - before)
        print "  -> max delta: %.2e\n" % np.max(r2 - res)


if __name__ == '__main__':
   main()
