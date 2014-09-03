"""
simple test of py2opencl
"""

import numpy as np
import time
import os.path
from PIL import Image

from ..driver import Py2OpenCL
from ..convert import lambda_to_kernel
from .. import F

from . import __file__ as test_directory


def avg_img_files( src_path, dest_path ):
    img = Image.open( src_path ).convert('RGB') # 3 uint8's per pixel
    img_arr = np.array(img)
    result = avg_img( img_arr )
    Image.fromarray( result, 'RGB').save( dest_path )



def avg_img( img_arr, purepy=False ):
    """
    load an image and set each pixel to the avg of its cardinal neighbors
    """
    # see http://stackoverflow.com/questions/15612373/convert-image-png-to-matrix-and-then-to-1d-array
    rows, cols, depth = img_arr.shape
    flat_arr = img_arr.ravel()
    rowcount = cols * depth   # of bytes per row
    totpix = len(flat_arr)

    def avg( i, dest, src ):
        """
        in order to enforce wrap-around, we'll take mod of each coord

        NOTE: the GID/pointer arithmetic gets a bit tricky (unsigned values?), so we add an extra
        @totpix before the mod in order to keep everything > 0
        """
        right = src[(totpix + i + depth) % totpix]
        left = src[(totpix + i - depth) % totpix]
        up = src[(totpix + i - rowcount) % totpix]
        down = src[(totpix + i + rowcount) % totpix]
        # (a + b + ... ) / 4 can cause overflow
        dest[i] = (right / 4) + (left / 4) + (up / 4) + (down / 4)

    if purepy:
        dest = np.empty_like(flat_arr)
        for i in range(len(flat_arr)):
            avg( i, dest, flat_arr )

    else:
        dest = Py2OpenCL( avg, bindings={'totpix': totpix, 'rowcount': rowcount, 'depth': depth} ).map( flat_arr )

    return dest.reshape( (rows, cols, depth) )


def main():

    raised_expected = False
    try:
        _ = Py2OpenCL( lambda x: F.sin( x ) ).map( (100 * np.random.rand( int(1e3) )).astype('int64') )
    except TypeError:
        raised_expected = True
    assert raised_expected

    img_path = os.path.join( os.path.dirname(test_directory), 'Lenna.png') 

    try:
        img = Image.open( img_path ).convert('RGB') # 3 uint8's per pixel
    except IOError:
        # had some trouble keeping the test image in the package
        img = None
        print "-- couldn't open test image '%s'; skipping that test" % img_path

    if img:
        img_arr = np.array(img)

        before = time.time()
        ocl_result = avg_img( img_arr )
        print "openCL img-avg: %.2es" % (time.time() - before)
        before = time.time()
        py_result = avg_img( img_arr, purepy=True )
        print "python img-avg: %.2es" % (time.time() - before)

        Image.fromarray( ocl_result.reshape(img_arr.shape), 'RGB').save('/tmp/oclfoo.png')
        Image.fromarray( py_result, 'RGB').save('/tmp/pyfoo.png')

        assert (ocl_result == py_result).all(), 'python and openCL computed different image averages'

    arr = np.random.rand( int(1e4) )

    print 'float: -> int:', Py2OpenCL( lambda x: int(x) ).map( 1000 * arr )
    print 'int -> float:', Py2OpenCL( lambda x: float(x) ).map( (1000 * arr).astype('int32') )

    def f( i, dest, src ):
        x = src[i]
        if x < 0.5:
            dest[i] = -src[i]
        else:
            dest[i] = F.sin(x)

    _a = Py2OpenCL( f ).map( arr )
    a = Py2OpenCL( lambda x: -x if x < 0.5 else F.sin(x) ).map( arr )

    before = time.time()

    assert (_a == a).all(), 'Lambda and function versions computed different results??'

    print "sine - OpenCL: for %d elements, took" % len(a), time.time() - before
    # b = lmb( arr )  # conditionals don't work this way in Numpy
    before = time.time()
    b = np.where( arr < 0.5, -arr, np.sin(arr) )
    print "sine - numpy: for %d elements, took" % len(a), time.time() - before
    print "max delta: %.2e\n" % np.max(a - b)

    before = time.time()
    a = Py2OpenCL( lambda x: F.atanpi(x) ).map( arr )
    print "arctan(x) / pi - openCL: for %d elements, took %.2es" % (len(a), time.time() - before)
    before = time.time()
    b = (lambda x: F.atanpi(x) / np.pi)( arr )
    print "arctan(x) / pi - numpy: for %d elements, took %.2es" % (len(a), time.time() - before)

    lmb = lambda x: -x if x < 0.5 else F.sin(x)

    for n in (100, 10000, 1000000, 10000000):
        rnd = np.random.rand(n).astype(np.float32)

        before = time.time()
        res_np = Py2OpenCL( lmb ).map( rnd )
        print "Simple tertiary operator case - OpenCL: for %d elements, took %.2es" % (len(rnd), time.time() - before)

        before = time.time()
        py = map( lmb, rnd )
        print "Simple tertiary operator case - Python: for %d elements, took %.2es" % (len(rnd), time.time() - before)
        print

    import math
    two = Py2OpenCL( lambda x, y: x + y )
    for size in (1e4, 1e5, 1e6, 1e7):
        a, b = np.random.rand(int(1e7)).astype(np.float32), np.random.rand(int(1e7)).astype(np.float32)

        before = time.time()
        res = two.map( a, b )
        print "Simple sum - OpenCL (size=1e%s): %.2es" % (math.log10(size), time.time() - before)
        before = time.time()
        r2 = a + b
        print "Simple sum - numpy (size=1e%s) %.2es" % (math.log10(size), time.time() - before)
        print "  -> max delta: %.2e\n" % np.max(r2 - res)

    two = Py2OpenCL( lambda x, y: x * y )
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
    two = Py2OpenCL( lambda x, y: x ** y )
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
