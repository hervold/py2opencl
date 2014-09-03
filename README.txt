=========
py2OpenCL
=========

OpenCL is a powerful means of applying the same simple function (a "kernel") to
large arrays of similar data.

py2OpenCL uses Python's AST module to convert a Python function to an OpenCL
kernel (written in a C-like language), then uses Andreas Klöckner's PyOpenCL
module to submit the kernel to the GPU.  It is not meant to convert arbitrary
Python code to OpenCL, as that would be impossible.  Instead, it is limited to
simple Python lambdas and functions containing only simple mathematical operations
and built-in OpenCL functions.


Examples
========

The following code returns a new numpy array holding the results of the lambda function:


    import numpy as np
    from py2opencl import Py2OpenCL, F
     
    py2 = Py2OpenCL( lambda x: -x if x < 0.5 else F.sin(x) )
     
    print py2.kernel

    >>  __kernel void sum( __global const float *x, __global float *res_g) {
    >>     int gid = get_global_id(0);
    >>     res_g[gid] = (((x[gid] < 0.5)) ? -x[gid] : sin( x[gid] ));
    >>  }
     
    a = py2.map( np.random.rand(10000000) )


More complex functions are supported, though there are many constraints.  The following function
averages the pixels of an image:

    import numpy as np
    from py2opencl import Py2OpenCL, F
    import Image
    
    img_path = 'py2opencl/test/Lenna.png'
    
    img = np.array( Image.open( img_path ).convert('RGB') )
    rows, cols, depth = img.shape
    flat_arr = img.ravel()
    rowcount = cols * depth   # of bytes per row
    totpix = len(flat_arr)
    
    def avg( i, dest, src ):
        """
        in order to enforce wrap-around, we'll take mod of each coord
    
        NOTE: the GID/pointer arithmetic gets a bit tricky (unsigned values?),
	so we add an extra @totpix before the mod in order to keep
        everything > 0
        """
        right = src[(totpix + i + depth) % totpix]
        left = src[(totpix + i - depth) % totpix]
        up = src[(totpix + i - rowcount) % totpix]
        down = src[(totpix + i + rowcount) % totpix]
        # (a + b + ... ) / 4 can cause overflow
        dest[i] = (right / 4) + (left / 4) + (up / 4) + (down / 4)
     
     
    # note that we can't determine external values via introspection
    dest = Py2OpenCL( avg, bindings={'totpix': totpix,
                                     'rowcount': rowcount, 'depth': depth} ).map( flat_arr )
    avg_img = Image.fromarray( dest.reshape( (rows, cols, depth) ), 'RGB')
    avg_img.save('x.png')


OpenCL Drivers
==============

Py2OpenCL should work out-of-the-box on Mac OS X.  If you're on Linux and don't have a fancy GPU,
I'd suggest AMD's ICD, found `here http://developer.amd.com/tools-and-sdks/opencl-zone/opencl-tools-sdks/amd-accelerated-parallel-processing-app-sdk/`
(as of 26 Jul 2014).  It supports modern Intel CPU's, no GPU required.  (Presumably it supports AMD CPUs as well.)

As of this writing, Intel's beignet driver appears to be broken on Ubuntu 14.04.



TODO
====

- support while loops and C-style for loops (ie, 'for i in range(n)')
- performance writeup
- on machines w/ multiple GPU drivers, remember selection

