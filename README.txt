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
    a = py2.map( np.random.rand(10000000) )

    print py2.kernel

    >>  __kernel void sum( __global const float *x, __global float *res_g) {
    >>     int gid = get_global_id(0);
    >>     res_g[gid] = (((x[gid] < 0.5)) ? -x[gid] : sin( x[gid] ));
    >>  }


More complex functions are supported, though there are many constraints.  The following function
averages the pixels of an image:

    import numpy as np
    from py2opencl import Py2OpenCL, F
    import Image
    
    img_path = 'py2opencl/test/Lenna.png'
    
    img = np.array( Image.open( img_path ).convert('RGB') )

    def avg( x, y, z, dest, src ):
        # note that the C code produced will handle wrapping automatically
        right = src[ x+1, y, z ]
        left = src[ x-1, y, z ]
        up = src[ x, y-1, z ]
        down = src[ x, y+1, z ]
        dest[x,y,z] = (right / 4) + (left / 4) + (up / 4) + (down / 4)
     
    dest = Py2OpenCL( avg ).map( img )
    Image.fromarray( dest, 'RGB').save('foo.png')


OpenCL Drivers
==============

Py2OpenCL should work out-of-the-box on Mac OS X.  If you're on Linux and don't have a fancy GPU,
I'd suggest AMD's ICD, found at http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/
(as of 13 Nov 2014).  It supports modern Intel CPU's, no GPU required.  (Presumably it supports AMD CPUs as well.)

As of this writing, Intel's beignet driver appears to be broken on Ubuntu 14.04.


Tested Platforms
================

- NVIDIA CUDA / nvidia-opencl-icd-331 on Ubuntu 14.04
- AMD Accelerated Parallel Processing (AMD-APP-SDK-v2.9) on Intel Core i7-3610QM, Ubuntu 14.04
- Apple's OpenCL drivers for the Intel Core i5-4258U, OS X 10.9
- Apple's OpenCL drivers for the Intel "Iris" (Intel HD 4000), OS X 10.9


TODO
====

- support while loops and C-style for loops (ie, 'for i in range(n)')
- performance writeup

