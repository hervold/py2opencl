=========
py2OpenCL
=========

OpenCL is a powerful means of applying the same simple function (a "kernel") to
large arrays of similar data.

py2OpenCL uses Python's AST module to convert a Python lambda to an OpenCL
kernel (written in a C-like language), then uses Andreas Klöckner's PyOpenCL
module to submit the kernel to the GPU.  It is not meant to convert arbitrary
Python code to OpenCL, as that would be impossible.  Instead, it is limited to
Python lambdas containing only simple mathematical operations and built-in
OpenCL functions.


Example
=======

The following code returns a new numpy array holding the results of the computation::


    import numpy as np
    from py2opencl import Py2OpenCL, F

    py2 = Py2OpenCL( lambda x: -x if x < 0.5 else F.sin(x) )

    print py2.kernel

    >>>>  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    >>>>
    >>>>  __kernel void sum( __global const float *x, __global float *res_g) {
    >>>>      int gid = get_global_id(0);
    >>>>      res_g[gid] = (((x[gid] < 0.5)) ? -x[gid] : sin( x[gid] ));
    >>>>  }

    a = py2.map( np.random.rand(10000000).astype(np.float32) )


OpenCL Drivers
==============

If you're on Linux and don't have a fancy GPU, I'd suggest AMD's ICD,
found `here http://developer.amd.com/tools-and-sdks/opencl-zone/opencl-tools-sdks/amd-accelerated-parallel-processing-app-sdk/`
(as of 26 Jul 2014).  It supports modern Intel CPU's, no GPU required.  (Presumably it supports AMD CPUs as well.)

As of this writing, Intel's beignet driver appears to be broken on Ubuntu 14.04.
