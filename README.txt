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

    a = Py2OpenCL( lambda x: -x if x < 0.5 else F.sin(x) ) \
        .map( np.random.rand(10000000).astype(np.float32) )

