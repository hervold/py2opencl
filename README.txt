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


