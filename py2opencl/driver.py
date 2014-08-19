"""
wrapper around PyOpenCL and py2opencl Python -> OpenCL conversion utility
"""

import pyopencl as cl
import numpy as np

from .convert import lambda_to_kernel


import os
os.environ['PYOPENCL_COMPILER_OUTPUT']='1'


class Py2OpenCL(object):
    argnames = None
    _kernel = None
    ctx = None
    queue = None
    prog = None
    bindings = None
    def __init__(self, lmb, context=None, bindings=None):
        """
        """
        self.ctx = context if context \
                   else cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.bindings = bindings
        self.lmb = lmb

    @property
    def kernel(self):
        return lambda_to_kernel( self.lmb, None, bindings=self.bindings )[1]

    def map(self, *arrays ):
        """
        verify types and number of numpy arrays, then compile kernel.

        note that kernel can't be generated until we know the types involved.
        """
        length, types = None, []
        for a in arrays:
            try:
                types.append( {np.dtype('float16'): 'half',
                               np.dtype('float32'): 'float',
                               np.dtype('float64'): 'double',
                               np.dtype('uint8'): 'uchar',
                               np.dtype('int16'): 'short',
                               np.dtype('int32'): 'int',
                               np.dtype('int64'): 'long'}[ a.dtype ] )
            except KeyError:
                raise ValueError("invalid numpy type: "+str(a.dtype))

            if length is None:
                length = len(a)
            else:
                # FIXME: this precludes legitimate use-cases ...
                assert len(a) == length

        self.argnames, self._kernel = lambda_to_kernel( self.lmb, types, bindings=self.bindings )
        assert self.argnames and len(self.argnames) == len(arrays)

        # compile openCL
        self.prog = cl.Program(self.ctx, self._kernel).build()

        mf = cl.mem_flags
        buffs, nbytes = [], arrays[0].nbytes
        for arr in arrays:
            buffs.append( cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr ))

        # results:
        buffs.append( cl.Buffer(self.ctx, mf.WRITE_ONLY, nbytes) )

        # run!
        self.prog.sum(self.queue, arrays[0].shape, None, *buffs)

        res_np = np.empty_like(arrays[0])
        cl.enqueue_copy(self.queue, res_np, buffs[-1])
        return res_np.copy()


