"""
wrapper around PyOpenCL and py2opencl Python -> OpenCL conversion utility
"""

import pyopencl as cl
import numpy as np

from .convert import lambda_to_kernel


class Py2OpenCL(object):
    argnames = None
    _kernel = None
    ctx = None
    queue = None
    prog = None
    def __init__(self, lmb):
        """
        """
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        self.lmb = lmb

    @property
    def kernel(self):
        return lambda_to_kernel( self.lmb, None )[1]

    def map(self, *arrays ):
        """
        verify types and number of numpy arrays, then compile kernel.

        note that kernel can't be generated until we know the types involved.
        """
        length, types = None, []
        for a in arrays:
            if a.dtype in (np.dtype('float16'), np.dtype('float32'), np.dtype('float64')):
                types.append('float')
            elif a.dtype in (np.dtype('int16'), np.dtype('int32'), np.dtype('int64')):
                types.append('int')
            else:
                raise ValueError("invalid numpy type: "+str(a.dtype))

            if length is None:
                length = len(a)
            else:
                # FIXME: this precludes legitimate use-cases ...
                assert len(a) == length

        self.argnames, self._kernel = lambda_to_kernel( self.lmb, types )
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

        return res_np


