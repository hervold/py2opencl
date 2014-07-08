"""
wrapper around PyOpenCL and py2opencl Python -> OpenCL conversion utility
"""

import pyopencl as cl
import numpy as np

from .convert import lambda_to_kernel


class Py2OpenCL(object):
    argnames = None
    kernel = None
    ctx = None
    queue = None
    prog = None
    def __init__(self, lmb):
        """
        """
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        self.argnames, self.kernel = lambda_to_kernel( lmb )

        # compile openCL
        self.prog = cl.Program(self.ctx, self.kernel).build()

    def map(self, *arrays ):
        assert self.argnames and len(self.argnames) == len(arrays)

        length = None
        for a in arrays:
            assert a.dtype in (np.dtype('float16'), np.dtype('float32'), np.dtype('float64'))
            if length is None:
                length = len(a)
            else:
                # FIXME: this precludes legitimate use-cases ...
                assert len(a) == length

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


