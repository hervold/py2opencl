"""
wrapper around PyOpenCL and py2opencl Python -> OpenCL conversion utility
"""

import pyopencl as cl
import numpy as np

from .convert import lambda_to_kernel, nptyp_to_cl, cltyp_to_np

import os
os.environ['PYOPENCL_COMPILER_OUTPUT']='1'


class Py2OpenCL(object):
    argnames = None
    _kernel = None
    ctx = None
    queue = None
    prog = None
    bindings = None
    user_dev_selection = None
    def __init__(self, lmb, prompt=False, user_dev_selection=None, bindings=None):
        """
        """
	assert not (prompt and user_dev_selection), "Can't ask for @prompt and provide @user_dev_selection at the same time"
	self.user_dev_selection = user_dev_selection

	if prompt:
	    self.user_dev_selection = ['0'] if Py2OpenCL.only_one_device() \
		else self.init()

        self.ctx = cl.create_some_context( interactive=False, answers=self.user_dev_selection ) \
		if self.user_dev_selection else cl.create_some_context()

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
            types.append( a.dtype )

            if length is None:
                length = len(a)
            else:
                # FIXME: this precludes legitimate use-cases ...
                assert len(a) == length

        for t in sorted(set(types)):
            try:
                assert self.ctx.devices[0].__getattribute__('preferred_vector_width_' + nptyp_to_cl[t]), \
                    "unsupported type on this platform:: numpy:%s openCL:%s" % (t,nptyp_to_cl[t])
            except AttributeError:
                pass

        self.argnames, self._kernel, cl_return_typ = lambda_to_kernel( self.lmb, types, bindings=self.bindings )
        return_typ = cltyp_to_np[cl_return_typ]

        assert self.argnames and len(self.argnames) == len(arrays)

        # compile openCL
        self.prog = cl.Program(self.ctx, self._kernel).build()


        mf = cl.mem_flags
        res_np = np.zeros( len(arrays[0]), dtype=return_typ )

        buffs = [ cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr )
                  for arr in arrays ]

        # results:
        buffs.append( cl.Buffer(self.ctx, mf.WRITE_ONLY, res_np.nbytes) )

        # run!
        self.prog.sum(self.queue, arrays[0].shape, None, *buffs)

        cl.enqueue_copy( self.queue, res_np, buffs[-1] )
        return res_np.copy()

    @staticmethod
    def only_one_device():
	p = cl.get_platforms()
 	return len(p) == 1 and len(p[0].get_devices()) == 1

    def init(self):
	"""
	optional helper method that records user responses about platform for later use
        """
        answers = []
        platforms = cl.get_platforms()

        if not platforms:
            raise Error("no platforms found")
        elif len(platforms) == 1:
            [platform] = platforms
        else:
            print "Choose platform:"
            for i, pf in enumerate(platforms):
                print "[%d] %s" % (i, pf)

            print "Choice [0]:",
            int_choice = int( raw_input() )
	    if 0 <= int_choice < len(platforms):
                platform = platforms[int_choice]

            answers.append( str(int_choice) )

        devices = platform.get_devices()

        if len(devices) == 1:
            answer = 0
        else:
            print "Choose device(s):"
            for i, dev in enumerate(devices):
                print "[%d] %s" % (i, dev)

            print "Choice, comma-separated [0]:",
            answer = raw_input()
            if answer:
                try:
                    if 0 <= int(answer) <= len(devices):
                        answers.append( answer )
                    else:
                        raise ValueError("didn't recognize device "+answer)
                except ValueError:
                    answers.append('0')
            else:
                answers.append('0')
	return answers
