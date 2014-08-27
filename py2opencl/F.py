"""
Stubs providing support for built-in OpenCL Floating Point Functions
"""
import numpy
import scipy.special

__func = lambda x: x
__func2 = lambda x, y: x
__func3 = lambda x, y, z: remquo



acos = numpy.arccos
acosh = numpy.arccosh
acospi = lambda x: numpy.arccos(x) / numpy.pi
acospi.types = numpy.arccos.types
asin = numpy.arcsin
asinh = numpy.arcsinh
asinpi = lambda x: numpy.arcsin(x) / numpy.pi
asinpi.types = numpy.arcsin.types
atan = numpy.arctan
atan2 = numpy.arctan2
atanh = numpy.arctanh
atanpi = lambda x: numpy.arctan(x) / numpy.pi
atanpi.types = numpy.arctan.types
atan2pi = lambda x, y: numpy.arctan2( x, y ) / numpy.pi
atan2pi.types = numpy.arctan2.types
cbrt = scipy.special.cbrt # cube root
ceil = numpy.ceil
copysign = numpy.copysign
cos = numpy.cos
cosh = numpy.cosh
cospi = lambda x: numpy.cos( numpy.pi * x )
cospi.types = numpy.cos.types
half_divide = lambda x, y: x / y
half_divide.types = ('')
native_divide = lambda x, y: x / y
erfc = scipy.special.erfc
erf = scipy.special.erf
exp = numpy.exp
exp2 = numpy.exp2
exp10 = scipy.special.exp10
expm1 = numpy.expm1
fabs = numpy.fabs
fdim = lambda x, y: numpy.abs( x - y )
floor = numpy.floor
fma = lambda a, b, c: a * b + c
fmax = numpy.maximum
fmin = numpy.minimum
fmod = numpy.mod
#fract = __func2  # fixme
#frexp = __func
hypot = numpy.hypot
ilogb = lambda x: numpy.log(x).astype('int32')
ldexp = __func
ldexp = __func
#lgamma = __func
#lgamma_r = __func
log = numpy.log
log2 = numpy.log2
log10 = numpy.log10
log1p = numpy.log1p
logb = __func
mad = lambda a, b, c: a * b + c
#modf = __func2
nextafter = numpy.nextafter
pow = numpy.power
pown = numpy.power
powr = numpy.power
half_recip = lambda x: 1.0 / x
native_recip = lambda x: 1.0 / x
remainder = numpy.remainder
#remquo = __func3
#rint = __func
rootn = lambda x, y: numpy.power( x, 1.0 / y )
round = numpy.round

rsqrt = lambda x: numpy.power( x, -0.5 )

sin = numpy.sin

#sincos = __func2 # fixme

sinh = numpy.sinh

sinpi = lambda x: numpy.sin( numpy.pi * x )
sinpi.argtypes = []
sinpi.rettype = ''

sqrt = numpy.sqrt

tan = numpy.tan

tanh = numpy.tanh

tanpi = lambda x: numpy.tan( numpy.pi * x )
tanpi.argtypes = []
tanpi.rettype = ''

tgamma = scipy.special.gamma

trunc = numpy.trunc


__floats = set( (numpy.dtype('float16'), numpy.dtype('float32'), numpy.dtype('float64')) )

'''
## annotate argument types & return type
import inspect
__varnames = locals().keys()
for k in __varnames:
    if not 'k'.startswith('_') and type( locals()[k] == type(acos) ):
        f = locals()[k]
        try:                                 f.argtypes = ['_float' for _ in inspect.getargspec( locals()[k] )[0]]
        except (AttributeError, TypeError):  pass
        try:                                 f.rettype = '_float'
        except (AttributeError, TypeError):  pass
'''
