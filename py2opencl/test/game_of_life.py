from py2opencl import Py2OpenCL
import numpy
from numpy.random import randint
from ..compat import SafeArray


X, Y = 40, 40

def show_iteration( grid, title=None ):
    if title is not None:
        print '=' * 5, title, '=' * 5

    grid = numpy.array(grid)

    x, y = grid.shape
    for j in range(y):
        print ''.join('*' if c else ' ' for c in grid[:,j])
    print

def next_it( x, y, dest, src ):
    """
      neighbor coordinates:

       0, 1, 2,
       3,    4,
       5, 6, 7

    """
    live_neighbors = ( src[ x-1, y-1 ] + src[ x, y-1 ] + src[ x+1, y-1 ]
                       + src[ x-1, y ] + src [ x+1, y ]
                       + src[ x-1, y+1 ] + src[ x, y+1 ] + src[ x+1, y+1 ] )
    if live_neighbors < 2:
        dest[x,y] = 0
    elif live_neighbors == 3:
        dest[x,y] = 1
    elif src[x,y] and live_neighbors == 2:
        dest[x,y] = 1
    elif live_neighbors > 3:
        dest[x,y] = 0
    else:
        dest[x,y] = 0

# initialize first grid w/ random choice of 0,
grid = SafeArray.wrap( randint( 0, 2, size=(X, Y) ).astype(numpy.dtype('uint8')) )

#show_iteration( grid, "Initial Random State")

iterate = Py2OpenCL( next_it )
iterate.bind( grid, return_type=numpy.dtype('uint8') )

print iterate.kernel


for i in range(int(1e5)):
    if not i % 10000:
        show_iteration( grid, "Generation %d" % i )
    grid = iterate.apply(grid)
    '''
    out = SafeArray.wrap(numpy.zeros( shape=grid.shape, dtype=grid.dtype))
    for x in range(X):
        for y in range(Y):
            next_it( x, y, out, grid )
    grid = out
    del out
    '''

show_iteration( grid, "Generation %d" % i )
