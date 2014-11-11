from py2opencl import Py2OpenCL
import numpy
from numpy.random import randint
from ..compat import SafeArray


X, Y = 40, 40

def show_iteration( grid, title=None ):
    if title is not None:
        print '=' * 5, title, '=' * 5

    x, y = grid.shape
    for j in range(y):
        print ''.join(' ' if c == 0 else '*' for c in grid[:,j])
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
grid = randint( 0, 2, size=(X, Y) ).astype(numpy.dtype('uint8'))

show_iteration( grid, "Initial Random State")

iterate = Py2OpenCL( next_it )
iterate.bind( grid, return_type='uchar' )

for i in range(120):

    grid = iterate.apply( grid )
    show_iteration( grid, "Generation %d" % i )
