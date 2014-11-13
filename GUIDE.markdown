# py2opencl: OpenCL without the C

Andreas Klöckner's [pyopencl package](http://mathema.tician.de/software/pyopencl/) provides a complete set of OpenCL bindings for Python, but doesn't spare the user from writing their kernel in C, as [his example illustrates](http://documen.tician.de/pyopencl/):

```
prg = cl.Program(ctx, """
__kernel void sum(__global const float *a_g, __global const float *b_g, __global float *res_g) {
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
""").build()
```

**py2opencl** addresses this shortcoming by converting Python to C, at least for a limited set of Python.  Where possible, I've tried to maintain compatibility with Numpy -- that is, one should be able to run ones function either in a for loop or using `py2opencl` and expect to see the same results.


## An Example: Conway's Game of Life in Pure Python

The rules for the Game of Life cellular automaton are easily expressed in a few lines of Python.  In the following example, I define a `next_it` function to generate the next iteration of an array, then convert that function to an OpenCL kernel and calculate 100 iterations.

```python
from py2opencl import Py2OpenCL
import numpy
from numpy.random import randint

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

grid = randint( 0, 2, size=(40, 40) ).astype(numpy.dtype('uint8'))

iterate = Py2OpenCL( next_it )
iterate.bind( grid, return_type = numpy.dtype('uint8') )

print iterate.kernel

for i in range(100):
    grid = iterate.apply( grid )

```

The `Py2OpenCL` constructor takes a function (or lambda) as an argument.  This function is converted to a kernel in the subset of C defined by OpenCL.  However, C is a statically typed language, while Python is dynamically typed -- the Python abstract syntax tree (AST) has nothing to say about the types of `dest` or `src` in this example.   These types must be inferred from the arguments supplied, which happens when `.bind` is called.  In this example, that isn't enough, as 0 and 1 are of ambiguous type, so `py2opencl` doesn't know the return type of the `dest` array.  This is supplied via the `return_type` argument to `.bind`

Note that any old function won't do; `py2opencl` expects function arguments to appear in a specific order:

1. index(es)
2. output/destination array
3 input/source array(s)

In the example above, the `src` array is 2-dimensional, so `py2opencl` expects two index arguments (`x` and `y`), and produces a 2D array as output.


## Auto-generated C Code
The example above is converted to this C kernel:

```C

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define XDIM 40
#define YDIM 40
#define TOTSIZE (XDIM * YDIM)
#define FLATTEN2( i, j ) (TOTSIZE + ((j * XDIM) + i)) % TOTSIZE


__kernel void sum( __global const uchar *src, __global uchar *res_g ) {

    size_t gid_x = get_global_id(0);
    size_t gid_y = get_global_id(1);

    uchar live_neighbors;
  
    live_neighbors = (((((((src[ FLATTEN2( (gid_x - 1),(gid_y - 1) ) ] + src[ FLATTEN2( gid_x,(gid_y - 1) ) ]) 
			   + src[ FLATTEN2( (gid_x + 1),(gid_y - 1) ) ]) + src[ FLATTEN2( (gid_x - 1),gid_y ) ]) 
			 + src[ FLATTEN2( (gid_x + 1),gid_y ) ]) + src[ FLATTEN2( (gid_x - 1),(gid_y + 1) ) ]) 
		       + src[ FLATTEN2( gid_x,(gid_y + 1) ) ]) + src[ FLATTEN2( (gid_x + 1),(gid_y + 1) ) ]);

    if( ((live_neighbors < 2)) ) {
      res_g[ FLATTEN2( gid_x,gid_y ) ] = 0;
    } else {
      if( ((live_neighbors == 3)) ) {
	res_g[ FLATTEN2( gid_x,gid_y ) ] = 1;
      } else {
	if( (src[ FLATTEN2( gid_x,gid_y ) ] && ((live_neighbors == 2))) ) {
	  res_g[ FLATTEN2( gid_x,gid_y ) ] = 1;
	} else {
	  if( ((live_neighbors > 3)) ) {
	    res_g[ FLATTEN2( gid_x,gid_y ) ] = 0;
	  } else {
	    res_g[ FLATTEN2( gid_x,gid_y ) ] = 0;
	  }
	}
      }
    }
  }
```

Note the `FLATTEN2` macro.  OpenCL only accepts flat arrays, but has support for multidimensional indexing via the `get_global_id` function; it's up to the user to convert those X and Y coordinates into a single offset.  `py2opencl` simplifies this by with the FLATTEN2 and FLATTEN3 macros.  It also handles wrap-around in order to prevent access to outside memory.


## Support for Built-in OpenCL Math Functions.

`py2opencl` provides support for built-in functions via the F module, as shown in this example:

```python
from py2opencl import Py2OpenCL, F
import numpy as np

a = Py2OpenCL( lambda x: F.sin( x ) ).map( (100 * np.random.rand( int(1e3) )).astype('int64') )
```

In python, `F.sin` is the same as `numpy.sin`, but `py2opencl` knows to convert it to OpenCL's native sin function.

This example also illustrates the `.map` helper function, which is equivalent calling `.bind` and `.apply`


## TODO

By its nature, `py2opencl` won't ever support many parts of Python: no lists, tuples, or dictionaries.  No calling outside functions.

I will likely add support for limited `for` and `while` loops.

I'd also like to add support for `len`, eg:

```python

def next_it( i, dest, src ):
    if i+1 >= len(src):
      dest[i] = 10
    else:
      dest[i] = dest[i+1]
```
