"""
convert simple Python lambda to OpenCL

currently relies upon 
"""

import ast
import inspect
from . import ast2xml
import xml.etree.ElementTree as ET


def conv( el, symbol_lookup=None ):
    def _conv( el ):
        return conv( el, symbol_lookup )

    def cpow( left_el, right_el ):
	return "pow( %s, %s )" % (_conv(left_el), _conv(right_el))

    def cnumeric( s ):
	try:
	    return str(int(s))
	except ValueError:
	    return s

    def conv_cmp( s ):
        # unsupported: Is | IsNot | In | NotIn
        try:
            return {'Eq': '==', 'NotEq': '!=', 'Lt': '<', 'LtE': '<=',
                    'Gt': '>', 'GtE': '>='}[ s ]
        except KeyError:
            raise ValueError("comparitor not supported: '%s'" % str(s))

    name = el.get('_name')
    if name == 'Name':
	# identity function
	iden = el.get('id')
        if iden == 'True' or iden == 'False':
            return iden.lower()
        if symbol_lookup is not None:
            return symbol_lookup(iden)[1]
        return '{{' + iden + '}}'

    if name == 'Num':
	# number literal
	return cnumeric( el.get('n') )

    if name == 'BoolOp':
	[op] = el.findall('./op')
	operands = [_conv(x) for x in el.findall('./values/_list_element')]
        return '(%s)' % ({'And': ' && ', 'Or': ' || '}[op.get('_name')]).join( operands )

    if name == 'UnaryOp':
	[operand] = el.findall("./operand")
	operand = _conv( operand )
	# Invert | Not | UAdd | USub
	[op] = el.findall('./op')
	return {'Invert':	'~' + operand,
		'Not':		'!' + operand,
		'UAdd':		operand,
		'USub':		'-' + operand}[ op.get('_name') ]

    if name == 'BinOp':
	[op] = el.findall('./op')
	[right] = el.findall('./right')
	[left] = el.findall('./left')
        if op.get('_name') == 'Pow':
            return cpow( left, right )

	cop = {'Add': '+', 'Sub':'-','Mult':'*','Div':'/','Mod':'%',
               'LShift':'<<','RShift':'>>','BitOr':'|',
               'BitXor':'^','BitAnd':'&','FloorDiv':'/'}[ op.get('_name') ]

	return '(%s %s %s)' % (_conv(left), cop, _conv(right))

    if name == 'IfExp':
	[test] = el.findall('./test')
        [iftrue] = el.findall('./body')
        [iffalse] = el.findall('./orelse')
        return '(%s ? %s : %s)' % (_conv(test), _conv(iftrue), _conv(iffalse))

    if name == 'Compare':
	# to suppory Python's y < 1 < x < 20 syntax ...
	[ops] = el.findall('./ops')
	ops = [conv_cmp(op.get('_name')) for op in ops.findall('./_list_element')]
        [left] = el.findall('./left')
	[operands] = el.findall('./comparators')
	operands = [left] + sorted( (item for item in operands.findall('_list_element')),
                                    key=lambda x: (int(x.get('lineno')), int(x.get('col_offset'))) )
	operands = [_conv( item ) for item in operands]
        assert len(operands) == len(ops) + 1
        l = []
        for i in range(len(operands)-1):
            l.append( '(%s %s %s)' % (operands[i], ops[i], operands[i+1]) )
        return '(' + ' && '.join(l) + ')'

    if name == 'Call':
        [funcname] = el.findall('./func')
        [module] = funcname.findall('./value')
        module = module.get('id')
        
        funcname = funcname.get('attr')
        args = el.findall('./args/_list_element')
        print "-- module: %s, funcname: %s, args: %s" % (module, funcname, map(_conv,args))
        # FIXME: problem here is that args could easily be a more complex expression ...
        return '%s( %s )' % (funcname, ', '.join( _conv(a) for a in args ))


import xml.dom.minidom
def pprint( s ):
    if not isinstance( s, basestring ):
        s = ET.tostring(s)
    return xml.dom.minidom.parseString( s ).toprettyxml()



def lambda_to_kernel( lmb, types ):
    # lstrip, b/c there's likely whitespace that WILL get parsed
    src = ast.parse( inspect.getsource( lmb ).lstrip() )
    root = ET.fromstring( ast2xml.ast2xml().convert(src) )
    [func] = root.findall(".//*[@_name='Lambda']")

    # argnames are used to (1) determine order of input matrices, and (2) enforce namespace
    args = func.findall("args/args/_list_element[@id]")
    argnames = [a.get('id') for a in args]
    assert argnames

    argname_to_type = dict( zip( argnames, types ) )

    def symbol_lookup( s ):
        if s in argname_to_type:
            return argname_to_type[s], (s + '[gid]')
        raise ValueError('symbol not found: %s' % str(s))

    [body] = func.findall("./body")
    kernel_body = conv(body, symbol_lookup=symbol_lookup)

    input_sig = ', '.join("__global const %s *%s" % (typ,aname) for typ,aname in zip(types,argnames))

    kernel = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void sum( %(sig)s, __global float *res_g) {
  int gid = get_global_id(0);
  res_g[gid] = %(body)s;
}
""" % {'sig': input_sig, 'body': kernel_body}

    return (argnames, kernel)



"""
params:
  ast.arguments object:  x.args[0].id -> 'x'

2 * x -> BinOp:{op=ast.Mult, left=ast.Num, right=ast.Name->right.id='x'}


a = numpy.array(...)
b = numpy.array(...)

# simple N-len, N-len -> N-len
OpenCL.apply( lambda a, b: a+b ) \
      .zip( OpenCL.zip(a,b) )
- OR -
# more complex: range(i)
c = numpy.array(...)

OpenCL.apply( lambda i, a, b: (i/2, a[i] + b[i/2]) ) \
      .torange( len(a) ) \
      .into( c ) \
      .using( a, b )
"""
