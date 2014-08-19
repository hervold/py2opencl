"""
convert simple Python lambda to OpenCL

currently relies upon 
"""

import ast
import inspect
from . import ast2xml
import xml.etree.ElementTree as ET
import re

def special_funcs( module, funcname, symbol_lookup, args ):
    if not module and funcname == 'int':
        # FIXME: should we check the type of args?
        #return 'convert_int4_rtz'
        return 'convert_int_rtz'
    if not module and funcname == 'float':
        #return 'convert_float4_rtz'
        return 'convert_float_rtz'
    return funcname


def conv( el, symbol_lookup=None, declarations=None ):
    def _conv( el ):
        return conv( el, symbol_lookup,  declarations)

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
            return symbol_lookup(iden)[2]
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

    if name == 'If':
        [test] = el.findall('./test')
        [body] = el.findall('./body')
        body = ';\n'.join( _conv(x) for x in body.findall('./_list_element') )
        ret = """if( %s ) {
%s
}""" % (_conv(test), body)

        if el.findall('./orelse'):
            [orelse] = el.findall('./orelse')
            orelse = ';\n'.join( _conv(x) for x in orelse.findall('./_list_element') )
            ret += " else { %s }" % orelse
        return ret

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
        module = funcname.findall('./value')
        if module:
            [module] = module
            module = module.get('id')

        funcname = funcname.get('attr') if funcname.get('attr') else funcname.get('id')

        args = map( _conv, el.findall('./args/_list_element') )
        funcname = special_funcs( module, funcname,  symbol_lookup, args )

        # FIXME: problem here is that args could easily be a more complex expression ...
        return '%s( %s )' % (funcname, ', '.join(args))

    if name == 'Assign':
        [target] = el.findall('./targets/_list_element')
        target = _conv(target) # eg, 'x' .get('id')

        [operand] =  el.findall('./value')
        operand = _conv(operand)

        # hackiness here:
        assert symbol_lookup
        target_name = re.match( r'(\w+)\[?', target ).group(1)
        if symbol_lookup( target_name )[0]:
            declarations[ target ] = 'uchar'
            return '%s = %s;' % (target, operand)
        return '%s = %s;' % (target, operand)

    if name == 'Subscript':
        [name] = el.findall('./value')
        [subscr] = el.findall('./slice')
        return '%s[%s]' % (_conv(name), _conv(subscr))

    if name == 'Index':
        [val] = el.findall('./value')
        return _conv(val)

    if name == 'Expr':
        # we can safely ignore these?  random strings (such as docstrings) come back as Expressions
        return ''


import xml.dom.minidom
def pprint( s ):
    if not isinstance( s, basestring ):
        s = ET.tostring(s)
    return xml.dom.minidom.parseString( s ).toprettyxml()



def lambda_to_kernel( lmb, types, bindings=None ):
    # lstrip, b/c there's likely whitespace that WILL get parsed
    src = ast.parse( inspect.getsource( lmb ).lstrip() )
    root = ET.fromstring( ast2xml.ast2xml().convert(src) )

    if not root.findall(".//*[@_name='Lambda']"):
        return function_to_kernel( lmb, types, bindings )

    [func] = root.findall(".//*[@_name='Lambda']")
    # argnames are used to (1) determine order of input matrices, and (2) enforce namespace
    args = func.findall("args/args/_list_element[@id]")
    argnames = [a.get('id') for a in args]
    assert argnames

    argname_to_type = dict( zip( argnames, types ) ) if types else None

    def symbol_lookup( s ):
        # returns: requires_declaration, type, string_representation
        if argname_to_type:
            if s in argname_to_type:
                return False, argname_to_type[s], (s + '[gid]')
            raise ValueError('symbol not found: %s' % str(s))
        return False, None, (s + '[gid]')

    [body] = func.findall("./body")
    kernel_body = conv(body, symbol_lookup=symbol_lookup)

    input_sig = ', '.join("__global const %s *%s" % (typ,aname) for typ,aname in zip(types,argnames)) \
                if types \
                   else ', '.join("__global const float *%s" % aname for aname in argnames)

    kernel = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void sum( %(sig)s, __global uchar *res_g) {
  int gid = get_global_id(0);
  res_g[gid] = %(body)s;
}""" % {'sig': input_sig, 'body': kernel_body}

    return (argnames, kernel)



def function_to_kernel( f, types, bindings=None ):
    #####
    # not a lambda, but a traditional function
    # lstrip, b/c there's likely whitespace that WILL get parsed
    src = ast.parse( inspect.getsource( f ).lstrip() )
    root = ET.fromstring( ast2xml.ast2xml().convert(src) )

    [func] = root.findall(".//*[@_name='FunctionDef']")
    argnames = [x.get('id') for x in
                func.findall("./args[@_name='arguments']/args/_list_element[@_name='Name']")]
    assert len(argnames) > 1
    idx_name = argnames.pop(0)
    results_name = argnames.pop(0)

    argname_to_type = dict( zip( argnames, types ) ) if types \
                      else dict( (a, None) for a in argnames )

    def symbol_lookup( s ):
        # returns: requires_declaration, type, string_representation
        if s == idx_name or s == 'gid':
            return False, None, 'gid'

        if s == results_name or s == 'res_g':
            return False, None, 'res_g'

        if argname_to_type:
            if s in argname_to_type:
                return False, argname_to_type[s], s

        if bindings:
            if s in bindings:
                return False, type(bindings[s]), str(bindings[s])

        return True, None, s

    # FIXME: this doesn't cover if/then, for, while ...
    [funcbod] = func.findall('./body')
    declarations = {}
    assignments = [conv(el, symbol_lookup=symbol_lookup, declarations=declarations) for el in funcbod.getchildren()]

    #assignments = [conv(el, symbol_lookup=symbol_lookup) for el in func.findall("./body/_list_element[@_name='Assign']")]

    [body] = func.findall("./body")
    kernel_body = conv(body, symbol_lookup=symbol_lookup, declarations=declarations)

    sigs = ["__global const %s *%s" % (typ,aname) for typ,aname in zip(types,argnames)] \
           if types else ["__global const float *%s" % aname for aname in argnames]
    sigs.append( '__global uchar *res_g' )

    input_sig =  ', '.join(sigs)
    decl = '\n'.join( '%s %s;' % (typ,nom) for nom, typ in declarations.items())

    kernel = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void sum( %(sig)s ) {
  %(decl)s
  int gid = get_global_id(0);
  %(body)s
}""" % {'decl': decl, 'sig': input_sig, 'body': '\n  '.join(assignments)}

    print kernel

    return (argnames, kernel)



"""

x = Py2OpenCL( lambda x: x + 1 ).map( numpy.array(...) )

# .bind used for type inference
print Py2OpenCL( lambda x: x + 1 ).bind( numpy.array(...) ).kernel

x = Py2OpenCL( lambda x: x + 1 ).bind( numpy.array(...) ).apply()

"""
