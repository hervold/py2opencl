"""
convert simple Python lambda to OpenCL

currently relies upon 
"""

import ast
import inspect
from . import ast2xml
import xml.etree.ElementTree as ET
import re
import numpy as np

USING_BEIGNET = False

#  numpy char-code : openCL scalar type
npchar_to_typ = {'b': 'char',
                 'h': 'short',
                 'i': 'int',
                 'l': 'long',
                 'B': 'uchar',
                 'H': 'ushort',
                 'I': 'uint',
                 'L': 'ulong',
                 'e': 'half',
                 'f': 'float',
                 'd': 'double',}
typ_to_npchar = dict( (v,k) for k,v in npchar_to_typ.items() )


nptyp_to_cl = {np.dtype('float16'): 'half',
                np.dtype('float32'): 'float',
                np.dtype('float64'): 'double',
                np.dtype('uint8'): 'uchar',
                np.dtype('int16'): 'short',
                np.dtype('int32'): 'int',
                np.dtype('int64'): 'long'}
cltyp_to_np = dict( (v,k) for k,v in nptyp_to_cl.items() )



def verify_apply( func, argtypes ):
    """
    verify_apply( func, argtypes )

    verify that function accepts arg types given
    returns return-type of function
    """

    #argtypes = [nptyp_to_cl[t] for t in argtypes]
    matching_ret = None
    for t in func.types:
        args, ret = t.split('->')
        assert len(argtypes) == len(args)
        for atyp, ch in zip(argtypes,args):
            if atyp is None:
                continue
            if npchar_to_typ.get(ch) == atyp: # FIXME: or args[i] is None?
                matching_ret = npchar_to_typ[ret]
                break
        if matching_ret:
            return matching_ret

    if set( argtypes ) != set([None]):
        raise TypeError("unfunc %s didn't match provided types -- %s not found among types %s" \
                        % (func.__name__, argtypes, func.types))
    return None


def special_funcs( modname, funcname, symbol_lookup, args ):

    if not modname and funcname == 'int':
        # FIXME: should we check the type of args?  also, need we worry about the return type required
        return 'convert_int_rtz', 'int'
    if not modname and funcname == 'float':
        return 'convert_double', 'double'

    # FIXME: enforce args
    import importlib
    try:
        mod = importlib.import_module(modname)
    except ImportError:
        mod = importlib.import_module('.'+modname, package='py2opencl')
    try:
        func = mod.__getattribute__(funcname)

        # requires_declaration, type, string_representation
        argtypes = [symbol_lookup(a)[1] for a,_ in args]
        return funcname, verify_apply( func, argtypes )
    except AttributeError:
        return funcname, None


def conv( el, symbol_lookup, declarations=None ):
    """
    returns <openCL-C string representation>, <openCL type>
    """
    def is_float(s):
        if s.startswith('float'):
            return s
        raise ValueError('%s is not float' % s)
    
    def is_int(s):
        if s.startswith('int') or s.startswith('uint'):
            return s
        raise ValueError('%s is not int' % s)

    # symbol_lookup returns: requires_declaration, type, string_representation
    def _conv( el ):
        return conv( el, symbol_lookup,  declarations)

    def cpow( left_el, right_el ):
        (lval, ltyp), (rval, rtyp) =  (_conv(left_el), _conv(right_el))
        assert None in (ltyp,rtyp) or ltyp == rtyp, "pow requires types match; got %s, %s" % (ltyp,rtyp)
	return "pow( %s, %s )" % (lval,rval), ltyp or rtyp

    def cnumeric( s ):
	try:
	    return str(int(s)), symbol_lookup(s)[1]
	except ValueError:
	    return s, None

    def conv_cmp( s ):
        # unsupported: Is | IsNot | In | NotIn
        try:
            return {'Eq': '==', 'NotEq': '!=', 'Lt': '<', 'LtE': '<=',
                    'Gt': '>', 'GtE': '>='}[ s ], None
        except KeyError:
            raise ValueError("comparitor not supported: '%s'" % str(s))

    name = el.get('_name')

    if name == 'Name':
	# identity function
	iden = el.get('id')
        if iden == 'True' or iden == 'False':
            return iden.lower(), 'bool'
        _, typ, nom = symbol_lookup(iden)
        return nom, typ

    if name == 'Num':
	# number literal
	return cnumeric( el.get('n') )

    if name == 'BoolOp':
	[op] = el.findall('./op')
	operands = [_conv(x) for x in el.findall('./values/_list_element')]
        return '(%s)' % ({'And': ' && ', 'Or': ' || '}[op.get('_name')]).join( operands ), 'bool'

    if name == 'UnaryOp':
	[operand] = el.findall("./operand")
	operand, typ = _conv( operand )
	# Invert | Not | UAdd | USub
	[op] = el.findall('./op')
	return {'Invert':	'~' + operand,
		'Not':		'!' + operand,
		'UAdd':		operand,
		'USub':		'-' + operand }[ op.get('_name') ], typ

    if name == 'BinOp':
	[op] = el.findall('./op')
	[right] = el.findall('./right')
	[left] = el.findall('./left')
        if op.get('_name') == 'Pow':
            return cpow( left, right )

	cop = {'Add':'+', 'Sub':'-', 'Mult':'*', 'Div':'/', 'Mod':'%',
               'LShift':'<<', 'RShift':'>>', 'BitOr':'|',
               'BitXor':'^', 'BitAnd':'&', 'FloorDiv':'/'}[ op.get('_name') ]

        (lval, ltyp), (rval, rtyp) =  _conv(left), _conv(right)
        typ = rtyp or ltyp
	return '(%s %s %s)' % (lval, cop, rval), typ

    if name == 'If':
        [test] = el.findall('./test')
        [body] = el.findall('./body')
        l = [_conv(x) for x in body.findall('./_list_element')]
        body = ';\n'.join( a for a,b in l )
        ret = """if( %s ) {
%s
}""" % (_conv(test)[0], body)

        if el.findall('./orelse'):
            [orelse] = el.findall('./orelse')
            l = [_conv(x) for x in orelse.findall('./_list_element')]
            orelse = ';\n'.join( a for a,_ in l )
            ret += " else { %s }" % orelse
        return ret, None

    if name == 'IfExp':
	[test] = el.findall('./test')
        [iftrue] = el.findall('./body')
        [iffalse] = el.findall('./orelse')
        (lval, ltyp), (rval, rtyp) = _conv(iftrue), _conv(iffalse)
        typ = ltyp or rtyp
        return '(%s ? %s : %s)' % (_conv(test)[0], lval, rval), typ

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
        # FIXME: enforce types?
        l = []
        for i in range(len(operands)-1):
            l.append( '(%s %s %s)' % (operands[i][0], ops[i][0], operands[i+1][0]) )
        return '(' + ' && '.join(l) + ')', 'bool'

    if name == 'Call':
        [funcname] = el.findall('./func')
        module = funcname.findall('./value')
        if module:
            [module] = module
            module = module.get('id')

        funcname = funcname.get('attr') or funcname.get('id')

        args = map( _conv, el.findall('./args/_list_element') )
        funcname, typ = special_funcs( module, funcname,  symbol_lookup, args )

        # FIXME: problem here is that args could easily be a more complex expression ...
        return '%s( %s )' % (funcname, ', '.join(a for a,t in args)), typ

    if name == 'Assign':
        [target] = el.findall('./targets/_list_element')
        target, ttyp = _conv(target) # eg, 'x' .get('id')

        [operand] =  el.findall('./value')
        operand, otyp = _conv(operand)

        # hackiness here:
        assert symbol_lookup
        target_name = re.match( r'(\w+)\[?', target ).group(1)
        decl, styp, nom = symbol_lookup( target_name )
        typ = styp or ttyp or otyp
        if decl:
            declarations[ target_name ] = typ

        return '%s = %s;' % (target, operand), typ

    if name == 'Subscript':
        [name] = el.findall('./value')
        [subscr] = el.findall('./slice')
        val, typ = _conv(name)
        sval, styp = _conv(subscr)
        return '%s[%s]' % (val, sval), typ

    if name == 'Index':
        [val] = el.findall('./value')
        return _conv(val)

    if name == 'Expr':
        # we can safely ignore these?  random strings (such as docstrings) come back as Expressions
        return '', None

    #raise ValueError, "We don't know what to do with an element of type '%s'" % name
    return '', None

import xml.dom.minidom
def pprint( s ):
    if not isinstance( s, basestring ):
        s = ET.tostring(s)
    return xml.dom.minidom.parseString( s ).toprettyxml()


def lambda_to_kernel( lmb, types, bindings=None ):
    """
    @types -- numpy types
    """
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

    declarations = dict( zip( argnames, types ) ) if types else {}
    def symbol_lookup( s ):
        target_name = None
        # returns: requires_declaration, type, string_representation
        if s in declarations:
            return True, nptyp_to_cl[declarations[s]], (s + '[gid]')  # requires_declaration=True, b/c it's moot
        else:
            # hackiness here:
            target_name = re.match( r'(\w+)\[?', s ).group(1)
            if target_name != s:
                return symbol_lookup( target_name ) # + [....] ?
        #raise ValueError('symbol not found: %s' % str(s))
        return False, None, s

    [body] = func.findall("./body")
    kernel_body, result_typ = conv(body, symbol_lookup=symbol_lookup, declarations=declarations)
    numpy_typ = cltyp_to_np[result_typ]

    sigs = ["__global const %s *%s" % (nptyp_to_cl[ typ ], aname) for typ,aname in zip(types,argnames)] \
           if types \
             else ["__global const float *%s" % aname for aname in argnames]

    sigs.append('__global %s *res_g' % result_typ)
    sigs = ', '.join(sigs)

    kernel = """

__kernel void sum( %(sigs)s ) {
  int gid = get_global_id(0);
  res_g[gid] = %(body)s;
}""" % {'sigs': sigs, 'body': kernel_body}

    # some platforms require this, and others complain ...
    kernel = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n" + kernel

    if bindings is None:
        kernel = "/* NOTE: without numpy bindings, some types might be incorrectly annotated as None */" + kernel

    return (argnames, kernel, result_typ)



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

    argname_to_type = dict( (nom, nptyp_to_cl[ntyp]) for nom, ntyp in zip(argnames, types ) ) if types \
                      else dict( (a, None) for a in argnames )

    declarations = {}
    def symbol_lookup( s ):
        # returns: requires_declaration, type, string_representation
        if s == idx_name or s == 'gid':
            return False, None, 'gid'

        if s == results_name or s == 'res_g':
            return True, None, 'res_g'

        if argname_to_type:
            if s in argname_to_type:
                return False, argname_to_type[s], s

        if bindings:
            if s in bindings:
                return False, type(bindings[s]), str(bindings[s])

        return True, declarations.get(s), s

    # FIXME: this doesn't cover if/then, for, while ...
    [funcbod] = func.findall('./body')
    assignments = [conv(el, symbol_lookup=symbol_lookup, declarations=declarations) for el in funcbod.getchildren()]
    assignments = [a for a,b in assignments]

    #assignments = [conv(el, symbol_lookup=symbol_lookup) for el in func.findall("./body/_list_element[@_name='Assign']")]

    [body] = func.findall("./body")
    _, typ = conv(body, symbol_lookup=symbol_lookup, declarations=declarations)

    # res_g should appear in declarations, as we want to know its inferred type, but we don't actually want to declare it
    result_typ = declarations['res_g']
    del declarations['res_g']

    sigs = ["__global const %s *%s" % ( nptyp_to_cl[ntyp], aname) for ntyp,aname in zip(types,argnames)] \
           if types else ["__global const float *%s" % aname for aname in argnames]
    sigs.append( '__global %s *res_g' % result_typ)

    input_sig =  ', '.join(sigs)
    decl = '\n'.join( '%s %s;' % (typ, nom) for nom, typ in declarations.items())

    kernel = """

__kernel void sum( %(sig)s ) {
  %(decl)s
  int gid = get_global_id(0);
  %(body)s
}""" % {'decl': decl, 'sig': input_sig, 'body': '\n  '.join(assignments)}

    # some platforms require this, and others complain ...
    kernel = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n" + kernel

    if bindings is None:
        kernel = "/* NOTE: without numpy bindings, some types might be incorrectly annotated as None */" + kernel

    return (argnames, kernel, result_typ)



"""

x = Py2OpenCL( lambda x: x + 1 ).map( numpy.array(...) )

# .bind used for type inference
print Py2OpenCL( lambda x: x + 1 ).bind( numpy.array(...) ).kernel

x = Py2OpenCL( lambda x: x + 1 ).bind( numpy.array(...) ).apply()

"""
