from functools import update_wrapper

def decorator(d):
    "Make function d a decorator: d wraps a function fn."
    def xd(fn):
        return update_wrapper(d(fn), fn)
    update_wrapper(xd, d)
    return xd

#---------------OR--------------

def decorator(d):
    "Make function d a decorator: d wraps a function fn."
    return lambda fn: update_wrapper(d(fn), fn)

@decorator
def memo(f):
    """Decorator that caches the return value for each call to f(args).
    Then when called again with same args, we can just look it up."""
    cache = {}
    def _f(*args):
        try:
            return cache[args]
        except KeyError:
            cache[args] = result = f(*args)
            return result
        except TypeError:
            return f(args)
    return _f

decorator = decorator(decorator)

def disabled(f): return f

def n_ary(f):
    """Given binary function f(x,y), return an n_ary function such that
    f(x,y,z) = f(x, f(y, z)), etc. Also allow f(x) = x."""
    def n_ary_fn(x, *args):
        return x if not args else f(x, n_ary_fn(*args))
    return n_ary_fn

def seq(x,y): return ('seq', x, y)

def trace(f):
    indent = '  '
    def xf(*args):
        signature = '%s(%s)' % (f.__name__, ', '.join(map(repr, args)))
        print '%s--> %s' % (trace.level*indent, signature)
        trace.level += 1
        try:
            #
            print '%s<-- %s === %s' % ((trace.level-1)*indent,
                                       signature, result)
        finally:
            #
        return #
    trace.level = 0
    return xf

     }
#str.partition(sep)
"""Split the string at the first occurrence of sep, and return a
3-tuple containing the part before the separator, the separator itself,
and the part after the separator. If the separator is not found, return
a 3-tuple containing the string itself, followed by two empty strings."""

#str.splitlines()
"""Return a list of the lines in the string"""
                               

seq = n_ary(seq)

print seq(1,3,4,5,6)
(1,(3,(4,(5,6))))
n_ary(f(x,y))
    new(x, (3434))
