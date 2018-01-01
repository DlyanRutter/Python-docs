def split(text, sep=None, maxsplit=-1):
    "Like str.split, but also strips whitespace"
    return [t.strip() for t in text.strip().split(sep, maxsplit) if t]

def what_grammar_should_output():
    "An example of the type of structure grammar function should output"
    return {'Expression': (['Term', '[+-]', 'Expression'], ['Term']),
     'Term': (['Factor', '[*/]', 'Term'], ['Factor']),
     }

def grammar(description, whitespace=r'\s*'):
    """Convert a description to a grammar. Each line is a rule for a
    non-terminal symbol; it looks liike this:
        Symbol => A1 A2 ... | B1 B2 ... | C1 C2 ...
    where the right-hand side is one or more alternatives, separated by
    the '|' sign. Each alternative is a sequence of atoms, separated by
    spaces. An atom is either a symbol on some left-hand side, or it is
    a regular expression that will be passed to re.match to match a token.
    Notation for *, +, or ? not allowed in a rule alternative (but ok
    within a token). Use '\' to continue long lines. You must include spaces
    or tabs around '=>' and '|'. That's within the grammar description itself.
    The grammar that gets defined allows whitespace between tokens by default
    specify '' as the second argument to grammar() to disallow this (or suppl
    any regular expressions to describe allowable whitespace between tokens)."""
    G = {' ': whitespace}
    description = description.replace('\t', ' ')
    for line in split(description, '\n'):
        lhs, rhs = split(line, ' => ', 1)
        alternatives = split(rhs, ' | ')
        G[lhs] = tuple(map(split, alternatives))
    return G

G = grammar("""
Expression  => Term [+-] Expression | Term
Term        => Factor[*/] Term | Factor
Factor      => Funcall | Var | Num [(] Exponent [)]
Funcall     => Var [(] Expressions [)] 
Expressions => Expression [,] Expressions | Expression
Variable    => [a-zA-Z_]\w*
Number      => [-+]?[0-9]+([.][0-9]*)?
""")

def what_parse_should_output():
    "What would be returned if parse('Exp', 'a * x', G) were entered"
    return (['Exp', ['Term', ['Factor', ['Var', 'a']],
                     '*',
                     ['Term', ['Factor', ['Var', 'x']]]]], '')

def parse(start_symbol, text, grammar):
    """Example call: parse('Exp', '3*x + b', G).
    Returns a (tree, remainder) pair. If remainder is '', it parsed the whole
    string. Failure iff remainder is None. This is a deterministic PEG parser,
    so rule order (left-to-right) matters. Do 'E => T op E | T', putting the
    longest parse first; don't do 'E => T | T op E'
    Also, no left recursion allowed: don't do 'E => E op T'"""
    tokenizer = grammar[' '] + '(%s)'
    def parse_sequence(sequence, text):
        result = []
        for atom in sequence:
            tree, text = parse_atom(atom, text)
            if text is None: return Fail
            result.append(tree)
        return result, text
    def parse_atom(atom, text):
        if atom in grammar:  
            for alternative in grammar[atom]:
                tree, rem = parse_sequence(alternative, text)
                if rem is not None: return [atom]+tree, rem  
            return Fail
        else:  
            m = re.match(tokenizer % atom, text)
            return Fail if (not m) else (m.group(1), text[m.end():])
    return parse_atom(start_symbol, text)

Fail = (None, None)

def verify(G):
    """Find all tokens on Lhs and Rhs and shows them where non-terminals are things
    on Lhs, Terminals are things on RHS but not on LHS (these should be reg-exps).
    Suspects are things that look like they're the name that should appear on LHS
    {alphanumeric characters} but don't because it was mispelled at some point.
    Orphans are things that appear on LHS but not RHS {they are useless}"""
    lhstokens = set(G) - set([' '])
    rhstokens = set(t for alts in G.values() for alt in alts for t in alt)
    def show(title, tokens): print title,'=',' '.join(sorted(tokens))
    show('Non-Terms', G)
    show('Terminals', rhstokens - lhstokens)
    show('Suspects ', [t for t in (rhstokens - lhstokens) if t.isalnum()])
    show('Orphans ', lhstokens - rhstokens)

#print what_grammar_should_output()
#print what_parse_should_output()
print verify(G)
