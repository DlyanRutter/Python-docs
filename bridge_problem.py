def path_states(path):
    """Return a list of states in this path. Paths are lists of states and
    actions. An state is a tuple of (here, there, time), and an action is a
    tuple of (person1, person2, direction, where person1 and person 2 are
    represented by the amount of time they take to cross the bridge."""
    newlist = []
    i = 0
    while i < len(path):
        newlist.append(path[i])
        i = i + 2
    return newlist

def path_actions(path):
    """Return a list of states in this path. Paths are lists of states and
    actions. An state is a tuple of (here, there, time), and an action is a
    tupe of (person1, person2, direction, where person1 and person 2 are
    represented by the amount of time they take to cross the bridge."""
    newlist = []
    i = 1
    while i < len(path):
        newlist.append(path[i])
        i = i + 2
    return newlist

def bsuccessors(state):
    """Return a dict of {state:action} pairs. A state is a (here, there, t)
    tuple, where here and there are frozensets of people (indicated by their
    times) and/or the 'light', and t is a number indicating the elapsed time.
    Action is represented as a tuple (person1, person2, arrow), where
    arrow is '->' for here to there and '<-' for there to here. Time in dict
    key = t + max(time) in here or there."""
    here, there, t = state
    if 'light' in here:
        return dict(((here  - frozenset([a,b, 'light']),
                      there | frozenset([a, b, 'light']),
                      t + max(a, b)),
                     (a, b, '->'))
                    for a in here if a is not 'light'
                    for b in here if b is not 'light')
    else:
        return dict(((here | frozenset([a,b, 'light']),
                      there - frozenset([a, b, 'light']),
                      t + max(a, b)),
                     (a, b, '<-'))
                    for a in there if a is not 'light'
                    for b in there if b is not 'light')

def bsuccessors2(state):
    """Return a dict of {state:action} pairs. A state is a
    (here, there) tuple, where here and there are frozensets
    of people (indicated by their travel times) and/or the light."""
    here, there = state
    if 'light' in here:
        return dict(((here - frozenset([a,b, 'light']),
                     there | frozenset([a, b, 'light'])),
                     (a,b, '->'))
                    for a in here if a is not 'light'
                    for b in here if b is not 'light')
    else:
        return dict(((here | frozenset([a,b, 'light']),
                      there - frozenset([a, b, 'light'])),
                     (a, b, '<-'))
                    for a in there if a is not 'light'
                    for b in there if b is not 'light')

def bridge_problem(here):
    """explored is a set of states we have visited, each state will be a
    (people-here, people-there, time-elapsed) tuple. frontier is an ordered
    list of paths we have blazed. 'if not here' means nobody is left there.
    Example is ({1, 2, 5, 10, 'light'}, {}, 0)."""
    here = frozenset(here) | frozenset(['light'])
    explored = set()
    frontier = [ [(here, frozenset(), 0)] ]
    if not here:
        return frontier[0]
    while frontier:
        path = frontier.pop(0)
        here1, there1, t1 = state1 = path[-1]
        if not here1 or here1 == set(['light']):
            return path
        for (state, action) in bsuccessors(state1).items():
            if state not in explored:
                here, there, t = state
                explored.add(state)
                path2 = path + [action, state]
                frontier.append(path2)
                frontier.sort(key=elapsed_time)
    return Fail

def elapsed_time(path):
    return path[-1][2]

def solutions_to_bridge_problem():
    """The answers to the bridge problem when input is 1,2,5,10"""
    S1 = [(2, 1, '->'), (1, 1, '<-'), (5, 10, '->'), (2, 2, '<-'), (2, 1, '->')]
    S2 = [(2, 1, '->'), (2, 2, '<-'), (5, 10, '->'), (1, 1, '<-'), (2, 1, '->')]
    answers = path_actions(bridge_problem([1,2,5,10])) in (S1, S2)
    True

def path_cost1(path):
    """The total cost of a path (which is stored in a tuple
    with the final action. path = (state, (action, total_cost), state, ...)"""
    if len(path) < 3:
        return 0
    else:
        newlist = []
        newlist2 = []
        i = 1
        while i < len(path):
            newlist.append(path[i])
            i = i + 2
        adam = max(newlist)
        eve = min(adam)
        newlist2.append(eve)
        return newlist2[0]

def path_cost(path):
    "The total cost of a path (which is stored in a tuple with the final action)."
    if len(path) < 3:
        return 0
    else:
        action, total_cost = path[-2]
        return total_cost

def bcost(action):
    """Returns the cost (a number) of an action in the
    bridge problem."""
    a, b, arrow = action
    return max(a,b)

def lowest_cost_search(start, successors, is_goal, action_cost):
    """Return the lowest cost path, starting from start state,
    and considering successors(state) => {state:action,...},
    that ends in a state for which is_goal(state) is true,
    where the cost of a path is the sum of action costs,
    which are given by action_cost(action)."""
    Fail = []
    explored = set() 
    frontier = [ [start] ]
    while frontier:
        path = frontier.pop(0)
        state1 = final_state(path)
        if is_goal(state1):  
            return path
        explored.add(state1)
        pcost = path_cost(path)
        for (state, action) in successors(state1).items():
            if state not in explored:
                total_cost = pcost + action_cost(action)
                path2 = path + [(action, total_cost), state]
                add_to_frontier(frontier, path2)
    return Fail


def bridge_problem3(here):
    "Find the fastest (least elapsed time) path to the goal in bridge problem"
    start = (frozenset(here) | frozenset(['light']), frozenset())
    return lowest_cost_search(start, bsuccessors2, all_over, bcost)

def all_over(state):
    here, there = state
    return not here or here == set('light')

def test_costs():
    assert path_cost(('fake_state1', ((2, 5, '->'), 5), 'fake_state2')) == 5
    assert path_cost(('fs1', ((2, 1, '->'), 2), 'fs2', ((3, 4, '<-'), 6), 'fs3')) == 6
    assert bcost((4, 2, '->'),) == 4
    assert bcost((3, 10, '<-'),) == 10
    return 'tests pass'
    
def bridge_problem2(here):
    """explored is a set of states we have visited, each state will be a
    (people-here, people-there, time-elapsed) tuple. frontier is an ordered
    list of paths we have blazed. 'if not here' means nobody is left there.
    Example is ({1, 2, 5, 10, 'light'}, {}, 0)."""
    here = frozenset(here) | frozenset(['light'])
    explored = set()
    frontier = [ [(here, frozenset(), 0)] ]
    while frontier:
        path = frontier.pop(0)
        here1, there1 = state1 = final_state(path)
        if not here1 or (len(here1)==1 and 'light' in here1):
            return path
        explored.add(state1)
        pcost = path_cost(path)
        for (state, action) in bsuccessors2(state1).items():
            if state not in explored:
                total_cost = pcost + bcost(action)
                path2 = path + [(action, total_cost), state]
                add_to_frontier(frontier, path2)
    return Fail

def final_state(path): return path[-1]

def add_to_frontier(frontier, path):
    """Add path to the frontier, replacing costlier path if there is one.
    Find if there is an old path to the final state in the path. If the old
    path was better, do nothing. If the old path was worse, delete it then
    add the new path and re-sort."""
    old = None
    for i,p in enumerate(frontier):
        if final_state(p) == final_state(path):
            old = i
            break
    if old is not None and path_cost(frontier[old]) < path_cost(path):
        return
    elif old is not None:
        del frontier[old]
    frontier.append(path)
    frontier.sort(key=path_cost)

def test_paths_and_actions():
    testpath = [(frozenset([1, 10]), frozenset(['light', 2, 5]), 5), # state 1
                (5, 2, '->'),                                        # action 1
                (frozenset([10, 5]), frozenset([1, 2, 'light']), 2), # state 2
                (2, 1, '->'),                                        # action 2
                (frozenset([1, 2, 10]), frozenset(['light', 5]), 5),
                (5, 5, '->'), 
                (frozenset([1, 2]), frozenset(['light', 10, 5]), 10),
                (5, 10, '->'), 
                (frozenset([1, 10, 5]), frozenset(['light', 2]), 2),
                (2, 2, '->'), 
                (frozenset([2, 5]), frozenset([1, 10, 'light']), 10),
                (10, 1, '->'), 
                (frozenset([1, 2, 5]), frozenset(['light', 10]), 10),
                (10, 10, '->'), 
                (frozenset([1, 5]), frozenset(['light', 2, 10]), 10),
                (10, 2, '->'), 
                (frozenset([2, 10]), frozenset([1, 5, 'light']), 5),
                (5, 1, '->'), 
                (frozenset([2, 10, 5]), frozenset([1, 'light']), 1),
                (1, 1, '->')]
    assert path_states(testpath) == [(frozenset([1, 10]), frozenset(['light', 2, 5]), 5), # state 1
                (frozenset([10, 5]), frozenset([1, 2, 'light']), 2), # state 2
                (frozenset([1, 2, 10]), frozenset(['light', 5]), 5),
                (frozenset([1, 2]), frozenset(['light', 10, 5]), 10),
                (frozenset([1, 10, 5]), frozenset(['light', 2]), 2),
                (frozenset([2, 5]), frozenset([1, 10, 'light']), 10),
                (frozenset([1, 2, 5]), frozenset(['light', 10]), 10),
                (frozenset([1, 5]), frozenset(['light', 2, 10]), 10),
                (frozenset([2, 10]), frozenset([1, 5, 'light']), 5),
                (frozenset([2, 10, 5]), frozenset([1, 'light']), 1)]
    assert path_actions(testpath) == [(5, 2, '->'), # action 1
                                      (2, 1, '->'), # action 2
                                      (5, 5, '->'), 
                                      (5, 10, '->'), 
                                      (2, 2, '->'), 
                                      (10, 1, '->'), 
                                      (10, 10, '->'), 
                                      (10, 2, '->'), 
                                      (5, 1, '->'), 
                                      (1, 1, '->')]
    print 'tests pass'

def test_successors():
    assert bsuccessors((frozenset([1, 'light']), frozenset([]), 3)) == {
                (frozenset([]), frozenset([1, 'light']), 4): (1, 1, '->')}

    assert bsuccessors((frozenset([]), frozenset([2, 'light']), 0)) =={
                (frozenset([2, 'light']), frozenset([]), 2): (2, 2, '<-')}    
    return 'tests pass'

def test_bridge_problem():
    assert bridge_problem(frozenset((1, 2),))[-1][-1] == 2
    assert bridge_problem(frozenset((1, 2, 5, 10),))[-1][-1] == 17
    return 'tests pass'

def test_successors2():
    here1 = frozenset([1, 'light']) 
    there1 = frozenset([])
    here2 = frozenset([1, 2, 'light'])
    there2 = frozenset([3])    
    assert bsuccessors2((here1, there1)) == {
            (frozenset([]), frozenset([1, 'light'])): (1, 1, '->')}
    assert bsuccessors2((here2, there2)) == {
            (frozenset([1]), frozenset(['light', 2, 3])): (2, 2, '->'), 
            (frozenset([2]), frozenset([1, 3, 'light'])): (1, 1, '->'), 
            (frozenset([]), frozenset([1, 2, 3, 'light'])): (2, 1, '->')}
    return 'tests pass'

def test_bridge_problem3():
    here = [1, 2, 5, 10]
    assert bridge_problem3(here) == [
            (frozenset([1, 2, 'light', 10, 5]), frozenset([])), 
            ((2, 1, '->'), 2), 
            (frozenset([10, 5]), frozenset([1, 2, 'light'])), 
            ((2, 2, '<-'), 4), 
            (frozenset(['light', 10, 2, 5]), frozenset([1])), 
            ((5, 10, '->'), 14), 
            (frozenset([2]), frozenset([1, 10, 5, 'light'])), 
            ((1, 1, '<-'), 15), 
            (frozenset([1, 2, 'light']), frozenset([10, 5])), 
            ((2, 1, '->'), 17), 
            (frozenset([]), frozenset([1, 10, 2, 5, 'light']))]
    return 'test passes'

print test_successors()
print test_paths_and_actions()
print test_bridge_problem()
print elapsed_time(bridge_problem([1,2,5,10]))
print test_successors2()
print test_costs()
print test_bridge_problem3()
