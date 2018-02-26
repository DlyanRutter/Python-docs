import inspect, pprint
import numpy as np

class Node(object):
    def __init__(self, value):
        #other classes will be able to access a value when a value is passed
        #through Node(value)
        #whenever someone makes an instance of node, they will be able to make
        #a list of edges by saying Node_object.edges.append
        self.value = value
        self.edges = []
        

class Edge(object):
    def __init__(self, value, node_from, node_to):
        #each Edge object will have a value, node_from, and node_to property
        #it can be accessed in other classes by Edge_object.property
        self.value = value
        self.node_from = node_from
        self.node_to = node_to


class Graph(object):
    def __init__(self, nodes=None, edges=None):
        self.nodes = nodes or []
        self.edges = edges or []
        self.node_names = []
        self._node_map = {}

    def set_node_names(self, names):
        """The Nth name in names should correspond to node number N.
        Node numbers are 0 based (starting at 0).
        """
        self.node_names = list(names)
        
    def insert_node(self, new_node_val):
        #calling Node class to get a new node, eg all nodes will be Node objects
        #next it will append the new Node object to our list of nodes
        new_node = Node(new_node_val) #create new node instance
        self.nodes.append(new_node) #append it to our node list
        self._node_map[new_node_val] = new_node #add it to our dictionary
        
    def insert_edge(self, new_edge_val, node_from_val, node_to_val):
            "Insert a new edge, creating new nodes if necessary"
            nodes = {node_from_val: None, node_to_val: None}
            for node in self.nodes:
                if node.value in nodes:
                    nodes[node.value] = node
                    if all(nodes.values()):
                        break
            for node_val in nodes:
                nodes[node_val] = nodes[node_val] or self.insert_node(node_val)
            node_from = nodes[node_from_val]
            node_to = nodes[node_to_val]
            new_edge = Edge(new_edge_val, node_from, node_to)
            node_from.edges.append(new_edge)
            node_to.edges.append(new_edge)
            self.edges.append(new_edge)

    def insert_edge1(self, new_edge_val, node_from_val, node_to_val):
        from_found = None
        to_found = None
        for node in self.nodes:#self.nodes are instances of Node so have "value"
            if node_from_val == node.value:#if starting node==Node obj in list:
                from_found = node#Node obj in list is used as initial node
            if node_to_val == node.value:#if end node==Node obj in list:
                to_found = node#Node obj is used as an end node
        if from_found == None:#if no Node obj == node_from_val:
            from_found = Node(node_from_val)#start node = new Node object from var
            self.nodes.append(from_found)#add Node object to Node object list
        if to_found == None:#if no Node object == node_to_val:
            to_found = Node(node_to_val)#end node = new Node obj of node_to_val
            self.nodes.append(to_found)#add new Node object to Node object list
        new_edge = Edge(new_edge_val, from_found, to_found) #making Edge instance
            #out of Node objects
        
        #print new_edge.node_from #this is identical to printing the same in below fn
        #print new_edge.node_from.__class__ #also identical to below
        #print new_edge.__class__ #also identical to below

        ###from_found and to_found are instances of Node class
        ###"edges" is a property of Node class, so it can  be accessed by Node
        ###objects like from_found. "edges" is a list, so .append method can
        ###be used. Each Node object is an instance of class Node, so it will
        ###have its own "edges" list. Each new_edge is an Edge object containing
        ###two Node objects. Each new edge will have access to "value", "node_from"
        ###and "node_to" because they are Edge instances.
        
        from_found.edges.append(new_edge)#append new Edge obj list in Node class 
        to_found.edges.append(new_edge)#append new Edge obj to edge list in Node class
        self.edges.append(new_edge)#append new Edge obj to our own list of edges 

    def get_edge_list(self):
        """Don't return a list of edge objects!
        Return a list of triples that looks like this:
        (Edge Value, From Node Value, To Node Value)"""
        edge_list = []
        for edge_object in self.edges:
            #print edge_object.node_from >>> <__main__.Node object at 0x10e5836d0>
            #print edge_object.node_from.__class__ >>> <class '__main__.Node'>
            #print edge_object.node_from.__bases__ >>> Node obj has no attr __bases__
            #print edge_object.__class__ >>> <class '__main__.Edge'>

            ###edge_object.node_from is an input argument for Edge class. Inputs
            ###to Edge class are node objects. Thus printing edge_object.node_from
            ###will return an instance of class Node, and so printing
            ###edge_object.node_from.__class__ will return the class name of
            ###the node object. edge_objects are instances of Class Edge. calling
            ###edge_object.__class__ will return the namespace of the edge
            ###instance's class AKA class '__main__.Edge'>
            
            ### edge_object is an instance of new_edge which was derived from
            ### Edge evaluated at a new edge value. Thus their "type" AKA class
            ### comes from the name Edge, and each edge_object is an Edge object
            ### AKA an object created by Edge
            ### you are allowed to call edge_object.node_from because Edge has
            ### init variables called 'node_from', 'node_to', 'value'. These
            ### are therefore attributes of Edge
            
            #print edge_object.__bases__ >>> same error as above
            #print edge_object.__dict__ #>>> {'node_from': <__main__.Node object at
            #                                0x1113acc50>, 'value':100, 'node_to':
            #                                <__main.Node object at 0x1113acbd0>}
            ### This returns same dictionary as new_edge.__dict__ in function above

            ###printing edge_object.__dict__ prints an (implied) dictionary of
            ###the above dictionaries. There is one for each Node object. E.G. if
            ###you insert 4 nodes, there will be four dictionary entries.
            ###printing edge_object.node_from.__dict__ returns a dictionary of
            ###class Node's __init__ function. 'edges' is a list of __main__.Edge
            ###objects and 'value' is the node_from value

            
            #print self.edges.__class__ >>> <type 'list'>
            #print type(edge_object) #>>> <class '__main__.Edge'>
            #print edge_object.node_from.__dict__

            #remember that node_to is a Node object and therefore has property
            # "value"
            edge = (edge_object.value, edge_object.node_from.value, \
                    edge_object.node_to.value)
            
            edge_list.append(edge)
            
        return edge_list
    
    def get_edge_list_names(self):
        """Return a list of triples that looks like this:
        (Edge Value, From Node Name, To Node Name)"""
        return [(edge.value,
                 self.node_names[edge.node_from.value],
                 self.node_names[edge.node_to.value])
                for edge in self.edges]
    
    def get_adjacency_list(self):
        """Don't return any Node or Edge objects!
        You'll return a list of lists.
        The indecies of the outer list represent
        "from" nodes.
        Each section in the list will store a list
        of tuples that looks like this:
        (To Node, Edge Value)"""
        adjacent_tuples_list = []
        adjacent_list = []
        i=0
        edge_list = Graph.get_edge_list(self)
        while i <= len(edge_list):
            for edge_val, from_node_val, to_node_val in edge_list:
                node_edge_tuple = (to_node_val, edge_val)
                if i == from_node_val:
                    adjacent_tuples_list.append(node_edge_tuple)
            if len(adjacent_tuples_list) == 0:
                adjacent_list.append(None)
            else:
                adjacent_list.append(adjacent_tuples_list)
            adjacent_tuples_list = []
            i+=1
        return adjacent_list

    def get_adjacency_list_names(self):
        """Each section in the list will store a list
        of tuples that looks like this:
        (To Node Name, Edge Value).
        Node names should come from the names set
        with set_node_names."""
        adjacency_list = self.get_adjacency_list()
        def convert_to_names(pair, graph=self):
            node_number, value = pair
            return (graph.node_names[node_number], value)
        def map_conversion(adjacency_list_for_node):
            if adjacency_list_for_node is None:
                return None
            return map(convert_to_names, adjacency_list_for_node)
        return [map_conversion(adjacency_list_for_node)
                for adjacency_list_for_node in adjacency_list]
    
    def get_adjacency_matrix(self):
        """Return a matrix, or 2D list.
        Row numbers represent from nodes,
        column numbers represent to nodes.
        Store the edge values in each spot,
        and a 0 if no edge exists."""
        matrix = []
        adjacent_list = Graph.get_adjacency_list(self)
        i = 0
        while i < len(adjacent_list):
            row = list(np.zeros(len(adjacent_list), dtype=np.int32))
            if adjacent_list[i] == None:
                matrix.append(row)
            else:
                in_list = [(node_val, edge_val) for node_val, edge_val in adjacent_list[i]]
                for x,y in in_list:
                    row[x] = y
                matrix.append(row)
            i+=1          
        return matrix

    def find_node(self, node_number):
        "Return the node with value node_number or None"
        return self._node_map.get(node_number)
    
    def _clear_visited(self):
        for node in self.nodes:
            node.visited = False

    def dfs_helper(self, start_node):
        """TODO: Write the helper function for a recursive implementation
        of Depth First Search iterating through a node's edges. The
        output should be a list of numbers corresponding to the
        values of the traversed nodes.
        ARGUMENTS: start_node is the starting Node
        MODIFIES: the value of the visited property of nodes in self.nodes 
        RETURN: a list of the traversed node values (integers).
        """
        ###Remember that we've got a dictionary called self._node_map
        ret_list = [start_node.value]
        # Your code here
        return ret_list

    def dfs(self, start_node_num):
        """Outputs a list of numbers corresponding to the traversed nodes
        in a Depth First Search.
        ARGUMENTS: start_node_num is the starting node number (integer)
        MODIFIES: the value of the visited property of nodes in self.nodes
        RETURN: a list of the node values (integers)."""
        self._clear_visited()
        start_node = self.find_node(start_node_num)
        return self.dfs_helper(start_node)

    def dfs_names(self, start_node_num):
        """Return the results of dfs with numbers converted to names."""
        return [self.node_names[num] for num in self.dfs(start_node_num)]

    def bfs(self, start_node_num):
        """TODO: Create an iterative implementation of Breadth First Search
        iterating through a node's edges. The output should be a list of
        numbers corresponding to the traversed nodes.
        ARGUMENTS: start_node_num is the node number (integer)
        MODIFIES: the value of the visited property of nodes in self.nodes
        RETURN: a list of the node values (integers)."""
        node = self.find_node(start_node_num)
        self._clear_visited()
        ret_list = [node.value]
        # Your code here
        return ret_list

    def bfs_names(self, start_node_num):
        """Return the results of bfs with numbers converted to names."""
        return [self.node_names[num] for num in self.bfs(start_node_num)]

    
### DFS uses stack. Start at one node and mark as seen. Pick an edge follow it then
### mark that node as seen and add it to the stack. Repeat as long as
### unseen edges and unseen nodes. When you hit a node you've seen before,
### go back to the previous node and try another edge. If you run out of edges
### with new nodes, you pop the current node from the stack and go back to the
### one before it, which is just the next one on the stack. Continue until
### you've popped everything off the stack, or you find the node you were
### originally looking for. OR could use recursion process of picking an
### edge and marking a node until you run out of nodes to explore. That becomes
### the base case, and you move back to the last level of recusion, which
### happens to be the previous node in the search.


### BFS you search through every edge of one node before moving on. Use a queue.
### in queues you remove the first element you put in it. For a stack, you
### remove the most recent. When you run out of edges, you deque a node from a
### queue and use that as the next starting place. Look at each node adjacent
### to it adding each node to the stack until we've exhausted our options.
### When you DQ you get a node adjacent to the one you started with.
    

"""
graph = Graph()

# You do not need to change anything below this line.
# You only need to implement Graph.dfs_helper and Graph.bfs

graph.set_node_names(('Mountain View',   # 0
                      'San Francisco',   # 1
                      'London',          # 2
                      'Shanghai',        # 3
                      'Berlin',          # 4
                      'Sao Paolo',       # 5
                      'Bangalore'))      # 6 

graph.insert_edge(51, 0, 1)     # MV <-> SF
graph.insert_edge(51, 1, 0)     # SF <-> MV
graph.insert_edge(9950, 0, 3)   # MV <-> Shanghai
graph.insert_edge(9950, 3, 0)   # Shanghai <-> MV
graph.insert_edge(10375, 0, 5)  # MV <-> Sao Paolo
graph.insert_edge(10375, 5, 0)  # Sao Paolo <-> MV
graph.insert_edge(9900, 1, 3)   # SF <-> Shanghai
graph.insert_edge(9900, 3, 1)   # Shanghai <-> SF
graph.insert_edge(9130, 1, 4)   # SF <-> Berlin
graph.insert_edge(9130, 4, 1)   # Berlin <-> SF
graph.insert_edge(9217, 2, 3)   # London <-> Shanghai
graph.insert_edge(9217, 3, 2)   # Shanghai <-> London
graph.insert_edge(932, 2, 4)    # London <-> Berlin
graph.insert_edge(932, 4, 2)    # Berlin <-> London
graph.insert_edge(9471, 2, 5)   # London <-> Sao Paolo
graph.insert_edge(9471, 5, 2)   # Sao Paolo <-> London
# (6) 'Bangalore' is intentionally disconnected (no edges)
# for this problem and should produce None in the
# Adjacency List, etc.

pp = pprint.PrettyPrinter(indent=2)

print "Edge List"
pp.pprint(graph.get_edge_list_names())

print "\nAdjacency List"
pp.pprint(graph.get_adjacency_list_names())

print "\nAdjacency Matrix"
pp.pprint(graph.get_adjacency_matrix())

print "\nDepth First Search"
pp.pprint(graph.dfs_names(2))

# Should print:
# Depth First Search
# ['London', 'Shanghai', 'Mountain View', 'San Francisco', 'Berlin', 'Sao Paolo']

print "\nBreadth First Search"
pp.pprint(graph.bfs_names(2))
# test error reporting
# pp.pprint(['Sao Paolo', 'Mountain View', 'San Francisco', 'London', 'Shanghai', 'Berlin'])

# Should print:
# Breadth First Search
# ['London', 'Shanghai', 'Berlin', 'Sao Paolo', 'Mountain View', 'San Francisco']

#  inspect.classify_class_attrs(Graph) is useful
#print inspect.classify_class_attrs(Graph)
#print Graph.__bases__   >>> (<type 'object'>)



#this is a single instance of Graph. That is why every time you call
#insert_edge, the value remains
"""
"""
graph1 = Graph()
graph1.insert_edge(100, 1, 2)
graph1.insert_edge(101, 1, 3)
graph1.insert_edge(102, 1, 4)
graph1.insert_edge(103, 3, 4)
# Should be [(100, 1, 2), (101, 1, 3), (102, 1, 4), (103, 3, 4)]
x = graph1.get_edge_list()
x.pop()
#print x

# Should be [None, [(2, 100), (3, 101), (4, 102)], None, [(4, 103)], None]
#print graph1.get_adjacency_list()
# Should be [[0, 0, 0, 0, 0], [0, 0, 100, 101, 102], [0, 0, 0, 0, 0], [0, 0, 0, 0, 103], [0, 0, 0, 0, 0]]
#print graph1.get_adjacency_matrix()
"""
def test(x=None):
    return x == True
print test() #shows false
print test(1) #shows true
    

class Vertex(object):
    def __init__(self, n):
        self.name = n
        self.neighbors = list()
        self.distance = 9999 #distance from first Node
        self.color = 'black' # black if not seen

    def add_neighbor(self, v):
        if v not in self.neighbors:#if vertexname notin neighbors list
            self.neighbors.append(v)#add vertex nmae to neighbors list
            self.neighbors.sort()#store neighbors as sorted list

class Graph(object):
    vertices = {} #key = 'name', value = Node object

    def add_vertex(self, vertex):
        if isinstance(vertex, Vertex) and vertex.name not in self.vertices:
            self.vertices[vertex.name] = vertex

    def add_edge(self, u, v): #u,v are letters at either end of edge
        if u in self.vertices and v in self.vertices: #make sure u,v are valid nodes
            for key, value in self.vertices.items():
                if key == u:
                    value.add_neighbor(v) #adding neighbors of each letter key
                    #to the value slot list in neighbor dict
                    
                if key == v:
                    value.add_neighbor(u)

    def print_graph(self):
        for key in sorted(list(self.vertices.keys())):
            print (key + str(self.vertices[key].neighbors) + " " + \
                   str(self.vertices[key].distance))
            print self.vertices

    def bfs(self, vert):
        q = list()
        vert.distance = 0 #distance from original node
        vert.color = "red" #red if seen
        
        for v in vert.neighbors: #neighbors of starting node
            self.vertices[v].distance = vert.distance + 1 #increase distance by 1
            q.append(v) #add node object to queue

        while len(q) > 0:
            u = q.pop(0) #one by one pop nodes off queue, u is letter name
            node_u = self.vertices[u] #node object
            node_u.color = 'red' #set to red b/c we're gonna visit that node

            for v in node_u.neighbors: #for each node object in node_u's neigh
                node_v = self.vertices[v] #get the neighbor node object 
                if node_v.color == 'black':#if neighbor not yet visited
                    q.append(v)# add to queue
                    if node_v.distance > node_u.distance + 1:
                        #update distance
                        node_v.distance = node_u.distance + 1

g = Graph()
a = Vertex('A')
g.add_vertex(a)
g.add_vertex(Vertex('B'))
for i in range(ord('A'), ord('K')):
    g.add_vertex(Vertex(chr(i)))

edges = ['AB', 'AE', 'BF', 'CG', 'DE', 'DH', 'EH', 'FG', 'FI', 'FJ','GJ', 'HI']
for edge in edges:
    g.add_edge(edge[:1], edge[1:])

g.bfs(a)
#g.print_graph()




class Vertex(object):

    def __init__(self, node):
        self.id = node
        self.adjacent = {}
        self.distance = sys.maxint
        self.visited = False
        self.previous = None

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight #makes neighbor dict for each vertex
        #form is neighbor name: value

    def get_connections(self):
        return self.adjacent.keys() #returns list of neighbor names

    def get_id(self):
        return self.id #returns name of current vertex

    def get_weight(self, neighbor):
        return self.adjacent[neighbor] #returns the value of a neighbor

    def set_distance(self, dist): #distance contains total weight of path
        self.distance = dist  #traveled so far

    def get_distance(self):
        return self.distance

    def set_previous(self, prev):
        self.previous = prev

    def set_visited(self):
        self.visited = True

   # def __str__(self):
#        return str(self.id) + ' adjacent: ' + \
#               str([x.id for x in self.adjacent])


class Graph(object):
    
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values()) #runs through vertex objects

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1 #node is just a name
        new_vertex = Vertex(node) #new vertex is a node object
        self.vert_dict[node] = new_vertex #node is a name, put node object
                            #as value
        #vert_dict is jus a map of name:node_object
        
        return new_vertex

    def get_vertex(self, n): #gets a vertex object
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost=0):
        if frm not in self.vert_dict:
            self.add_vertex(frm) #would make vert_dict[frm name] = frm
        if to not in self.vert_dict:
            self.add_vertex(to)
        #if the vertices are already in the dictionary, we add this edge
        #as a neigbor to the existing vertices
        #self.vert_dict frm name
        #self.vert_dict[to] is just a vertex object
        #therefore we are performing add_neighbor on (object, cost)
        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)
        #putting . after vert_dict[to] allows you to access all attributes
        #of object contained inside the value [to] In this case we are calling
        #add_neighbor, which takes as input neighbor and weight and adds
        #those values to a dictionary that is unique to the node
        #remember add_neighbor takes form (node_name, cost)

    def get_vertices(self): #returns list of vertex names
        print self.vert_dict
        return self.vert_dict.keys()

    def set_previous(self, current):
        self.previous = current

    def get_previous(self, current):
        return self.previous
    
    
def shortest(v, path):
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
    return

def search(graph, start):
    start.set_distance(0)

    #v in graph are graph objects
    unvisited_queue = [(v.get_distance(),v) for v in graph]
    heapq.heapify(unvisited_queue) #make it so you can use heap functions

    while len(unvisited_queue):
        uv = heapq.heappop(unvisited_queue) #uv = heap[0]
        current = uv[1]#vertex object
        current.set_visited()#set vertex object as visited

        for next in current.adjacent: #access adjacency list unique to vertex ob
            if next.visited:
                continue

            #new dist = dist of current vertex + weight of incoming 
            new_dist = current.get_distance() + current.get_weight(next)
            
            if new_dist < next.get_distance():
                next.set_distance(new_dist)
                next.set_previous(current)
                print 'updated : current = %s next = %s new_dist = %s' \
                        %(current.get_id(), next.get_id(), next.get_distance())
            else:
                print 'not updated : current = %s next = %s new_dist = %s' \
                        %(current.get_id(), next.get_id(), next.get_distance())

    while len(unvisited_queue):
        heapq.heappop(unvisited_queue)
        unvisited_queue = [(v.get_distance(),v) for v in graph if not v.visited]
        heapq.heapify(unvisited_queue)
        
g = Graph()
g.add_vertex('a')
g.add_vertex('b')
g.add_vertex('c')
g.add_vertex('d')
g.add_vertex('e')
g.add_vertex('f')

g.add_edge('a', 'b', 7)
g.add_edge('a', 'c', 9)
g.add_edge('a', 'f', 14)
g.add_edge('b', 'c', 10)
g.add_edge('b', 'd', 15)
g.add_edge('c', 'd', 11)
g.add_edge('c', 'f', 2)
g.add_edge('d', 'e', 6)
g.add_edge('e', 'f', 9)

#tuples_list = sorted([item for z in [x[e] for e in x.keys()] for item in z],\
#              key = lambda x: x[1])



