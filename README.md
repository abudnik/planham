## Introduction.
An algorithm for finding Hamiltonian cycles in undirected planar graph, presented in this article, is based on an assumption, that the following condition works for every connected planar graph: graph G is hamiltonian if and only if there is a subset of faces of G, whose merging forms a Hamiltonian cycle. Merging of faces is an operation of symmetric difference of sets of edges, where each particular set of edges contains all edges of a face. For example, merging of all inner faces of some plane graph forms an external face.

Euler's formula states that if graph G is a finite, connected, planar graph, and V is the number of vertices, E is the number of edges and F is the number of faces (including the external one), then: `V - E + F = 2`. In a finite, connected, simple, planar graph, any face (except possibly the external one) is bounded by at least three edges and every edge touches at most two faces; using Euler's formula, one can then show that: `E ≤ 3 * V - 6`. If the right-hand side substitutes E in Euler's formula, we finally obtain the maximum number of faces in a planar graph: `F ≤ 2 * V - 4`. Algorithm that tries to merge every possible subset of faces has exponential complexity `O(2^F)`, in the worst case is `O(2^(2*V))`. Finding a Hamiltonian cycle in a planar graph is proven to be an NP-Complete problem [1]. In this article we describe an approach based on backtracking, that drastically reduces the search space, and where the algorithm execution time is proportional to the number of faces for many instances of planar graphs.

## A formal description of the problem.
Let's give a formal description of the problem. Let G be a planar 2-edge-connected unordered graph. Let Gf be a graph in which every vertex corresponds to one and only one face of G, except a single external face. Any two vertices v1 and v2 of Gf are connected by unordered edge if and only if corresponding faces f1 and f2 of G have at least one common edge. In other words, Gf is a dual graph of a plane graph G without parallel edges and without a single vertex corresponding to an external face. Every subset Vs of vertices of Gf corresponds to equipotent subset Fs of faces of G. The goal is to find such a nonempty subset of vertices Vs, for which merge of corresponding set of faces Fs forms a Hamiltonian cycle of G. <br />
Let's give a formal description of the elements, used in solving the given problem. The subgraph formed by removing the vertices of the graph Gf, which are not included in the subset Vs of vertices of the graph Gf, is denoted by Gs. The subgraph formed by removing the vertices of the graph Gf, which are included in the subset Vs of vertices of the graph Gf, is denoted by Gc. Graph Gr is formed by merging faces of graph G, where each face corresponds to a single vertex of graph Gs. Graph Gt is formed by merging faces of graph G, where each face corresponds to a single vertex of graph Gc. <br />
Let I be a subset of the Cartesian product of sets of vertices of G and Gf. The pair `(v1, v2) ∈ I` if and only if `v1 ∈ G` belongs to a face F, where F corresponds to a `v2 ∈ Gf`. <br />
Let's give the necessary conditions for the existence of a Hamiltonian cycle in graph G, that are checked at each iteration of the search algorithm: <br />
C1. For each vertex `v1 ∈ G` there is a vertex `v2 ∈ Gs`, that `(v1, v2) ∈ I`. <br />
C2. Graph Gs must be connected. <br />
C3. Let St be the set of all edges of some connectivity component of graph Gt. Let Sr be a set of all edges of graph Gr. Consequently: `St \ Sr ≠ ∅`. <br />
C4. All vertices of graph Gr must have degree 2.

## Description of the algorithm.
Algorithm implementation in pseudocode:

```Python
def find_hamiltonian_cycle_in_unordered_planar_graph(g):
    if not check_graph_invariants(g):
        print "Graph is not Hamiltonian"
        return

    faces = get_planar_graph_faces(g)
    # build Gf graph
    dual = build_dual_graph(faces)
    # check that Gf is connected
    if dual.get_num_connectivity_components() > 1:
        print "Graph is not Hamiltonian"
        return

    faces_order = reorder_faces(faces, dual, first_face=0)
    # face_to_faces_order mapping: face -> index of face in faces_order,
    # e.g. if first_face is 0 and faces[0] is adjacent only to faces[5],
    # then first two items in faces_order list is [0, 5] and
    # (face_to_faces_order[0] == 0) and (face_to_faces_order[5] == 1)
    face_to_faces_order = {}
    for i in range(len(faces_order)):
        face_to_faces_order[faces_order[i]] = i

    # every vertex in G belongs to multiple faces,
    # vertex_to_faces mapping: vertex -> number of adjacent to this vertex faces
    vertex_to_faces = vertex_to_number_of_adjacent_faces(faces)
    # every edge in G belongs to one or two faces,
    # edge_to_faces mapping: edge -> number of adjacent to this edge faces
    edge_to_faces = edge_to_number_of_adjacent_faces(faces)

    # create Gr graph, that will contain Hamiltonian cycle, if G is Hamiltonian
    chosen_faces = [False] * len(faces)
    gr = Graph(g.get_num_vertices())

    if search_hamiltonian_cycle(g, gr, dual, chosen_faces, faces,
                                faces_order, face_to_faces_order,
                                vertex_to_faces, edge_to_faces, 0):
        return gr

def search_hamiltonian_cycle(g, gr, dual, chosen_faces, faces,
                             faces_order, face_to_faces_order,
                             vertex_to_faces, edge_to_faces, index):
    # check C4 condition
    if index > 0 and face_vertices_has_extra_edges(gr, edge_to_faces, faces[faces_order[index - 1]]):
        return False

    # base case
    if index == len(faces):
        is_hamiltonian = gr.get_num_edges() == g.get_num_vertices() and gr.get_num_connectivity_components() == 1
        return is_hamiltonian

    # check C2 and C3 conditions
    if has_isolated_faces_component(gr, dual, chosen_faces, faces, faces_order, face_to_faces_order, index):
        return False

    face_index = faces_order[index]
    increase_edge_to_number_of_adjacent_faces(edge_to_faces, faces[face_index], -1)

    # First branch - current face is chosen. Current face is faces[face_order[index]].
    # Only chosen faces used in faces merging.
    chosen_faces[face_index] = True
    merge_face(gr, faces[face_index], True)
    # Merging current face with adjacent faces shouldn't leave a single vertex without adjacent edges
    if check_face_vertices(gr, faces[index]):
        if search_hamiltonian_cycle(g, gr, dual, chosen_faces, faces,
                                    faces_order, face_to_faces_order,
                                    vertex_to_faces, edge_to_faces, index + 1):
            return True
    merge_face(gr, faces[face_index], False)
    chosen_faces[face_index] = False

    # Second branch - current face is not chosen
    increase_vertex_to_number_of_adjacent_faces(vertex_to_faces, faces[face_index], -1)
    # check C1 condition: every vertex must belong at least to one chosen face
    if chosen_faces_contain_face_vertices(vertex_to_faces, faces[face_index]):
        if search_hamiltonian_cycle(g, gr, dual, chosen_faces, faces,
                                    faces_order, face_to_faces_order,
                                    vertex_to_faces, edge_to_faces, index + 1):
            return True
    increase_vertex_to_number_of_adjacent_faces(vertex_to_faces, faces[face_index], 1)

    increase_edge_to_number_of_adjacent_faces(edge_to_faces, faces[face_index], 1)

def chosen_faces_contain_face_vertices(vertex_to_faces, face):
    for v in face.get_vertices():
        if vertex_to_faces[v] == 0:
            return False
    return True

def check_face_vertices(gr, face):
    for v in face.get_vertices():
        if len(gr.get_adjacent_edges(v)) == 0:
            return False
    return True

def merge_face(gr, face, chosen):
    for e in face.get_edges():
        if chosen:
            gr.insert_edge(e)
        else:
            gr.remove_edge(e)

def increase_edge_to_number_of_adjacent_faces(edge_to_faces, face, val):
    for e in face.get_edges():
        edge_to_faces[e] += val

def increase_vertex_to_number_of_adjacent_faces(vertex_to_faces, face, val):
    for v in face.get_vertices():
        vertex_to_faces[v] += val

def face_vertices_has_extra_edges(gr, edge_to_faces, face):
    for v in face.get_vertices():
        num_stable_edges = 0
        for edge in gr.get_adjacent_edges(v):
            if edge_to_faces[edge] == 0:
                num_stable_edges += 1
                if num_stable_edges > 2:
                    return True
    return False

def has_isolated_faces_component(gr, dual, chosen_faces, faces, faces_order, face_to_faces_order, index):
    # If connected subset of faces S, where each face has the same property of being chosen or not,
    # hasn't adjacent unvisited faces, then subsequent iterations of search_hamiltonian_cycle()
    # will not affect this subset S. If merging faces of S forms a cycle including all vertices
    # of all faces in S, then Gr is not Hamiltonian.
    visited = [False] * len(dual)
    for i in range(index):
        if visited[faces_order[i]]:
            continue
        faces_component = []
        visited[faces_order[i]] = True
        q = [faces_order[i]]
        all_neighbor_visited = True
        while q and all_neighbor_visited:
            v = q.pop(0)
            faces_component.append(v)
            for adj in dual.get_adjacent_vertices(v):
                if not visited[adj]:
                    if face_to_faces_order[adj] < index:
                        if chosen_faces[adj] != chosen_faces[faces_order[i]]:
                            continue
                    else:
                        all_neighbor_visited = False
                        break
                    visited[adj] = True
                    q.append(adj)

        if all_neighbor_visited:
            vertices = set()
            edges = set()
            for face_index in faces_component:
                vertices.update(faces[face_index].get_vertices())
                for edge in faces[face_index].get_edges():
                    if gr.has_edge(edge):
                        edges.add(edge)
            has_isolated_component = len(vertices) == len(edges)
            if has_isolated_component:
                return True

    return False

def edge_to_number_of_adjacent_faces(faces):
    edge_to_faces = {}
    for f in faces:
        for e in f.get_edges():
            edge_to_faces[e] += 1
    return edge_to_faces

def vertex_to_number_of_adjacent_faces(faces):
    vertex_to_faces = {}
    for f in faces:
        for v in f.get_vertices():
            vertex_to_faces[v] += 1
    return vertex_to_faces

def reorder_faces(faces, dual, first_face):
    # BFS in dual graph (Gf) will visit vertices (i.e. faces) in some order.
    # Get order of faces accordingly to BFS traversal of dual graph (Gf).
    faces_order = []
    visited = [False] * dual.get_num_vertices()
    q = [first_face]
    while q:
        v = q.pop(0)
        for adj in dual.get_adjacent_vertices(v):
            if not visited[adj]:
                visited[adj] = True
                q.append(adj)
                faces_order.append(adj)
    return faces_order

def build_dual_graph(faces):
    n = len(faces)
    dual = Graph(n)
    for i in range(n):
        for j in range(i+1, n):
            if is_faces_intersects(faces[i], faces[j]):
                g.insert_edge(i, j)
    return dual

def is_faces_intersects(f1, f2):
    for edge1 in f1.get_edges():
        for edge2 in f2.get_edges():
            if same_edge(edge1, edge2):
                return True
    return False

def get_planar_graph_faces(g):
    planar_embedding = boyer_myrvold_planarity_test(g)
    return planar_face_traversal(g, planar_embedding, exclude_external_face=True)

def check_graph_invariants(g):
    # check that g is 2-edge-connected graph
    if g.get_num_vertices() < 3 or g.get_minimum_vertex_degree() < 2 or g.get_num_connectivity_components() > 1:
        return False
    # if graph g is bipartite, check that both parts have the same number of vertices 
    is_bipartite, num_vert_part1, num_vert_part2 = g.is_bipartite()
    if is_bipartite and num_vert_part1 != num_vert_part2:
        return False
    return True
```

Let's demonstrate steps of the algorithm on an example of 3x3 grid graph, consisting of 9 faces, 16 vertices and 24 edges:

```
1--2--3--4
|  |  |  |
5--6--7--8
|  |  |  |
9--10-11-12
|  |  |  |
13-14-15-16
```

Search of planar faces (function get_planar_graph_faces) gives 9 faces: 1-2-6-5, 2-3-7-6, etc. Search of the faces traversal order in accordance with breadth-first search starting at face 1-2-6-5 through adjacent faces (function reorder_faces) gives the following sequence:

```
*--*--*--*
|1 |2 |4 |
*--*--*--*
|3 |5 |7 |
*--*--*--*
|6 |8 |9 |
*--*--*--*
```

The further step is to recursively search through all planar faces in the order obtained in the previous step, i.e. first considered face 1, then adjacent faces 2 and 3, etc. At each step of the algorithm current face can be either chosen for merging with previous chosen faces or not. If face is chosen, then it is labeled with its own order number, otherwise its labeled with a dash '-'. If a face is not yet considered by the algotihm, then this face is not labeled. Let's assume that faces 1 and 2 are chosen, face 3 is not chosen and all remaining faces are not yet been considered by the algorithm, then we obtain following graph diagram:

```
*--*--*--*
|1 |2 |  |
*--*--*--*
|- |  |  |
*--*--*--*
|  |  |  |
*--*--*--*
```

For the above-described example with the chosen faces 1 and 2, graph Gr contains a cycle formed by merging faces 1 and 2: 1-2-3-7-6-5.

Let's demonstrate steps of the recursive search algorithm (function search_hamiltonian_cycle). At each step, the algorithm tries to merge i-th face with previously chosen faces. If search of the Hamiltonian cycle for subsequent faces is not succeeded, then i-th faces is marked as not being chosen and search of the Hamiltonian cycle is continued from the next (i+1)-th face. In the example with 3x3 grid graph, the algorithm choses faces 1, 2, 3 and 4 for merging during the first four steps. When the algorithm reviews 5th face, then it detects that this face could not being chosen for merging, since otherwise, the vertex 6 remains without edges in graph Gr. This case is checked by check_face_vertices function. We get the following graph diagram after 5th step:

```
*--*--*--*
|1 |2 |4 |
*--*--*--*
|3 |- |  |
*--*--*--*
|  |  |  |
*--*--*--*
```

Next, faces 6 and 7 merge:

```
*--*--*--*
|1 |2 |4 |
*--*--*--*
|3 |- |7 |
*--*--*--*
|6 |  |  |
*--*--*--*
```

When face 8 is being chosen for merging, then condition C3 is violated: previously non-chosen face 5 forms an isolated component consisting of one face, that generates cycle 6-7-11-10, which excludes further successful attempts to find the Hamiltonian cycle. This case is checked by has_isolated_faces_component function. Then, the algorithm backtracks at step 8, where face 8 is not being chosen for merging:

```
*--*--*--*
|1 |2 |4 |
*--*--*--*
|3 |- |7 |
*--*--*--*
|6 |- |  |
*--*--*--*
```

Next, face 9 merges that finally gives a Hamiltonian cycle 1-2-3-4-8-12-16-15-11-7-6-10-14-13-9-5:

```
*--*--*--*
|1 |2 |4 |
*--*--*--*
|3 |- |7 |
*--*--*--*
|6 |- |9 |
*--*--*--*
```

## Discussion.
As a result, described algorithm has exponential complexity: various subsets of faces are reviewed. However, during the space search, many branches are cut off, because they cannot lead to finding a Hamiltonian cycle. This branch cuts significantly reduce the state space. Taking into account all possible planar graphs, detailed analysis of the algorithm complexity is not provided in this article. Therefore, exact complexity of the algorithm and examples of the worst cases are open questions. <br />
Check of the correctness and effectiveness of the algorithm implementation has been carried out experimentally. To check the correctness of the described algorithm, its results have been compared with results of trivial brute-force algorithm. Both algorithms must give the same results concerning Hamiltonicity of a graph. This check has been performed for the class of graphs obtained by removing all the possible subsets of the MxN grid graph, as well as for specific cases of other planar graphs. The same test has been used to test the effectiveness of the algorithm. <br />
We obtained the following results during above-mentioned tests: algorithm execution time is proportional to the number of faces for most instances of graphs, while approximately 1/40 of the remaining instances of graphs are processed with exponential execution time. Next, we can evaluate the constant C in the search algorithm with C^F complexity, where F - number of faces: `C = num_iter ^ (1/F)`, where num_iter - the obtained number of iterations for some input graph. We obtained the maximum value of `C = 1.3608`. Thus, a rough estimate of the algorithm complexity is `O(1.3608^F)` in the worst case. <br />
It is necessary to take into account, that the total number of iterations of the algorithm to the great extent depends on the order of faces. It is possible to interrupt execution of the algorithm in case of exceeding a certain limit of the number of iterations and to start over the search using another initial face. Thereby, you can restart algorithm at most (F - 1) times, but that does not eliminate completely the worst-case scenario.

## References

[1] M.R. Garey, D. S. Johnson and R. E. Tarjan, The planar Hamilton circuit problem is NP-complete, SIAM J. Comp., 5 (1976), pp. 704–714 <br />
[2] A. Itai, C.H. Papadimitriou, J.L. Szwarcfiter, Hamiltonian paths in grid graphs, SIAM J. Comput., 11 (1982) pp. 676–686. <br />
[3] E. M. Arkin, S. P. Fekete, K. Islam [et al.], Not being (super)thin or solid is hard: A study of grid Hamiltonicity, Comput. Geom., 42, No. 6–7, 582–605, 2009. <br />
[4] B. Vandegriend. Finding Hamiltonian Cycles: Algorithms, Graphs and Performance. Master’s thesis, Univ. of Alberta, Dept. of Computing Science, 1998. <br />
[5] C. Umans and W. Lenhart, Hamiltonian cycles in solid grid graphs, in Proc. 38th IEEE Symp. Foundations Comput. Sci., Miami Beach, FL, USA, Oct. 20–22, 1997, pp. 496–507, IEEE Comput. Soc., Los Alamitos, USA, 1997. <br />
[6] B. Yao, C. Ras, H. Mokhtar, An algorithm for finding Hamiltonian Cycles in Cubic Planar Graphs, 11 (2015), arXiv:1512.01324