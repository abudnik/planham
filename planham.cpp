#include <iostream>
#include <iterator>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <set>
#include <cstring>
#include <unordered_set>
#include <algorithm>

class Graph
{
public:
	Graph(size_t n) {
		init(n);
	}

	Graph(const std::string &mat_file) {
		std::ifstream matrix(mat_file);
		if (!matrix.is_open()) {
			throw std::logic_error(std::string("Could not open: ") + mat_file);
		}

		size_t row_index = 0;
		std::string row;
		while (getline(matrix, row)) {
			std::istringstream iss(row);
			std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
					std::istream_iterator<std::string>{}};
			if (weights.empty()) {
				init(tokens.size());
			}
			for (size_t i = 0; i < tokens.size(); ++i) {
				weights[row_index][i] = std::atoi(tokens[i].c_str());
			}
			++row_index;
		}
	}

	Graph(const Graph &) = default;

	Graph(Graph &&other)
	: weights(std::move(other.weights)) {
	}

	void insert(size_t v1, size_t v2, int w) {
		weights[v1][v2] = weights[v2][v1] = w;
	}

	void remove(size_t v1, size_t v2) {
		weights[v1][v2] = weights[v2][v1] = 0;
	}

	void remove_vertex(size_t v) {
		weights.erase(weights.begin() + v);
		for (auto &vec : weights) {
			vec.erase(vec.begin() + v);
		}
	}

	int get_edge(size_t v1, size_t v2) const {
		return weights[v1][v2];
	}

	size_t get_num_vertices() const {
		return weights.size();
	}

	bool operator == (const Graph &other) const {
		for (size_t i = 0; i < get_num_vertices(); ++i) {
			if (!std::equal(weights[i].begin(), weights[i].end(), other.weights[i].begin()))
				return false;
		}
		return true;
	}

	void print_matlab() const {
		std::cout << "A = [";
		for (const auto &row : weights) {
			for (auto v : row) {
				std::cout << v << ' ';
			}
			std::cout << '\n';
		}
		std::cout << "];" << std::endl;
	}

private:
	void init(size_t n) {
		weights.resize(n);
		for (auto &row : weights) {
			row.resize(n);
		}
	}

private:
	std::vector<std::vector<int> > weights;

	friend class VertexAdjacent;
};

class VertexAdjacent
{
public:
	VertexAdjacent(const Graph &g, size_t v)
	: g(g),
	 v(v),
	 it(-1)
	{}

	bool next() {
		for (size_t i = it + 1; i < g.get_num_vertices(); ++i) {
			if (g.get_edge(v, i)) {
				it = i;
				return true;
			}
		}
		return false;
	}

	size_t get_vertex() const {
		return it;
	}

private:
	const Graph &g;
	size_t v;
	int64_t it;
};

std::vector<size_t> get_shortest_path(const Graph &g, size_t v1, size_t v2) {
	//std::cout << "get_shortest_path: " << v1+1 << ' ' << v2+1 << std::endl;
	std::vector<bool> visited(g.get_num_vertices());
	std::vector<size_t> prev(g.get_num_vertices());
	prev[v1] = v1;

	std::queue<size_t> q;
	q.push(v1);
	visited[v1] = true;
	while (!q.empty()) {
		const auto v = q.front(); q.pop();
		VertexAdjacent adj(g, v);
		while (adj.next()) {
			auto next = adj.get_vertex();
			if (visited[next]) continue;
			prev[next] = v;
			if (next == v2) {
				std::vector<size_t> result;
				while (prev[next] != next) {
					result.emplace_back(next);
					next = prev[next];
				}
				result.emplace_back(next);
				//std::cout << "get_shortest_path: " << v1+1 << ' ' << v2+1 << ' ' << result.size() << std::endl;
				//for (auto r : result) std::cout << r+1 << ' '; std::cout << std::endl;
				return result;
			}
			visited[next] = true;
			q.push(next);
		}
	}
	//std::cout << "get_shortest_path: " << v1+1 << ' ' << v2+1 << " not found!" << std::endl;
	return {};
}

bool is_singly_connected(const Graph &g) {
	std::vector<bool> visited(g.get_num_vertices());
	std::queue<size_t> q;
	size_t num_visited = 0;
	visited[0] = true;
	q.push(0);
	while (!q.empty()) {
		const auto v = q.front(); q.pop();
		++num_visited;
		VertexAdjacent adj(g, v);
		while (adj.next()) {
			auto next = adj.get_vertex();
			if (visited[next]) continue;
			visited[next] = true;
			q.push(next);
		}
	}
	return num_visited == g.get_num_vertices();
}

bool check_graph_invariants(const Graph &g) {
	if (g.get_num_vertices() < 3)
		return false;
	for (size_t v = 0; v < g.get_num_vertices(); ++v) {
		VertexAdjacent adj(g, v);
		size_t deg = 0;
		while (adj.next() && ++deg < 2);
		if (deg < 2)
			return false;
	}
	return is_singly_connected(g);
}

bool is_bipartite(const Graph &g, size_t &part1, size_t &part2) {
	std::vector<int> color(g.get_num_vertices());
	std::queue<size_t> q;
	color[0] = 1;
	part1 = 1;
	part2 = 0;
	q.push(0);
	while (!q.empty()) {
		const auto v = q.front(); q.pop();
		const int c = (color[v] == 1) ? 2 : 1;
		VertexAdjacent adj(g, v);
		while (adj.next()) {
			auto next = adj.get_vertex();
			if (color[next]) {
				if (color[next] == color[v]) {
					return false;
				} else {
					continue;
				}
			}
			if (c == 1) {
				++part1;
			} else {
				++part2;
			}
			color[next] = c;
			q.push(next);
		}
	}
	//std::cout << "is bipartite " << part1 << ' ' << part2 << std::endl;
	return true;
}

struct Face
{
	Face(std::vector<size_t> &&path)
	: vertices(std::move(path)),
	 uniq_vertices(vertices.begin(), vertices.end()) {
		std::sort(uniq_vertices.begin(), uniq_vertices.end());
	}

	bool operator < (const Face &other) const {
		if (vertices.size() == other.vertices.size()) {
			return std::lexicographical_compare(uniq_vertices.begin(), uniq_vertices.end(),
							    other.uniq_vertices.begin(), other.uniq_vertices.end());
		}
		return vertices.size() < other.vertices.size();
	}

	void print() const {
		for (auto v : vertices) std::cout << v+1 << ' ';
		std::cout << std::endl;
	}

	std::vector<size_t> vertices;
	std::vector<size_t> uniq_vertices;
};

class FaceMergingHam
{
public:
	FaceMergingHam(const Graph &g, bool hamiltonicity_only)
	: g(g),
	 dual(build()),
	 edge_to_faces(g.get_num_vertices()),
	 num_iter(0),
	 max_iter(0),
	 num_cycles(0),
	 hamiltonicity_only(hamiltonicity_only),
	 need_stop(false) {
	}

	void find_hamiltonian_cycles() {
		if (!check_graph_invariants(g)) {
			//std::cout << "graph isn't singly connected or have single vertices" << std::endl;
			return;
		}

		size_t part1, part2;
		if (is_bipartite(g, part1, part2) && part1 != part2) {
			//std::cout << "graph is bipartite and have different part sizes" << std::endl;
			return;
		}

		build_face_degree_and_edge_refcnt();

		Graph gr(g.get_num_vertices());
		std::vector<size_t> deg(g.get_num_vertices());
		std::vector<bool> chosen_faces(dual.get_num_vertices());

		face_to_faces_order.resize(dual.get_num_vertices());

		if (hamiltonicity_only) {
			max_iter = dual.get_num_vertices();
			while (!need_stop) {
				max_iter *= 2;
				//if (max_iter > dual.get_num_vertices() * 8)
				//  std::cout << "limit is " << max_iter << std::endl;
				for (size_t first_face = 0; first_face < dual.get_num_vertices(); ++first_face) {
					reorder_faces(first_face);
					//for (auto r : faces_order) std::cout << r+1 << ' '; std::cout << std::endl;
					if (faces_order.size() != dual.get_num_vertices()) {
						//std::cout << "dual graph isn't connected" << std::endl;
						return;
					}
					for (size_t i = 0; i < faces_order.size(); ++i) {
						face_to_faces_order[faces_order[i]] = i;
					}

					num_iter = 0;
					search_hamiltonian_cycles(gr, chosen_faces, 0, deg, 0);

					need_stop = need_stop || num_iter < max_iter;
					if (need_stop) {
						break;
					}
				}
			}
		} else {
			reorder_faces(0);
			//for (auto r : faces_order) std::cout << r+1 << ' '; std::cout << std::endl;
			if (faces_order.size() != dual.get_num_vertices()) {
				//std::cout << "dual graph isn't connected" << std::endl;
				return;
			}
			for (size_t i = 0; i < faces_order.size(); ++i) {
				face_to_faces_order[faces_order[i]] = i;
			}

			search_hamiltonian_cycles(gr, chosen_faces, 0, deg, 0);
		}
	}

	void print_stats() const {
		std::cout << "faces merging: num_iter: " << num_iter << ", num_cycles: " << num_cycles << std::endl;
	}

	size_t get_num_cycles() const {
		return num_cycles;
	}

private:
	void search_hamiltonian_cycles(Graph &gr, std::vector<bool> &chosen_faces, size_t index,
																 std::vector<size_t> &deg, size_t deg_sum) {
		if (hamiltonicity_only && num_iter > max_iter) {
			return;
		}

		++num_iter;

		//if (num_iter % 1000000 == 0) std::cout << "num_iter: " << num_iter << std::endl;

		if (index > 0) {
			const auto &face = faces[faces_order[index-1]];
			if (face_vertices_has_extra_edges(gr, face)) {
				//std::cout << "face_vertices_has_extra_edges" << std::endl;
				return;
			}
		}

		if (index == faces.size()) {
			if (deg_sum == 2 * g.get_num_vertices() && is_singly_connected(gr)) {
				/*std::cout << "found hamiltonian cycle" << std::endl;

				for (size_t i=0; i<g.get_num_vertices(); ++i) {
					VertexAdjacent adj(gr, i);
					while (adj.next()) {
						auto next = adj.get_vertex();
						if (next > i)
							std::cout << i+1 << ' ' << next+1 << '\n';
					}
				}
				std::cout << std::endl;*/

				//gr.print_matlab();

				/*for (bool b : chosen_faces)
					std::cout << b << ' ';
				std::cout << std::endl;*/

				/*size_t i=0;
				for (bool b : chosen_faces) {
					std::cout << b << ' ';
					if (++i % 3 == 0) {
						std::cout << '\n';
					}
				}
				std::cout << std::endl;*/
				need_stop = hamiltonicity_only;
				++num_cycles;
			}
			return;
		}

		if (!check_cycle_connectivity(gr, chosen_faces, index)) {
			/*for (bool b : chosen_faces) {
				std::cout << b << ' ';
			}
			std::cout << '\n';
			std::cout << "  " << index << ' ' << faces_order[index]+1 << std::endl;*/
			return;
		}

		const auto &face = faces[faces_order[index]];

		increase_edge_to_number_of_adjacent_faces(face, -1);

		apply_face(gr, face, deg, deg_sum, 1);
		chosen_faces[faces_order[index]] = true;
		if (!need_stop && check_face_vertices(face, deg)) {
			search_hamiltonian_cycles(gr, chosen_faces, index + 1, deg, deg_sum);
		}
		chosen_faces[faces_order[index]] = false;
		apply_face(gr, face, deg, deg_sum, -1);

		if (!need_stop && chosen_faces_contain_face_vertices(face)) {
			increase_vertex_to_number_of_adjacent_faces(face, -1);
			search_hamiltonian_cycles(gr, chosen_faces, index + 1, deg, deg_sum);
			increase_vertex_to_number_of_adjacent_faces(face, 1);
		}

		increase_edge_to_number_of_adjacent_faces(face, 1);
	}

	void increase_edge_to_number_of_adjacent_faces(const Face &f, int val) {
		for (size_t i = 0; i < f.vertices.size(); ++i) {
			auto v1 = f.vertices[i];
			auto v2 = f.vertices[(i+1) % f.vertices.size()];
			edge_to_faces.insert(v1, v2, edge_to_faces.get_edge(v1, v2) + val);
		}
	}

	void apply_face(Graph &g, const Face &f, std::vector<size_t> &deg, size_t &deg_sum, int step) {
		auto apply_edge = [&](size_t v1, size_t v2) {
			if (g.get_edge(v1, v2)) {
				g.remove(v1, v2);
				--deg[v1]; --deg[v2];
				deg_sum -= 2;
			} else {
				g.insert(v1, v2, 1);
				++deg[v1]; ++deg[v2];
				deg_sum += 2;
			}
		};

		for (size_t i = 0; i < f.vertices.size(); ++i) {
			auto v1 = f.vertices[i];
			auto v2 = f.vertices[(i+1) % f.vertices.size()];
			apply_edge(v1, v2);
		}
	}

	bool check_face_vertices(const Face &f, const std::vector<size_t> &deg) const {
		for (auto v : f.vertices) {
			if (!deg[v])
				return false;
		}
		return true;
	}

	bool face_vertices_has_extra_edges(const Graph &gr, const Face &f) const {
		for (auto v : f.vertices) {
			size_t num_stable_edges = 0;
			VertexAdjacent adj(gr, v);
			while (adj.next()) {
				auto next = adj.get_vertex();
				if (!edge_to_faces.get_edge(v, next)) {
					if (++num_stable_edges > 2)
						return true;
				}
			}
		}
		return false;
	}

	Graph build() {
		build_faces();

		Graph dual(faces.size());
		for (size_t i = 0; i < faces.size(); ++i) {
			for (size_t j = i + 1; j < faces.size(); ++j) {
				if (is_intersects(faces[i], faces[j])) {
					//std::cout << "dual edge: " << i+1 << ' ' << j+1 << std::endl;
					dual.insert(i, j, 1);
				}
			}
		}
		return dual;
	}

	void build_faces() {
		// Find all the shortest paths between two nodes in graph g using Floyd–Warshall algorithm
		std::vector<std::vector<size_t>> sp(g.get_num_vertices());

		for (size_t v = 0; v < g.get_num_vertices(); ++v) {
			sp[v].resize(g.get_num_vertices(), std::numeric_limits<size_t>::max() / 2);
			VertexAdjacent adj(g, v);
			while (adj.next()) {
				auto next = adj.get_vertex();
				sp[v][next] = 1;
			}
		}

		for (size_t k = 0; k < g.get_num_vertices(); ++k) {
			for (size_t i = 0; i < g.get_num_vertices(); ++i) {
				for (size_t j = 0; j < g.get_num_vertices(); ++j) {
					const auto d = sp[i][k] + sp[k][j];
					if (d < sp[i][j]) {
						sp[i][j] = d;
					}
				}
			}
		}

		Graph cg(g);
		auto apply_path = [&cg] (const std::vector<size_t> &path, bool remove) {
			for (size_t i = 0; i < path.size() - 1; ++i) {
				if (remove) {
					cg.remove(path[i], path[i+1]);
				} else {
					cg.insert(path[i], path[i+1], 1);
				}
			}
		};

		// Two vertex-disjoint shortest paths between any two nodes forms a cycle C.
		// Let V denote all vertices from cycle C. Let shortest_path_length(G, V1, V2) denote
		// length of shortest path between vertices V1 and V2 in some graph G.
		// If shortest_path_length(g, v1, v2) = shortest_path_length(C, v1, v2) for all v1, v2 ∈ V,
		// then C is a face of planar graph g.
		// NB! Complexity of this function is O(V^4). Use O(V + E) algorithm instead.
		std::set<Face> faces_set;
		for (size_t v1 = 0; v1 < g.get_num_vertices(); ++v1) {
			for (size_t v2 = v1 + 1; v2 < g.get_num_vertices(); ++v2) {
				auto path1 = get_shortest_path(cg, v1, v2);
				if (path1.size() < 2) continue;
				apply_path(path1, true);
				auto path2 = get_shortest_path(cg, v1, v2);
				apply_path(path1, false);
				if (path2.size() < 2) continue;

				bool is_shortest = true;
				for (size_t i = 1; i < path1.size() - 1 && is_shortest; ++i) {
					for (size_t j = 1; j < path2.size() - 1; ++j) {
						const auto d = std::min(i + j, path1.size()-1-i + path2.size()-1-j);
						if (sp[path1[i]][path2[j]] < d) {
							is_shortest = false;
							break;
						}
					}
				}

				if (is_shortest) {
					for (size_t i = path2.size() - 2; i >= 1; --i) {
						path1.emplace_back(path2[i]);
					}
					faces_set.emplace(std::move(path1));
				}
			}
		}
		faces.assign(faces_set.begin(), faces_set.end());
		/*for (const auto &f : faces) {
			f.print();
		}*/
		std::cout << "num_faces: " << faces.size() << std::endl;
	}

	bool is_intersects(const Face &f1, const Face &f2) const {
		auto same_edge = [](size_t v1_1, size_t v1_2, size_t v2_1, size_t v2_2) -> bool {
			if (v1_1 > v1_2) std::swap(v1_1, v1_2);
			if (v2_1 > v2_2) std::swap(v2_1, v2_2);
			return v1_1 == v2_1 && v1_2 == v2_2;
		};

		for (size_t i = 0; i < f1.vertices.size(); ++i) {
			for (size_t j = 0; j < f2.vertices.size(); ++j) {
				if (same_edge(f1.vertices[i], f1.vertices[(i+1) % f1.vertices.size()],
					     f2.vertices[j], f2.vertices[(j+1) % f2.vertices.size()]))
					return true;
			}
		}
		return false;
	}

	void reorder_faces(const size_t first_face) {
		if (memoized_faces_order.empty()) {
			memoized_faces_order.resize(dual.get_num_vertices());
		}

		if (!memoized_faces_order[first_face].empty()) {
			faces_order = memoized_faces_order[first_face];
			return;
		}

		faces_order.clear();
		faces_order.reserve(dual.get_num_vertices());

		std::vector<bool> visited(dual.get_num_vertices());
		std::queue<size_t> q;
		faces_order.emplace_back(first_face);
		visited[first_face] = true;
		q.push(first_face);
		while (!q.empty()) {
			const auto v = q.front(); q.pop();
			VertexAdjacent adj(dual, v);
			while (adj.next()) {
				auto next = adj.get_vertex();
				if (visited[next]) continue;
				visited[next] = true;
				faces_order.emplace_back(next);
				q.push(next);
			}
		}

		memoized_faces_order[first_face] = faces_order;
	}

	void build_face_degree_and_edge_refcnt() {
		vertex_to_faces.resize(g.get_num_vertices());
		for (const auto &face : faces) {
			for (size_t i = 0; i < face.vertices.size(); ++i) {
				auto v1 = face.vertices[i];
				auto v2 = face.vertices[(i+1) % face.vertices.size()];
				edge_to_faces.insert(v1, v2, edge_to_faces.get_edge(v1, v2) + 1);
				++vertex_to_faces[v1];
			}
		}
	}

	void increase_vertex_to_number_of_adjacent_faces(const Face &f, int val) {
		for (auto v : f.vertices)
			vertex_to_faces[v] += val;
	}

	bool chosen_faces_contain_face_vertices(const Face &f) const {
		for (auto v : f.vertices)
			if (vertex_to_faces[v] < 2)
				return false;
		return true;
	}

	bool check_cycle_connectivity(const Graph &gr, const std::vector<bool> &chosen_faces, size_t index) const {
		std::vector<bool> visited(dual.get_num_vertices());
		for (size_t i = 0; i < index; ++i) {
			if (visited[faces_order[i]]) continue;
			std::queue<size_t> q;
			std::vector<size_t> face_indexes;
			visited[faces_order[i]] = true;
			q.push(faces_order[i]);
			const bool chosen = chosen_faces[faces_order[i]];
			bool all_neighbor_visited = true;
			while (!q.empty() && all_neighbor_visited) {
				const auto v = q.front(); q.pop();
				face_indexes.emplace_back(v);
				VertexAdjacent adj(dual, v);
				while (adj.next()) {
					auto next = adj.get_vertex();
					if (visited[next]) continue;
					if (face_to_faces_order[next] < index) {
						if (chosen_faces[next] != chosen) continue;
					} else {
						all_neighbor_visited = false;
						break;
					}
					visited[next] = true;
					q.push(next);
				}
			}

			if (all_neighbor_visited) {
				std::unordered_set<size_t> vertices;
				std::set<std::pair<size_t, size_t> > edges;
				for (auto face_index : face_indexes) {
					const auto &f = faces[face_index];
					for (size_t j = 0; j < f.vertices.size(); ++j) {
						auto v1 = f.vertices[j];
						auto v2 = f.vertices[(j+1) % f.vertices.size()];
						vertices.insert(v1);
						if (gr.get_edge(v1, v2)) {
							if (v1 > v2) std::swap(v1, v2);
							edges.emplace(v1, v2);
						}
					}
				}
				if (edges.size() == vertices.size())
					return false;
			}
		}

		return true;
	}

private:
	const Graph &g;
	std::vector<Face> faces;
	Graph dual;
	Graph edge_to_faces;
	std::vector<size_t> vertex_to_faces;

	std::vector<std::vector<size_t>> memoized_faces_order;
	std::vector<size_t> face_to_faces_order;
	std::vector<size_t> faces_order;

	uint64_t num_iter;
	uint64_t max_iter;
	uint64_t num_cycles;
	const bool hamiltonicity_only;
	bool need_stop;
};

class BruteForceHam
{
public:
	BruteForceHam(const Graph &g, bool hamiltonicity_only)
	: g(g),
	 visited(g.get_num_vertices()),
	 path(g.get_num_vertices()),
	 num_iter(0),
	 num_cycles(0),
	 hamiltonicity_only(hamiltonicity_only),
	 need_stop(false)
	{}

	void find_hamiltonian_cycles() {
		if (check_graph_invariants(g))
			search_hamiltonian_cycles(0, 1);
	}

	void print_stats() const {
		std::cout << "trivial: num_iter: " << num_iter << ", num_cycles: " << num_cycles << std::endl;
	}

	size_t get_num_cycles() const {
		return num_cycles;
	}

private:
	void search_hamiltonian_cycles(size_t v, size_t depth) {
		++num_iter;
		path[depth - 1] = v;

		if (depth == g.get_num_vertices()) {
			//std::cout << "found hamiltonian path" << std::endl;

			VertexAdjacent adj(g, v);
			while (adj.next()) {
				auto next = adj.get_vertex();
				if (next == path.front()) {
					++num_cycles;
					if (hamiltonicity_only) {
						need_stop = true;
						break;
					}
					/*std::cout << "found hamiltonian cycle: ";
					for (auto p : path) {
						std::cout << p+1 << ", ";
					}
					std::cout << next+1 << std::endl;*/
				}
			}
			return;
		}

		visited[v] = true;
		VertexAdjacent adj(g, v);
		while (!need_stop && adj.next()) {
			auto next = adj.get_vertex();
			if (!visited[next]) {
				search_hamiltonian_cycles(next, depth + 1);
			}
		}
		visited[v] = false;
	}

private:
	const Graph &g;
	std::vector<bool> visited;
	std::vector<size_t> path;
	size_t num_iter, num_cycles;
	bool hamiltonicity_only, need_stop;
};

Graph create_nxn_squares(size_t n) {
	/*
	  1--2--3--4
	  |  |  |  |
	  5--6--7--8
	  |  |  |  |
	  9--10-11-12
	  |  |  |  |
	  13-14-15-16
	 */
	++n;
	Graph g(n * n);
	for (size_t v1 = 0; v1 < n * n - 1; ++v1) {
		if ((v1+1) % n != 0) {
			g.insert(v1, v1 + 1, 1);
		}
	}
	for (size_t v1 = 0; v1 < n * (n - 1); ++v1) {
		g.insert(v1, v1 + n, 1);
	}
	return g;
}

Graph create_nxn_triangles(size_t n) {
	/*
	  1--2--3--4
	  | \| \| \|
	  5--6--7--8
	  | \| \| \|
	  9--10-11-12
	  | \| \| \|
	  13-14-15-16
	 */
	++n;
	Graph g(n * n);
	for (size_t v1 = 0; v1 < n * n - 1; ++v1) {
		if ((v1+1) % n != 0) {
			g.insert(v1, v1 + 1, 1);
		}
	}
	for (size_t v1 = 0; v1 < n * (n - 1); ++v1) {
		g.insert(v1, v1 + n, 1);
		if ((v1+1) % n != 0) {
			g.insert(v1, v1 + n + 1, 1);
		}
	}
	return g;
}

Graph create_nxn_hexagonal(size_t n) {
	/*
	 01--02  03--04  05
	 |   |   |   |
	 06  07--08  09--10
	 |   |   |   |   |
	 11--12  13--14  15
	 |   |   |   |   |
	 16  17--18  19--20
	 |   |   |   |
	 21--22  23--24  25
	 */
	n *= 5;
	Graph g(n * n);
	for (size_t i = 0; i < n; ++i) {
		bool has = (i % 2 == 0);
		for (size_t j = 0; j < n - 1; ++j) {
			auto v = j * n + i;
			g.insert(v, v + n, 1); // vertical
			if (has) {
				v = i * n + j;
				g.insert(v, v + 1, 1); // horizontal
			}
			has = !has;
		}
	}
	g.remove_vertex(g.get_num_vertices() - 1); // 25
	g.remove_vertex(n - 1); // 05
	return g;
}

Graph create_random(size_t v, size_t e) {
	Graph g(v);
	double p = 2.0 * e / v / (v - 1);
	for (size_t i = 0; i < v; ++i)
		for (size_t j = 0; j < i; ++j)
			if (rand() < p * RAND_MAX)
				g.insert(i, j, 1);

	size_t num_edges = 0;
	for (size_t v = 0; v < g.get_num_vertices(); ++v) {
		VertexAdjacent adj(g, v);
		while (adj.next()) ++num_edges;
	}
	std::cout << "num_edges: " << num_edges << std::endl;
	return g;
}

Graph create_petersen() {
	Graph g(10);
	g.insert(0, 2, 1);
	g.insert(2, 1, 1);
	g.insert(1, 3, 1);
	g.insert(3, 6, 1);
	g.insert(6, 0, 1);
	g.insert(0, 9, 1);
	g.insert(4, 2, 1);
	g.insert(8, 1, 1);
	g.insert(3, 5, 1);
	g.insert(6, 7, 1);
	g.insert(7, 4, 1);
	g.insert(7, 8, 1);
	g.insert(5, 4, 1);
	g.insert(5, 9, 1);
	g.insert(8, 7, 1);
	g.insert(8, 9, 1);
	g.insert(4, 5, 1);
	g.insert(4, 7, 1);
	g.insert(9, 8, 1);
	g.insert(9, 5, 1);
	return g;
}

Graph create_star() {
	Graph g(6);
	g.insert(0, 1, 1);
	g.insert(1, 2, 1);
	g.insert(2, 3, 1);
	g.insert(3, 4, 1);
	g.insert(4, 5, 1);
	g.insert(5, 0, 1);
	g.insert(0, 2, 1);
	g.insert(2, 4, 1);
	g.insert(4, 0, 1);
	return g;
}

void find_hamiltonian_cycle(const Graph &g) {
	FaceMergingHam fm(g, true);
	fm.find_hamiltonian_cycles();
	fm.print_stats();
	if (!fm.get_num_cycles()) {
		std::exit(1);
	}
}

void test(const Graph &g) {
	const bool hamiltonicity_only = true;

	FaceMergingHam fm(g, hamiltonicity_only);
	fm.find_hamiltonian_cycles();
	fm.print_stats();

	BruteForceHam bf(g, hamiltonicity_only);
	bf.find_hamiltonian_cycles();
	bf.print_stats();

	if (fm.get_num_cycles() != bf.get_num_cycles()) {
		std::cout << "bug" << std::endl;
		std::exit(1);
	}
}

void combinatorial_test(size_t n) {
	Graph g = create_nxn_squares(n);
	++n; n *= n;
	const uint64_t num_test = 1ULL << n;
	for (uint64_t i = 0; i < num_test; ++i) {
		Graph gc(g);
		for (uint64_t j = n - 1; ; --j) {
			const bool t = (i & (1 << j)) != 0;
			//std::cout << t;
			if (!t) gc.remove_vertex(j);
			if (j == 0) break;
		}
		//std::cout << '\n';
		test(gc);
		std::cout << "test " << i+1 << '/' << num_test << std::endl;
	}
}

void combinatorial_test2(size_t n) {
	Graph g = create_nxn_squares(n);
	++n; n *= n;
	const uint64_t num_test = 1ULL << n;
	for (uint64_t i = 0; i < num_test; ++i) {
		Graph gc(g);
		for (uint64_t j = n - 1; ; --j) {
			const bool t = (i & (1 << j)) != 0;
			//std::cout << t;
			if (!t) gc.remove(i % n, j);
			if (j == 0) break;
		}
		//std::cout << '\n';
		test(gc);
		std::cout << "test " << i+1 << '/' << num_test << std::endl;
	}
}

void random_test(size_t n, size_t r, size_t cnt) {
	Graph g = create_nxn_triangles(n);
	++n; n *= n;

	for (size_t i = 0; i < cnt; ++i) {
		Graph gc(g);
		const auto saved = n;
		for (size_t j = 0; j < r; ++j) {
			const int v = rand() % n;
			n--;
			std::cout << v << ' ';
			gc.remove_vertex(v);
		}
		n = saved;
		std::cout << std::endl;
		test(gc);
		std::cout << "test " << i+1 << '/' << cnt << std::endl;
	}
}

int main(int argc, char* argv[])
{
	if (argc > 2) {
		if (!strcmp(argv[1], "-t")) {
			test(Graph(argv[2]));
		} else if (!strcmp(argv[1], "-f")) {
			find_hamiltonian_cycle(Graph(argv[2]));
		} else {
			std::cout << "unknown opt: " << argv[1] << std::endl;
		}
		return 0;
	}

	srand(time(nullptr));
	const size_t n = 4;
	//test(create_nxn_squares(n));
	//test(create_nxn_triangles(n));
	//test(create_nxn_hexagonal(n));
	//test(create_random(n, n * 3));
	//test(create_petersen());
	//test(create_star());
	combinatorial_test(n);
	//random_test(n, 10, 1000);

	std::cout << "done" << std::endl;
	return 0;
}
