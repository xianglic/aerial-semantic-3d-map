// Parallel BFS hole filling using ParlayLib.
// On-the-fly neighbor queries per frontier vertex — avoids precomputing
// the full adjacency graph over all 5M points.
//
// Binary protocol (stdin -> stdout):
//   IN:  int64 N, int64 n_seeds, int64 max_iters, float64 radius, float64 color_thresh,
//        float64[3] ref_color, float32[N*6] xyzrgb, int32[n_seeds] seed_indices
//   OUT: int64 n_result, int32[n_result] result_indices

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <parlay/primitives.h>
#include <parlay/sequence.h>

using vertex = int32_t;

// ---------------------------------------------------------------------------
// CSR spatial grid — read-only after build, safe for parallel queries
// ---------------------------------------------------------------------------
struct Grid {
    float cell, ox, oy, oz;
    int64_t nx, ny, nz;
    parlay::sequence<vertex>  sorted;   // point indices sorted by cell id
    parlay::sequence<int32_t> offsets;  // offsets[c]..offsets[c+1] = points in cell c

    int64_t cell_id(float x, float y, float z) const {
        int64_t cx = (int64_t)((x - ox) / cell);
        int64_t cy = (int64_t)((y - oy) / cell);
        int64_t cz = (int64_t)((z - oz) / cell);
        return cx * ny * nz + cy * nz + cz;
    }

    void build(const float* pts, int N, float radius) {
        cell = radius;
        ox = oy = oz =  1e38f;
        float mx = -1e38f, my = -1e38f, mz = -1e38f;
        for (int i = 0; i < N; i++) {
            ox = std::min(ox, pts[i*6]);   mx = std::max(mx, pts[i*6]);
            oy = std::min(oy, pts[i*6+1]); my = std::max(my, pts[i*6+1]);
            oz = std::min(oz, pts[i*6+2]); mz = std::max(mz, pts[i*6+2]);
        }
        nx = (int64_t)((mx - ox) / cell) + 2;
        ny = (int64_t)((my - oy) / cell) + 2;
        nz = (int64_t)((mz - oz) / cell) + 2;
        int64_t total_cells = nx * ny * nz;

        fprintf(stderr, "Grid: bbox %.3f x %.3f x %.3f | cells %lld x %lld x %lld = %lld | cell_size=%.4f\n",
                mx-ox, my-oy, mz-oz, (long long)nx, (long long)ny, (long long)nz,
                (long long)total_cells, cell);

        auto ids = parlay::tabulate(N, [&](int i) {
            return cell_id(pts[i*6], pts[i*6+1], pts[i*6+2]);
        });
        sorted = parlay::tabulate(N, [](int i) { return (vertex)i; });
        parlay::integer_sort_inplace(sorted, [&](vertex i) { return (size_t)ids[i]; });

        offsets = parlay::sequence<int32_t>(total_cells + 1, N);
        parlay::parallel_for(0, N, [&](int i) {
            if (i == 0 || ids[sorted[i]] != ids[sorted[i-1]])
                offsets[ids[sorted[i]]] = i;
        });
        for (int64_t c = total_cells - 1; c >= 0; c--)
            if (offsets[c] == N) offsets[c] = offsets[c + 1];
    }

    parlay::sequence<vertex> neighbors(const float* pts, vertex idx, float radius) const {
        float x = pts[idx*6], y = pts[idx*6+1], z = pts[idx*6+2];
        float r2 = radius * radius;
        int64_t cx = (int64_t)((x - ox) / cell);
        int64_t cy = (int64_t)((y - oy) / cell);
        int64_t cz = (int64_t)((z - oz) / cell);

        parlay::sequence<vertex> result;
        for (int64_t dx = -1; dx <= 1; dx++)
        for (int64_t dy = -1; dy <= 1; dy++)
        for (int64_t dz = -1; dz <= 1; dz++) {
            int64_t c = (cx+dx)*ny*nz + (cy+dy)*nz + (cz+dz);
            if (c < 0 || c >= nx*ny*nz) continue;
            int32_t s = offsets[c], e = offsets[c+1];
            for (int32_t k = s; k < e; k++) {
                vertex j = sorted[k];
                float ddx = pts[j*6]-x, ddy = pts[j*6+1]-y, ddz = pts[j*6+2]-z;
                if (ddx*ddx + ddy*ddy + ddz*ddz <= r2)
                    result.push_back(j);
            }
        }
        return result;
    }
};

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------
template<typename T> void read_val(T& v) {
    if (fread(&v, sizeof(T), 1, stdin) != 1) { fprintf(stderr, "read error\n"); exit(1); }
}
template<typename T> void read_arr(T* buf, size_t n) {
    if (fread(buf, sizeof(T), n, stdin) != n) { fprintf(stderr, "read error\n"); exit(1); }
}
template<typename T> void write_val(T v)                   { fwrite(&v, sizeof(T), 1, stdout); }
template<typename T> void write_arr(const T* b, size_t n)  { fwrite(b, sizeof(T), n, stdout); }

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    int64_t N, n_seeds, max_iters;
    double radius_d, color_thresh_d, ref_r, ref_g, ref_b;
    read_val(N); read_val(n_seeds); read_val(max_iters);
    read_val(radius_d); read_val(color_thresh_d);
    read_val(ref_r); read_val(ref_g); read_val(ref_b);

    float radius       = (float)radius_d;
    float color_thresh = (float)color_thresh_d;
    float ct2          = color_thresh * color_thresh;
    (void)ref_r; (void)ref_g; (void)ref_b;

    std::vector<float> pts_buf(N * 6);
    read_arr(pts_buf.data(), N * 6);
    const float* pts = pts_buf.data();

    std::vector<vertex> seeds_vec(n_seeds);
    read_arr(seeds_vec.data(), n_seeds);

    fprintf(stderr, "BFS: N=%ld seeds=%ld radius=%.4f color_thresh=%.4f max_iters=%ld\n",
            (long)N, (long)n_seeds, radius, color_thresh, (long)max_iters);

    // Build grid
    fprintf(stderr, "BFS: building grid...\n");
    Grid grid;
    grid.build(pts, (int)N, radius);
    fprintf(stderr, "BFS: grid built\n");

    // BFS state
    auto visited = parlay::tabulate<std::atomic<bool>>(N, [](long) { return false; });
    for (vertex s : seeds_vec) visited[s].store(true);

    parlay::sequence<vertex> frontier(seeds_vec.begin(), seeds_vec.end());
    parlay::sequence<vertex> result(seeds_vec.begin(), seeds_vec.end());

    // Initial frontier: only boundary seeds (those with unvisited neighbors)
    fprintf(stderr, "BFS: computing boundary of %zu seeds...\n", frontier.size());
    auto is_boundary = parlay::map(frontier, [&](vertex u) {
        for (vertex v : grid.neighbors(pts, u, radius))
            if (!visited[v].load(std::memory_order_relaxed)) return true;
        return false;
    });
    frontier = parlay::pack(frontier, is_boundary);
    fprintf(stderr, "BFS: initial frontier %zu boundary seeds\n", frontier.size());

    int iteration = 0;
    while (frontier.size() > 0 && iteration < (int)max_iters) {
        auto next_seqs = parlay::map(frontier, [&](vertex u) {
            auto nbrs = grid.neighbors(pts, u, radius);
            // Local gate: v must be color-similar to its frontier neighbor u
            return parlay::filter(nbrs, [&](vertex v) {
                if (visited[v].load(std::memory_order_relaxed)) return false;
                float dr = pts[v*6+3] - pts[u*6+3];
                float dg = pts[v*6+4] - pts[u*6+4];
                float db = pts[v*6+5] - pts[u*6+5];
                if (dr*dr + dg*dg + db*db >= ct2) return false;
                bool expected = false;
                return visited[v].compare_exchange_strong(expected, true);
            });
        });
        frontier = parlay::flatten(next_seqs);
        for (vertex v : frontier) result.push_back(v);
        fprintf(stderr, "BFS iter %d: frontier=%zu total=%zu\n",
                ++iteration, frontier.size(), result.size());
    }

    fprintf(stderr, "BFS done: %zu result pts\n", result.size());

    int64_t n_result = (int64_t)result.size();
    write_val(n_result);
    write_arr(result.data(), n_result);
    fflush(stdout);
    return 0;
}
