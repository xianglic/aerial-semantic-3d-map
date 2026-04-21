// Parallel BFS hole filling using ParlayLib.
//
// Binary protocol (stdin -> stdout):
//   IN:  int64 N, int64 n_seeds, float64 radius, float64 color_thresh,
//        float64[3] ref_color, float32[N*6] xyzrgb, int32[n_seeds] seed_indices
//   OUT: int64 n_result, int32[n_result] result_indices

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <parlay/primitives.h>
#include <parlay/sequence.h>

// ---------------------------------------------------------------------------
// Grid-based spatial index (read-only after build, safe for parallel queries)
// ---------------------------------------------------------------------------
struct SpatialGrid {
    float cell, ox, oy, oz;
    int ny, nz;

    // Points sorted by cell id; cell_start[cell] = first pos in sorted_pts
    parlay::sequence<int32_t> sorted_pts;
    std::unordered_map<int64_t, int32_t> cell_start; // cell_id -> start in sorted_pts
    std::unordered_map<int64_t, int32_t> cell_end;

    int64_t cell_id(float x, float y, float z) const {
        int cx = (int)((x - ox) / cell);
        int cy = (int)((y - oy) / cell);
        int cz = (int)((z - oz) / cell);
        return (int64_t)cx * 100003LL * 100003LL + (int64_t)cy * 100003LL + cz;
    }

    void build(const float* pts, int N, float radius) {
        cell = radius;
        ox = oy = oz = 1e38f;
        for (int i = 0; i < N; i++) {
            ox = std::min(ox, pts[i * 6 + 0]);
            oy = std::min(oy, pts[i * 6 + 1]);
            oz = std::min(oz, pts[i * 6 + 2]);
        }

        // Assign cell id per point, sort by it
        auto ids = parlay::tabulate(N, [&](int i) {
            return cell_id(pts[i*6], pts[i*6+1], pts[i*6+2]);
        });
        sorted_pts = parlay::tabulate(N, [](int i) { return (int32_t)i; });
        parlay::sort_inplace(sorted_pts, [&](int32_t a, int32_t b) {
            return ids[a] < ids[b];
        });

        // Build cell_start / cell_end from sorted run boundaries
        for (int i = 0; i < N; i++) {
            int64_t c = ids[sorted_pts[i]];
            if (i == 0 || ids[sorted_pts[i-1]] != c) cell_start[c] = i;
            if (i == N-1 || ids[sorted_pts[i+1]] != c) cell_end[c] = i + 1;
        }
    }

    // Returns all neighbor indices within radius (thread-safe, read-only)
    parlay::sequence<int32_t> neighbors(const float* pts, int idx, float radius) const {
        float x = pts[idx*6], y = pts[idx*6+1], z = pts[idx*6+2];
        float r2 = radius * radius;
        int cx = (int)((x - ox) / cell);
        int cy = (int)((y - oy) / cell);
        int cz = (int)((z - oz) / cell);

        parlay::sequence<int32_t> result;
        for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dz = -1; dz <= 1; dz++) {
            int64_t c = cell_id(x + dx*cell, y + dy*cell, z + dz*cell);
            auto it = cell_start.find(c);
            if (it == cell_start.end()) continue;
            int start = it->second, end = cell_end.at(c);
            for (int k = start; k < end; k++) {
                int32_t j = sorted_pts[k];
                float ddx = pts[j*6] - x;
                float ddy = pts[j*6+1] - y;
                float ddz = pts[j*6+2] - z;
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
    if (fread(buf, sizeof(T), n, stdout) != n) {} // ignore
    if (fread(buf, sizeof(T), n, stdin) != n) { fprintf(stderr, "read error\n"); exit(1); }
}
template<typename T> void write_val(T v) {
    fwrite(&v, sizeof(T), 1, stdout);
}
template<typename T> void write_arr(const T* buf, size_t n) {
    fwrite(buf, sizeof(T), n, stdout);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    int64_t N, n_seeds;
    double radius_d, color_thresh_d, ref_r, ref_g, ref_b;

    read_val(N);
    read_val(n_seeds);
    read_val(radius_d);
    read_val(color_thresh_d);
    read_val(ref_r); read_val(ref_g); read_val(ref_b);

    float radius      = (float)radius_d;
    float color_thresh = (float)color_thresh_d;
    float ref_color[3] = { (float)ref_r, (float)ref_g, (float)ref_b };

    std::vector<float> pts_buf(N * 6);
    read_arr(pts_buf.data(), N * 6);
    const float* pts = pts_buf.data();

    std::vector<int32_t> seeds(n_seeds);
    read_arr(seeds.data(), n_seeds);

    fprintf(stderr, "BFS: N=%ld seeds=%ld radius=%.4f color_thresh=%.4f\n",
            N, n_seeds, radius, color_thresh);

    // Build spatial grid
    fprintf(stderr, "BFS: building spatial grid...\n");
    SpatialGrid grid;
    grid.build(pts, (int)N, radius);
    fprintf(stderr, "BFS: grid built\n");

    // BFS state
    auto visited = parlay::tabulate<std::atomic<bool>>(N, [](long) { return false; });
    for (int32_t s : seeds) visited[s].store(true);

    parlay::sequence<int32_t> frontier(seeds.begin(), seeds.end());
    parlay::sequence<int32_t> result(seeds.begin(), seeds.end());

    int iteration = 0;
    while (frontier.size() > 0) {
        // For each frontier vertex, find neighbors in parallel
        auto neighbor_seqs = parlay::map(frontier, [&](int32_t u) {
            auto nbrs = grid.neighbors(pts, u, radius);
            // Filter: not visited + color gate + atomic claim
            return parlay::filter(nbrs, [&](int32_t v) {
                if (visited[v].load(std::memory_order_relaxed)) return false;
                float dr = pts[v*6+3] - ref_color[0];
                float dg = pts[v*6+4] - ref_color[1];
                float db = pts[v*6+5] - ref_color[2];
                if (dr*dr + dg*dg + db*db >= color_thresh * color_thresh) return false;
                bool expected = false;
                return visited[v].compare_exchange_strong(expected, true);
            });
        });

        frontier = parlay::flatten(neighbor_seqs);
        for (int32_t v : frontier) result.push_back(v);

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
