// Parallel BFS hole filling using ParlayLib.

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/monoid.h>

using vertex = int32_t;

// ---------------------------------------------------------------------------
// Point layout: the flat float buffer from Python is [x, y, z, r, g, b] * N
// ---------------------------------------------------------------------------
struct Point { float x, y, z, r, g, b; };

inline const Point& pt(const float* buf, vertex i) {
    return reinterpret_cast<const Point*>(buf)[i];
}

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

    void build(const float* buf, int N, float radius) {
        cell = radius;

        // Parallel bounding box via single reduce with custom monoid
        struct BBox { float minx, maxx, miny, maxy, minz, maxz; };
        auto bbox = parlay::reduce(
            parlay::tabulate(N, [&](int i) {
                const Point& p = pt(buf, i);
                return BBox{p.x, p.x, p.y, p.y, p.z, p.z};
            }),
            parlay::make_monoid([](BBox a, BBox b) {
                return BBox{
                    std::min(a.minx, b.minx), std::max(a.maxx, b.maxx),
                    std::min(a.miny, b.miny), std::max(a.maxy, b.maxy),
                    std::min(a.minz, b.minz), std::max(a.maxz, b.maxz)
                };
            }, BBox{1e38f, -1e38f, 1e38f, -1e38f, 1e38f, -1e38f})
        );
        ox = bbox.minx; float mx = bbox.maxx;
        oy = bbox.miny; float my = bbox.maxy;
        oz = bbox.minz; float mz = bbox.maxz;

        nx = (int64_t)((mx - ox) / cell) + 2;
        ny = (int64_t)((my - oy) / cell) + 2;
        nz = (int64_t)((mz - oz) / cell) + 2;
        int64_t total_cells = nx * ny * nz;

        fprintf(stderr, "Grid: bbox %.3f x %.3f x %.3f | cells %lld x %lld x %lld = %lld | cell_size=%.4f\n",
                mx-ox, my-oy, mz-oz, (long long)nx, (long long)ny, (long long)nz,
                (long long)total_cells, cell);

        auto ids = parlay::tabulate(N, [&](int i) {
            const Point& p = pt(buf, i);
            return cell_id(p.x, p.y, p.z);
        });
        sorted = parlay::tabulate(N, [](int i) { return (vertex)i; });
        parlay::integer_sort_inplace(sorted, [&](vertex i) { return (size_t)ids[i]; });

        offsets = parlay::sequence<int32_t>(total_cells + 1, N);
        parlay::parallel_for(0, N, [&](int i) {
            if (i == 0 || ids[sorted[i]] != ids[sorted[i-1]])
                offsets[ids[sorted[i]]] = i;
        });
        // Backward fill: for each empty cell (sentinel N), propagate the nearest
        // populated offset from the right. Parallel via suffix scan: reverse,
        // scan with "keep left if set, else take right" monoid, then write back.
        auto sentinel = (int32_t)N;
        auto rev = parlay::tabulate(total_cells + 1, [&](int64_t i) {
            return offsets[total_cells - i];
        });
        parlay::scan_inclusive_inplace(rev, parlay::make_monoid(
            [sentinel](int32_t a, int32_t b) { return b == sentinel ? a : b; },
            sentinel
        ));
        parlay::parallel_for(0, total_cells + 1, [&](int64_t i) {
            offsets[total_cells - i] = rev[i];
        });
    }

    parlay::sequence<vertex> neighbors(const float* buf, vertex idx, float radius) const {
        const Point& center = pt(buf, idx);
        float r2 = radius * radius;
        int64_t cx = (int64_t)((center.x - ox) / cell);
        int64_t cy = (int64_t)((center.y - oy) / cell);
        int64_t cz = (int64_t)((center.z - oz) / cell);

        parlay::sequence<vertex> result;
        for (int64_t dx = -1; dx <= 1; dx++)
        for (int64_t dy = -1; dy <= 1; dy++)
        for (int64_t dz = -1; dz <= 1; dz++) {
            int64_t c = (cx+dx)*ny*nz + (cy+dy)*nz + (cz+dz);
            if (c < 0 || c >= nx*ny*nz) continue;
            int32_t s = offsets[c], e = offsets[c+1];
            for (int32_t k = s; k < e; k++) {
                const Point& q = pt(buf, sorted[k]);
                float ddx = q.x - center.x;
                float ddy = q.y - center.y;
                float ddz = q.z - center.z;
                if (ddx*ddx + ddy*ddy + ddz*ddz <= r2)
                    result.push_back(sorted[k]);
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

    std::vector<float> pts_buf(N * sizeof(Point) / sizeof(float));
    read_arr(pts_buf.data(), pts_buf.size());
    const float* buf = pts_buf.data();

    std::vector<vertex> seeds_vec(n_seeds);
    read_arr(seeds_vec.data(), n_seeds);

    fprintf(stderr, "BFS: N=%ld seeds=%ld radius=%.4f color_thresh=%.4f max_iters=%ld\n",
            (long)N, (long)n_seeds, radius, color_thresh, (long)max_iters);

    // Build grid
    fprintf(stderr, "BFS: building grid...\n");
    Grid grid;
    grid.build(buf, (int)N, radius);
    fprintf(stderr, "BFS: grid built\n");

    // BFS state
    auto visited = parlay::tabulate<std::atomic<bool>>(N, [](long) { return false; });

    // Parallel seed marking
    parlay::parallel_for(0, n_seeds, [&](long i) {
        visited[seeds_vec[i]].store(true, std::memory_order_relaxed);
    });

    parlay::sequence<vertex> frontier(seeds_vec.begin(), seeds_vec.end());

    // Collect per-iteration frontiers; flatten once at the end
    parlay::sequence<parlay::sequence<vertex>> all_frontiers;
    all_frontiers.push_back(parlay::sequence<vertex>(seeds_vec.begin(), seeds_vec.end()));

    // Initial frontier: only boundary seeds (those with unvisited neighbors)
    fprintf(stderr, "BFS: computing boundary of %zu seeds...\n", frontier.size());
    auto is_boundary = parlay::map(frontier, [&](vertex u) {
        for (vertex v : grid.neighbors(buf, u, radius))
            if (!visited[v].load(std::memory_order_relaxed)) return true;
        return false;
    });
    frontier = parlay::pack(frontier, is_boundary);
    fprintf(stderr, "BFS: initial frontier %zu boundary seeds\n", frontier.size());

    int iteration = 0;
    while (frontier.size() > 0 && iteration < (int)max_iters) {
        auto next_seqs = parlay::map(frontier, [&](vertex u) {
            auto nbrs = grid.neighbors(buf, u, radius);
            return parlay::filter(nbrs, [&](vertex v) {
                if (visited[v].load(std::memory_order_relaxed)) return false;
                const Point& pv = pt(buf, v);
                const Point& pu = pt(buf, u);
                float dr = pv.r - pu.r;
                float dg = pv.g - pu.g;
                float db = pv.b - pu.b;
                if (dr*dr + dg*dg + db*db >= ct2) return false;
                bool expected = false;
                return visited[v].compare_exchange_strong(expected, true);
            });
        });
        frontier = parlay::flatten(next_seqs);
        all_frontiers.push_back(frontier);
        fprintf(stderr, "BFS iter %d: frontier=%zu\n", ++iteration, frontier.size());
    }

    // Flatten all collected frontiers into the result in parallel
    auto result = parlay::flatten(all_frontiers);
    fprintf(stderr, "BFS done: %zu result pts\n", result.size());

    int64_t n_result = (int64_t)result.size();
    write_val(n_result);
    write_arr(result.data(), n_result);
    fflush(stdout);
    return 0;
}
