"""Python port of the 2-D FIRI corridor code from ST-opt-tools.

The implementation mirrors the C++ pieces:

* ``SDMN2D`` solves the small randomized minimum-norm separation problem.
* ``MVIE2D`` updates the maximum-volume inscribed ellipse.
* ``FIRISolver`` alternates RSI half-plane generation and MVIE updates.
* ``CorridorGenerator`` extracts occupancy boundary points and builds a
  heading-aligned local corridor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HalfPlane2D:
    normal: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    offset: float = 0.0

    def __post_init__(self):
        self.normal = np.asarray(self.normal, dtype=np.float64).reshape(2)
        self.offset = float(self.offset)


@dataclass
class FootprintSpec:
    length: float = 0.0
    width: float = 0.0
    offset_x: float = 0.0


@dataclass
class BoundingBoxSpec:
    ahead: float = 0.0
    behind: float = 0.0
    side: float = 0.0


@dataclass
class GeneratorOptions:
    max_iter: int = 10
    convergence_rho: float = 0.02
    use_path_seed: bool = True
    path_seed_count: int = 4
    path_lookahead: float = 4.0
    obstacle_filter_radius: float | None = None


@dataclass
class CorridorResult:
    planes: list[HalfPlane2D] = field(default_factory=list)
    vertices: list[np.ndarray] = field(default_factory=list)
    seed: list[np.ndarray] = field(default_factory=list)
    obstacles: list[np.ndarray] = field(default_factory=list)
    iterations: int = 0
    solve_time_ms: float = 0.0

    def as_hpoly(self) -> np.ndarray:
        if not self.planes:
            return np.empty((0, 3), dtype=np.float64)
        return np.array([[p.normal[0], p.normal[1], p.offset] for p in self.planes], dtype=np.float64)


@dataclass
class Ellipsoid:
    L: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=np.float64))
    d: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))

    def volume(self) -> float:
        return float(np.pi * abs(self.L[0, 0] * self.L[1, 1]))


class SDMN2D:
    """Small-dimensional minimum-norm solver used by FIRI RSI."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def solve(self, e, f):
        normals = [np.asarray(v, dtype=np.float64).reshape(2) for v in e]
        bounds = [float(v) for v in f]
        dim = len(normals)
        if dim == 0:
            return np.zeros(2, dtype=np.float64), True

        perm = np.arange(dim)
        self.rng.shuffle(perm)
        y = np.zeros(2, dtype=np.float64)

        for ii, idx in enumerate(perm):
            if float(normals[idx].dot(y)) <= bounds[idx] + 1e-12:
                continue

            eh = normals[idx]
            fh = bounds[idx]
            e_te = float(eh.dot(eh))
            if e_te < 1e-15:
                return np.zeros(2, dtype=np.float64), False

            v = (fh / e_te) * eh
            j = 0 if abs(v[0]) >= abs(v[1]) else 1
            k = 1 - j
            v_norm = float(np.linalg.norm(v))

            if v_norm < 1e-15:
                m_col = np.array([-eh[1], eh[0]], dtype=np.float64)
                m_norm = float(np.linalg.norm(m_col))
                if m_norm <= 1e-15:
                    return np.zeros(2, dtype=np.float64), False
                m_col /= m_norm
            else:
                sign_vj = 1.0 if v[j] >= 0.0 else -1.0
                unit_j = np.zeros(2, dtype=np.float64)
                unit_j[j] = 1.0
                unit_k = np.zeros(2, dtype=np.float64)
                unit_k[k] = 1.0
                u_ref = v + sign_vj * v_norm * unit_j
                u_t_u = float(u_ref.dot(u_ref))
                if u_t_u < 1e-15:
                    m_col = np.array([-eh[1], eh[0]], dtype=np.float64)
                    m_col /= max(np.linalg.norm(m_col), 1e-15)
                else:
                    m_col = unit_k - (2.0 * u_ref[k] / u_t_u) * u_ref

            lo = -1e18
            hi = 1e18
            feasible = True
            for pp in range(ii):
                pidx = perm[pp]
                a_1d = float(normals[pidx].dot(m_col))
                b_1d = bounds[pidx] - float(normals[pidx].dot(v))
                if abs(a_1d) < 1e-15:
                    if b_1d < -1e-10:
                        feasible = False
                        break
                    continue
                bound = b_1d / a_1d
                if a_1d > 0.0:
                    hi = min(hi, bound)
                else:
                    lo = max(lo, bound)

            if not feasible or lo > hi + 1e-10:
                return np.zeros(2, dtype=np.float64), False

            if lo <= 0.0 <= hi:
                t = 0.0
            elif lo > 0.0:
                t = lo
            else:
                t = hi
            y = m_col * t + v

        return y, True


class MVIE2D:
    """Barrier Newton solver for a 2-D maximum-volume inscribed ellipse."""

    def solve(self, A, b, center_hint):
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        center_hint = np.asarray(center_hint, dtype=np.float64).reshape(2)
        m = A.shape[0]

        c = center_hint.copy()
        radius = 1e18
        for i in range(m):
            norm_ai = float(np.linalg.norm(A[i]))
            if norm_ai > 1e-10:
                gap = float(b[i] - A[i].dot(c))
                radius = min(radius, gap / norm_ai)
        if radius <= 0.0 or not np.isfinite(radius):
            radius = 1e-4
        radius = max(radius * 0.9, 1e-6)

        x = np.array([radius, 0.0, radius, c[0], c[1]], dtype=np.float64)
        barrier_t = 1.0
        mu = 4.0

        for _ in range(20):
            for _ in range(40):
                grad = self._gradient(A, b, x, barrier_t)
                hess = self._hessian(A, b, x, barrier_t)
                hess += 1e-8 * np.eye(5)
                try:
                    dx = np.linalg.solve(hess, -grad)
                except np.linalg.LinAlgError:
                    dx = np.linalg.lstsq(hess, -grad, rcond=None)[0]

                lambda_sq = float(-grad.dot(dx))
                if lambda_sq < 1e-6:
                    break

                alpha = 1.0
                f0 = self._objective(A, b, x, barrier_t)
                for _ in range(32):
                    xn = x + alpha * dx
                    if xn[0] > 1e-10 and xn[2] > 1e-10:
                        fn = self._objective(A, b, xn, barrier_t)
                        if np.isfinite(fn) and fn < f0 + 0.3 * alpha * float(grad.dot(dx)):
                            x = xn
                            break
                    alpha *= 0.5
                    if alpha < 1e-12:
                        break

            if m / barrier_t < 1e-3:
                break
            barrier_t *= mu

        L = np.array([[x[0], 0.0], [x[1], x[2]]], dtype=np.float64)
        d = np.array([x[3], x[4]], dtype=np.float64)
        return Ellipsoid(L=L, d=d)

    def _objective(self, A, b, x, t):
        L11, L21, L22, d1, d2 = x
        if L11 <= 0.0 or L22 <= 0.0:
            return 1e18
        val = -t * (np.log(L11) + np.log(L22))
        for i in range(A.shape[0]):
            a1, a2 = A[i, 0], A[i, 1]
            r1 = L11 * a1 + L21 * a2
            r2 = L22 * a2
            gap = b[i] - a1 * d1 - a2 * d2 - np.sqrt(r1 * r1 + r2 * r2)
            if gap <= 0.0:
                return 1e18
            val -= np.log(gap)
        return float(val)

    def _gradient(self, A, b, x, t):
        L11, L21, L22, d1, d2 = x
        g = np.zeros(5, dtype=np.float64)
        g[0] = -t / L11
        g[2] = -t / L22

        for i in range(A.shape[0]):
            a1, a2 = A[i, 0], A[i, 1]
            r1 = L11 * a1 + L21 * a2
            r2 = L22 * a2
            nr = float(np.sqrt(r1 * r1 + r2 * r2))
            gap = b[i] - a1 * d1 - a2 * d2 - nr
            gap = max(float(gap), 1e-15)
            inv_gap = 1.0 / gap
            if nr > 1e-15:
                inv_nr = 1.0 / nr
                g[0] += inv_gap * r1 * a1 * inv_nr
                g[1] += inv_gap * r1 * a2 * inv_nr
                g[2] += inv_gap * r2 * a2 * inv_nr
            g[3] += inv_gap * a1
            g[4] += inv_gap * a2
        return g

    def _hessian(self, A, b, x, t):
        eps = 1e-6
        H = np.zeros((5, 5), dtype=np.float64)
        for j in range(5):
            xp = x.copy()
            xm = x.copy()
            xp[j] += eps
            xm[j] -= eps
            H[:, j] = (self._gradient(A, b, xp, t) - self._gradient(A, b, xm, t)) / (2.0 * eps)
        return 0.5 * (H + H.T)


class FIRISolver:
    """Alternating RSI/MVIE FIRI solver."""

    def __init__(self):
        self.sdmn = SDMN2D()
        self.mvie_solver = MVIE2D()

    def compute(
        self,
        obstacles,
        seed_vertices,
        bbox_planes,
        max_iter: int = 10,
        rho: float = 0.02,
    ):
        start = time.perf_counter()
        obstacles = [np.asarray(p, dtype=np.float64).reshape(2) for p in obstacles]
        seed_vertices = [np.asarray(p, dtype=np.float64).reshape(2) for p in seed_vertices]
        bbox_planes = [HalfPlane2D(p.normal, p.offset) for p in bbox_planes]

        if not seed_vertices or not obstacles:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return list(bbox_planes), 0, elapsed_ms

        center = np.mean(np.vstack(seed_vertices), axis=0)
        radius = max(self._inscribed_radius(seed_vertices, center) * 0.8, 1e-4)
        L = radius * np.eye(2)
        prev_vol = radius * radius * np.pi
        best_planes = list(bbox_planes)
        iterations = 0

        for k in range(int(max_iter)):
            iterations = k + 1
            planes = self._run_rsi(obstacles, seed_vertices, L, center, bbox_planes)
            best_planes = planes

            A = np.vstack([p.normal for p in planes])
            b = np.array([p.offset for p in planes], dtype=np.float64)
            mvie = self.mvie_solver.solve(A, b, center)
            new_vol = mvie.volume()
            if k > 0 and (new_vol - prev_vol) / (prev_vol + 1e-15) < rho:
                break

            prev_vol = new_vol
            L = mvie.L
            center = mvie.d

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return best_planes, iterations, elapsed_ms

    def _inscribed_radius(self, verts, center):
        radius = 1e18
        n = len(verts)
        for i in range(n):
            a = verts[i]
            b = verts[(i + 1) % n]
            edge = b - a
            length = float(np.linalg.norm(edge))
            if length < 1e-15:
                continue
            normal = np.array([-edge[1], edge[0]], dtype=np.float64) / length
            radius = min(radius, abs(float(normal.dot(center - a))))
        if not np.isfinite(radius):
            return 1e-4
        return float(radius)

    def _run_rsi(self, obstacles, seed_vertices, L, center, bbox_planes):
        try:
            L_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            L_inv = np.linalg.pinv(L)
        L_inv_t = L_inv.T

        seed_bar = [L_inv @ (v - center) for v in seed_vertices]
        obs_bar = [L_inv @ (u - center) for u in obstacles]
        n_seed = len(seed_bar)

        base_normals = [None] * (n_seed + 1)
        base_bounds = [0.0] * (n_seed + 1)
        for i in range(n_seed):
            base_normals[i] = seed_bar[i]
            base_bounds[i] = 1.0

        candidates = []
        for obs_idx, obs in enumerate(obs_bar):
            base_normals[n_seed] = -obs
            base_bounds[n_seed] = -1.0
            y, feasible = self.sdmn.solve(base_normals, base_bounds)
            if not feasible:
                continue
            y_sq = float(y.dot(y))
            if y_sq <= 1e-10:
                continue
            a = y / y_sq
            candidates.append((float(np.linalg.norm(a)), obs_idx, y, a))

        candidates.sort(key=lambda item: item[0])
        separated = np.zeros(len(obs_bar), dtype=bool)
        result_planes = list(bbox_planes)

        for _, obs_idx, b_sol, a in candidates:
            if separated[obs_idx]:
                continue

            n_orig = L_inv_t @ a
            d_orig = float(a.dot(a) + n_orig.dot(center))
            n_norm = float(np.linalg.norm(n_orig))
            if n_norm < 1e-15:
                continue

            result_planes.append(HalfPlane2D(n_orig / n_norm, d_orig / n_norm))

            for i, obs in enumerate(obs_bar):
                if not separated[i] and float(b_sol.dot(obs)) >= 1.0 - 1e-8:
                    separated[i] = True

            if len(result_planes) > 50:
                break

        return result_planes


class CorridorGenerator:
    """FIRI corridor generator for occupancy grids."""

    def __init__(self):
        self.solver = FIRISolver()

    def generate(
        self,
        grid_map,
        robot_pos,
        robot_yaw: float,
        footprint: FootprintSpec,
        bbox: BoundingBoxSpec,
        options: GeneratorOptions | None = None,
        path=None,
        obstacles=None,
    ) -> CorridorResult:
        if options is None:
            options = GeneratorOptions()
        if path is None:
            path = []

        result = CorridorResult()
        if obstacles is None:
            result.obstacles = self.extract_boundary_obstacles(grid_map)
        else:
            result.obstacles = [np.asarray(p, dtype=np.float64).reshape(2) for p in obstacles]
        if options.obstacle_filter_radius is not None:
            robot_pos_arr = np.asarray(robot_pos, dtype=np.float64).reshape(2)
            radius = float(options.obstacle_filter_radius)
            result.obstacles = [
                p for p in result.obstacles
                if float(np.linalg.norm(p - robot_pos_arr)) <= radius
            ]
        result.seed = self.build_footprint_seed(robot_pos, robot_yaw, footprint)

        if options.use_path_seed and len(path) > 0:
            result.seed.extend(
                self.sample_path_seed(
                    path,
                    robot_pos,
                    footprint,
                    options.path_seed_count,
                    options.path_lookahead,
                )
            )

        bbox_planes = self.build_heading_aligned_bbox(robot_pos, robot_yaw, bbox)
        planes, iterations, solve_time_ms = self.solver.compute(
            result.obstacles,
            result.seed,
            bbox_planes,
            options.max_iter,
            options.convergence_rho,
        )
        result.planes = planes
        result.iterations = iterations
        result.solve_time_ms = solve_time_ms
        result.vertices = self.compute_polytope_vertices(result.planes)
        return result

    def extract_boundary_obstacles(self, grid_map) -> list[np.ndarray]:
        occ = np.asarray(grid_map.occ)
        ny, nx = occ.shape
        res = float(grid_map.resolution)
        origin_x = float(grid_map.origin_x)
        origin_y = float(grid_map.origin_y)

        directions = [
            (1, 0, 0.0, 0.5),
            (-1, 0, 0.0, -0.5),
            (0, 1, 0.5, 0.0),
            (0, -1, -0.5, 0.0),
        ]
        obstacles = []
        for row in range(ny):
            for col in range(nx):
                if occ[row, col] <= 0:
                    continue
                for dr, dc, face_dx, face_dy in directions:
                    nr = row + dr
                    nc = col + dc
                    if nr < 0 or nr >= ny or nc < 0 or nc >= nx:
                        neighbor_is_free = True
                    else:
                        neighbor_is_free = occ[nr, nc] <= 0
                    if not neighbor_is_free:
                        continue
                    x = origin_x + (col + 0.5 + face_dx) * res
                    y = origin_y + (row + 0.5 + face_dy) * res
                    obstacles.append(np.array([x, y], dtype=np.float64))
        return obstacles

    def build_footprint_seed(self, robot_pos, robot_yaw: float, footprint: FootprintSpec):
        robot_pos = np.asarray(robot_pos, dtype=np.float64).reshape(2)
        half_len = footprint.length / 2.0
        half_width = footprint.width / 2.0
        R = self._rotation(robot_yaw)
        center = robot_pos + R[:, 0] * footprint.offset_x
        return [
            center + R @ np.array([half_len, half_width]),
            center + R @ np.array([half_len, -half_width]),
            center + R @ np.array([-half_len, -half_width]),
            center + R @ np.array([-half_len, half_width]),
        ]

    def sample_path_seed(self, path, robot_pos, footprint: FootprintSpec, path_seed_count: int, path_lookahead: float):
        pts = []
        path = [np.asarray(p, dtype=np.float64).reshape(2) for p in path]
        robot_pos = np.asarray(robot_pos, dtype=np.float64).reshape(2)
        if not path or path_seed_count <= 0:
            return pts

        dists_sq = [float(np.sum((p - robot_pos) ** 2)) for p in path]
        closest_idx = int(np.argmin(dists_sq))
        if np.sqrt(dists_sq[closest_idx]) > path_lookahead:
            return pts

        front_edge_dist = footprint.offset_x + footprint.length / 2.0
        skip_dist_sq = front_edge_dist * front_edge_dist
        for i in range(closest_idx + 1, len(path)):
            if len(pts) >= path_seed_count:
                break
            if float(np.sum((path[i] - robot_pos) ** 2)) < skip_dist_sq:
                continue
            pts.append(path[i])
        return pts

    def build_heading_aligned_bbox(self, robot_pos, robot_yaw: float, bbox: BoundingBoxSpec):
        robot_pos = np.asarray(robot_pos, dtype=np.float64).reshape(2)
        R = self._rotation(robot_yaw)
        fwd = R[:, 0]
        left = R[:, 1]
        return [
            HalfPlane2D(fwd, float(fwd.dot(robot_pos) + bbox.ahead)),
            HalfPlane2D(-fwd, float(-fwd.dot(robot_pos) + bbox.behind)),
            HalfPlane2D(left, float(left.dot(robot_pos) + bbox.side)),
            HalfPlane2D(-left, float(-left.dot(robot_pos) + bbox.side)),
        ]

    def compute_polytope_vertices(self, planes: list[HalfPlane2D]):
        vertices = []
        n = len(planes)
        for i in range(n):
            for j in range(i + 1, n):
                M = np.vstack([planes[i].normal, planes[j].normal])
                det = float(np.linalg.det(M))
                if abs(det) < 1e-10:
                    continue
                rhs = np.array([planes[i].offset, planes[j].offset], dtype=np.float64)
                vertex = np.linalg.solve(M, rhs)
                inside = True
                for plane in planes:
                    if float(plane.normal.dot(vertex)) > plane.offset + 1e-6:
                        inside = False
                        break
                if inside and not self._has_duplicate(vertices, vertex):
                    vertices.append(vertex)

        if len(vertices) < 3:
            return vertices
        centroid = np.mean(np.vstack(vertices), axis=0)
        vertices.sort(key=lambda v: np.arctan2(v[1] - centroid[1], v[0] - centroid[0]))
        return vertices

    @staticmethod
    def _rotation(yaw: float):
        c = np.cos(yaw)
        s = np.sin(yaw)
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    @staticmethod
    def _has_duplicate(vertices, candidate, eps=1e-8):
        for existing in vertices:
            if float(np.linalg.norm(existing - candidate)) <= eps:
                return True
        return False


def build_firi_corridors(
    grid_map,
    waypoints,
    search_radius: float = 6.0,
    max_iter: int = 10,
    convergence_rho: float = 0.02,
    robot_radius: float | None = None,
) -> list[np.ndarray]:
    """Build a FIRI polytope for every segment in ``waypoints``."""

    waypoints = np.asarray(waypoints, dtype=np.float64)
    piece_num = len(waypoints) - 1
    if piece_num <= 0:
        return []

    res = float(getattr(grid_map, "resolution", 0.05))
    if robot_radius is None:
        robot_radius = float(getattr(grid_map, "robot_radius", 0.0))
    seed_half_width = max(float(robot_radius), 0.5 * res)

    generator = CorridorGenerator()
    boundary_obstacles = generator.extract_boundary_obstacles(grid_map)
    hpolys = []
    for i in range(piece_num):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        center = 0.5 * (p0 + p1)
        delta = p1 - p0
        seg_len = float(np.linalg.norm(delta))
        yaw = float(np.arctan2(delta[1], delta[0])) if seg_len > 1e-9 else 0.0
        seed_pad = max(seed_half_width, res)

        footprint = FootprintSpec(
            length=max(seg_len + 2.0 * seed_pad, 2.0 * seed_pad),
            width=max(2.0 * seed_half_width, res),
            offset_x=0.0,
        )
        bbox = BoundingBoxSpec(
            ahead=0.5 * seg_len + search_radius,
            behind=0.5 * seg_len + search_radius,
            side=search_radius,
        )
        options = GeneratorOptions(
            max_iter=max_iter,
            convergence_rho=convergence_rho,
            use_path_seed=True,
            path_seed_count=2,
            path_lookahead=max(search_radius, seg_len + search_radius),
            obstacle_filter_radius=0.5 * seg_len + search_radius + seed_pad,
        )
        result = generator.generate(
            grid_map,
            center,
            yaw,
            footprint,
            bbox,
            options,
            path=[p0, p1],
            obstacles=boundary_obstacles,
        )
        hpoly = result.as_hpoly()
        if hpoly.size == 0:
            hpoly = _fallback_bbox_hpoly(grid_map, center, search_radius)
        hpolys.append(hpoly)

    return hpolys


def _fallback_bbox_hpoly(grid_map, center, fallback_radius):
    if hasattr(grid_map, "min_boundary") and hasattr(grid_map, "max_boundary"):
        mn = grid_map.min_boundary
        mx = grid_map.max_boundary
        xmin, ymin, xmax, ymax = float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])
    else:
        xmin = float(center[0] - fallback_radius)
        xmax = float(center[0] + fallback_radius)
        ymin = float(center[1] - fallback_radius)
        ymax = float(center[1] + fallback_radius)
    return np.array(
        [
            [1.0, 0.0, xmax],
            [-1.0, 0.0, -xmin],
            [0.0, 1.0, ymax],
            [0.0, -1.0, -ymin],
        ],
        dtype=np.float64,
    )
