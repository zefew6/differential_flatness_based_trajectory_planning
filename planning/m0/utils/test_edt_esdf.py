"""Quick sanity test for strict Euclidean EDT-based ESDF.

This test is intentionally lightweight and doesn't depend on pytest.
It compares ESDF values against an analytic circle obstacle distance.

Run:
    python test_edt_esdf.py

Expected:
    - max error should be small (on the order of one grid cell).
"""

from __future__ import annotations

import numpy as np

from gridmap_2d_v2 import GridMap2D, GridMap2DParams


def analytic_signed_distance_circle(p: np.ndarray, center: np.ndarray, radius: float) -> float:
    # positive outside, negative inside (SDF convention)
    return float(np.linalg.norm(p - center) - radius)


def main() -> None:
    params = GridMap2DParams(resolution=0.05, size_x=4.0, size_y=4.0, origin_at_center=True)
    gm = GridMap2D(params)

    # Circle obstacle at origin
    center = np.array([0.0, 0.0])
    radius = 0.6

    # Occupancy: mark a cell occupied if its center is inside circle
    occ = np.zeros((gm.nx, gm.ny), dtype=np.int8)
    for ix in range(gm.nx):
        for iy in range(gm.ny):
            c = gm.index_to_pos(np.array([ix, iy], dtype=np.int32))
            if np.linalg.norm(c - center) <= radius:
                occ[ix, iy] = 1

    gm.set_occupancy(occ, update_esdf=True)

    # sample random points inside the map
    rng = np.random.default_rng(0)
    pts = rng.uniform(low=gm.min_boundary, high=gm.max_boundary, size=(300, 2))

    errs = []
    grad_norms = []
    for p in pts:
        d, g = gm.get_distance_and_gradient(p)
        da = analytic_signed_distance_circle(p, center, radius)
        errs.append(abs(d - da))
        grad_norms.append(float(np.linalg.norm(g)))

    errs = np.array(errs)
    grad_norms = np.array(grad_norms)

    print("ESDF strict-EDT test")
    print(f"  mean abs error: {errs.mean():.4f} m")
    print(f"  max  abs error: {errs.max():.4f} m")
    print(f"  grad norm    : min={grad_norms.min():.3f}, mean={grad_norms.mean():.3f}, max={grad_norms.max():.3f}")

    # Loose thresholds: discretization + bilinear interpolation + occupied-by-cell-center approximation
    # Error should typically be within a couple of cells.
    assert errs.max() < 3.5 * params.resolution, "ESDF error too large; check EDT implementation"


if __name__ == "__main__":
    main()
