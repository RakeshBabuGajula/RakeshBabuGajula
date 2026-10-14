"""
scripts/trace_logos.py
======================
Traces reference vector logos for Rakesh's top stack technologies (Python, Docker, Git),
samples 900 travelling particles per logo shape, normalizes their geometry to fit the
portrait frame (180x180 centered in 300x340), and computes the Optimal Transport
minimum-cost assignment between consecutive logos.

Outputs:
- logos/traced/{name}-traced.svg: Normalized vector silhouettes
- logos/particle-trajectories.json: 900-particle matched trajectory coordinates
"""

import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import resvg_py

BASE_DIR = Path(__file__).resolve().parent.parent
LOGOS_SRC = BASE_DIR / "logos" / "source-references"
LOGOS_TRACED = BASE_DIR / "logos" / "traced"

TARGET_POINTS = 900
FRAME_W = 300
FRAME_H = 340
LOGO_BOX_SIZE = 180
# Centered in 300x340:
OFFSET_X = (FRAME_W - LOGO_BOX_SIZE) / 2.0  # 60.0
OFFSET_Y = (FRAME_H - LOGO_BOX_SIZE) / 2.0  # 80.0

LOGOS = ["python", "docker", "git"]


def sample_logo_points(svg_path, num_points=TARGET_POINTS, canvas_res=400):
    """
    Renders SVG path to a high-resolution alpha mask and samples num_points uniformly
    across the filled silhouette using stratified rejection sampling.
    """
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()

    png_bytes = resvg_py.svg_to_bytes(svg_content, width=canvas_res, height=canvas_res)
    img = Image.open(io.BytesIO(png_bytes))
    arr = np.array(img)

    # In RGBA, mask is alpha channel > 128
    if arr.shape[-1] == 4:
        mask = arr[:, :, 3] > 128
    else:
        mask = arr[:, :, 0] < 128

    ys, xs = np.where(mask)
    if len(xs) < num_points:
        raise ValueError(f"Logo mask has only {len(xs)} pixels, fewer than {num_points} requested!")

    # Normalize coordinates to [0, 1] relative to the logo's tight bounding box
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    width = max_x - min_x
    height = max_y - min_y
    aspect = width / max(1, height)

    # Scale within [0, 1] preserving aspect ratio
    if aspect >= 1.0:
        norm_w = 1.0
        norm_h = height / width
        offset_norm_x = 0.0
        offset_norm_y = (1.0 - norm_h) / 2.0
    else:
        norm_h = 1.0
        norm_w = width / height
        offset_norm_x = (1.0 - norm_w) / 2.0
        offset_norm_y = 0.0

    norm_xs = (xs - min_x) / width * norm_w + offset_norm_x
    norm_ys = (ys - min_y) / height * norm_h + offset_norm_y

    # Stratified point selection to ensure uniform density without clustering
    np.random.seed(42)
    # Divide into small 2D bins
    n_bins = int(np.sqrt(num_points * 2))
    bin_x = np.clip((norm_xs * n_bins).astype(int), 0, n_bins - 1)
    bin_y = np.clip((norm_ys * n_bins).astype(int), 0, n_bins - 1)
    bin_ids = bin_y * n_bins + bin_x

    # Sample iteratively across non-empty bins
    unique_bins = np.unique(bin_ids)
    chosen_indices = []

    # First pass: 1 point per bin
    bin_to_indices = {b: np.where(bin_ids == b)[0] for b in unique_bins}
    for b in unique_bins:
        chosen_indices.append(np.random.choice(bin_to_indices[b]))

    # Fill remainder uniformly at random without replacement
    remaining_needed = num_points - len(chosen_indices)
    if remaining_needed > 0:
        all_pool = list(set(range(len(xs))) - set(chosen_indices))
        extra = np.random.choice(all_pool, size=remaining_needed, replace=False)
        chosen_indices.extend(extra)
    elif remaining_needed < 0:
        chosen_indices = list(np.random.choice(chosen_indices, size=num_points, replace=False))

    final_norm_xs = norm_xs[chosen_indices]
    final_norm_ys = norm_ys[chosen_indices]

    # Convert to frame pixel coordinates
    pts_x = OFFSET_X + final_norm_xs * LOGO_BOX_SIZE
    pts_y = OFFSET_Y + final_norm_ys * LOGO_BOX_SIZE

    return np.column_stack([pts_x, pts_y])


def compute_optimal_transport_matching(points_a, points_b):
    """
    Computes minimum-cost Euclidean bipartite matching between points_a and points_b.
    Returns optimal permutation array of indices in points_b.
    """
    diff = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    # row_ind is 0..N-1, col_ind is matching index in points_b
    return col_ind


def trace_and_match_logos():
    LOGOS_TRACED.mkdir(parents=True, exist_ok=True)
    raw_points = {}

    print(f"Sampling {TARGET_POINTS} points per logo...")
    for logo_name in LOGOS:
        src_file = LOGOS_SRC / f"{logo_name}.svg"
        if not src_file.exists():
            print(f"[REPORT] Missing reference logo: {src_file}")
            continue

        pts = sample_logo_points(src_file, num_points=TARGET_POINTS)
        raw_points[logo_name] = pts
        print(f"  {logo_name}: sampled {len(pts)} points, centroid: ({pts[:,0].mean():.1f}, {pts[:,1].mean():.1f})")

        # Save traced SVG preview
        traced_svg_path = LOGOS_TRACED / f"{logo_name}-traced.svg"
        circles = "".join([f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.4" fill="#22D3EE"/>' for x, y in pts])
        svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {FRAME_W} {FRAME_H}" width="{FRAME_W}" height="{FRAME_H}">
  <rect width="{FRAME_W}" height="{FRAME_H}" fill="#0A161F"/>
  {circles}
</svg>"""
        with open(traced_svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        print(f"  Saved traced visualization: {traced_svg_path}")

    # Optimal Transport Matching:
    # 1. P0 = Python points
    # 2. Match Python -> Docker
    # 3. Match Docker -> Git
    # 4. Cycle returns Git -> Python
    pts_python = raw_points["python"]
    pts_docker_raw = raw_points["docker"]
    pts_git_raw = raw_points["git"]

    print("Computing Optimal Transport point assignments (linear sum assignment)...")
    docker_perm = compute_optimal_transport_matching(pts_python, pts_docker_raw)
    pts_docker = pts_docker_raw[docker_perm]
    d_p2d = np.linalg.norm(pts_python - pts_docker, axis=-1).mean()
    print(f"  Python -> Docker mean travel distance: {d_p2d:.2f}px")

    git_perm = compute_optimal_transport_matching(pts_docker, pts_git_raw)
    pts_git = pts_git_raw[git_perm]
    d_d2g = np.linalg.norm(pts_docker - pts_git, axis=-1).mean()
    print(f"  Docker -> Git mean travel distance: {d_d2g:.2f}px")

    d_g2p = np.linalg.norm(pts_git - pts_python, axis=-1).mean()
    print(f"  Git -> Python mean return travel distance: {d_g2p:.2f}px")

    # Initial dispersal positions around portrait centroid for entrance transition:
    # Drift particles condense from slight dispersion (radius ~25px around centroid)
    centroid = np.array([FRAME_W / 2.0, FRAME_H / 2.0])
    np.random.seed(1337)
    angles = np.random.uniform(0, 2 * np.pi, TARGET_POINTS)
    radii = np.random.uniform(10, 45, TARGET_POINTS)
    pts_init = centroid + np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # Package trajectories
    trajectories = []
    for i in range(TARGET_POINTS):
        trajectories.append({
            "id": i,
            "p_init": [round(float(pts_init[i, 0]), 2), round(float(pts_init[i, 1]), 2)],
            "p_python": [round(float(pts_python[i, 0]), 2), round(float(pts_python[i, 1]), 2)],
            "p_docker": [round(float(pts_docker[i, 0]), 2), round(float(pts_docker[i, 1]), 2)],
            "p_git": [round(float(pts_git[i, 0]), 2), round(float(pts_git[i, 1]), 2)],
        })

    out_json = BASE_DIR / "logos" / "particle-trajectories.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, indent=2)
    print(f"Successfully saved {len(trajectories)} matched trajectories to {out_json}!")

    return trajectories


if __name__ == "__main__":
    trace_and_match_logos()
