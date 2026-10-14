"""
scripts/generate_banner.py
==========================
Generates self-contained, production-grade animated SVG GitHub Profile Banners:
- output/dark.svg
- output/light.svg

Architecture:
- ViewBox: 0 0 1180 610 (responsive terminal window titled 'profile.sh live')
- Left 38%: VISUAL.MAP containing 300x340 portrait & logo particle morph engine
  - Animation A: 3.2s progressive intro reveal across 60 interleaved spatial groups
  - Animation B (Layer 1): 14.2s repeating portrait drift across 94 bands with sigma=4 noise
  - Animation B (Layer 2): 14.2s repeating 900-particle Optimal Transport morph
    (Portrait -> Python -> Docker -> Git -> Portrait)
- Right 62%: SYSTEM.INFO telemetry readout with programmatically calculated dotted leaders,
  pulsing LIVE badge, handle pill @RakeshBabuGajula, and strictly aligned rows.
"""

import json
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
LOGOS_DIR = BASE_DIR / "logos"
OUTPUT_DIR = BASE_DIR / "output"

GRID_W = 300
GRID_H = 340
PORTRAIT_X = 76
PORTRAIT_Y = 126

# Terminal dimensions
VIEWBOX_W = 1180
VIEWBOX_H = 610
TERM_X = 20
TERM_Y = 20
TERM_W = 1140
TERM_H = 570
SPLIT_X = 455

# 15 Information Rows in strict order
ROWS = [
    ("Subject", "Rakesh Babu Gajula", None),
    ("Role", "AI/ML Engineer", None),
    ("Origin", "Hyderabad, India", None),
    ("Education", "B.Tech CSE / GNITC", None),
    ("Status", "Building + Shipping", None),
    ("ToolChain", "VS Code / Git / Docker", None),
    ("Core.Lang", "Python / SQL / JS", None),
    ("Core.Frontend", "HTML / CSS / JS", None),
    ("Core.Backend", "FastAPI / Flask / Node", None),
    ("Core.Database", "PostgreSQL / MongoDB", None),
    ("Core.Infra", "Docker / GitHub", None),
    ("Grid.Mail", "grakesh2605@gmail.com", "mailto:grakesh2605@gmail.com"),
    ("Grid.Portfolio", "my_portfolio", "https://rakeshbabugajula.github.io/my_portfolio/"),
    ("Grid.LinkedIn", "rakesh-babu-gajula", "https://www.linkedin.com/in/rakesh-babu-gajula/"),
    ("Grid.GitHub", "RakeshBabuGajula", "https://github.com/RakeshBabuGajula"),
]


def load_assets():
    with open(ASSETS_DIR / "portrait-dots.json", "r", encoding="utf-8") as f:
        portrait_data = json.load(f)
    with open(LOGOS_DIR / "particle-trajectories.json", "r", encoding="utf-8") as f:
        trajectories = json.load(f)
    return portrait_data, trajectories


def build_intro_groups(dot_coords, num_groups=60):
    """
    Divides portrait dots into num_groups interleaved, spatially distributed groups.
    Uses 4x4 spatial stratification to guarantee normalized distribution imbalance <= 0.05.
    """
    coords = np.array(dot_coords)
    num_dots = len(coords)
    xs, ys = coords[:, 0], coords[:, 1]

    cell_x = np.clip(xs // (GRID_W // 4), 0, 3)
    cell_y = np.clip(ys // (GRID_H // 4), 0, 3)
    cell_ids = cell_y * 4 + cell_x

    np.random.seed(42)
    group_ids = np.zeros(num_dots, dtype=int)
    for c in range(16):
        c_idx = np.where(cell_ids == c)[0]
        if len(c_idx) > 0:
            perm = np.random.permutation(len(c_idx))
            group_ids[c_idx] = perm % num_groups

    # Build SVG path runs for each group
    group_paths = []
    for g in range(num_groups):
        g_idx = np.where(group_ids == g)[0]
        g_pts = coords[g_idx]
        # Sort by y then x
        sort_order = np.lexsort((g_pts[:, 0], g_pts[:, 1]))
        sorted_pts = g_pts[sort_order]

        # Convert to run paths
        runs = []
        cur_y = None
        start_x = None
        last_x = None

        for x, y in sorted_pts:
            if cur_y != y or last_x is None or x != last_x + 1:
                if start_x is not None:
                    run_w = last_x - start_x + 1
                    runs.append(f"M{start_x},{cur_y}h{run_w}v1h-{run_w}Z")
                cur_y = y
                start_x = x
                last_x = x
            else:
                last_x = x
        if start_x is not None:
            run_w = last_x - start_x + 1
            runs.append(f"M{start_x},{cur_y}h{run_w}v1h-{run_w}Z")

        group_paths.append("".join(runs))

    return group_paths


def build_drift_bands(dot_coords, num_bands=94, target_dots=17000):
    """
    Divides portrait dots into num_bands drift bands with sigma=4 per-dot positional noise
    to prevent straight grid boundaries.
    """
    coords = np.array(dot_coords)
    num_dots = len(coords)

    # Subsample to approx target_dots if needed
    np.random.seed(42)
    if num_dots > target_dots:
        sample_idx = np.random.choice(num_dots, size=target_dots, replace=False)
        coords = coords[sample_idx]
    num_dots = len(coords)

    xs, ys = coords[:, 0], coords[:, 1]
    # Logo 1 centroid in frame:
    cx, cy = 151.4, 168.8

    # Vector toward centroid
    dx = cx - xs
    dy = cy - ys
    dist = np.sqrt(dx**2 + dy**2)

    # Add positional noise (sigma = 4.0)
    noise_x = np.random.normal(0, 4.0, num_dots)
    noise_y = np.random.normal(0, 4.0, num_dots)
    proj = np.sqrt((xs + noise_x - cx)**2 + (ys + noise_y - cy)**2)

    # Assign to quantile bands
    band_quantiles = np.linspace(0, 100, num_bands + 1)
    band_edges = np.percentile(proj, band_quantiles)
    band_ids = np.clip(np.digitize(proj, band_edges) - 1, 0, num_bands - 1)

    bands = []
    for b in range(num_bands):
        b_idx = np.where(band_ids == b)[0]
        if len(b_idx) == 0:
            continue
        b_pts = coords[b_idx]
        mean_dx = float(np.mean(cx - b_pts[:, 0]) * 0.42)
        mean_dy = float(np.mean(cy - b_pts[:, 1]) * 0.42)

        # Build run path for band b
        sort_order = np.lexsort((b_pts[:, 0], b_pts[:, 1]))
        sorted_pts = b_pts[sort_order]

        runs = []
        cur_y = None
        start_x = None
        last_x = None
        for x, y in sorted_pts:
            if cur_y != y or last_x is None or x != last_x + 1:
                if start_x is not None:
                    run_w = last_x - start_x + 1
                    runs.append(f"M{start_x},{cur_y}h{run_w}v1h-{run_w}Z")
                cur_y = y
                start_x = x
                last_x = x
            else:
                last_x = x
        if start_x is not None:
            run_w = last_x - start_x + 1
            runs.append(f"M{start_x},{cur_y}h{run_w}v1h-{run_w}Z")

        bands.append({
            "band_id": b,
            "d": "".join(runs),
            "dx": round(mean_dx, 2),
            "dy": round(mean_dy, 2),
            "dot_count": len(b_idx),
        })

    return bands


def build_system_info_rows(theme):
    """
    Generates SVG markup for the 15 SYSTEM.INFO telemetry rows with programmatically
    calculated dotted leaders and exact alignment using SVG textLength.
    """
    is_dark = (theme == "dark")
    label_color = "#689182" if is_dark else "#475569"
    leader_color = "#1E3A4F" if is_dark else "#CBD5E1"
    val_color = "#E2E8F0" if is_dark else "#0F172A"
    accent_color = "#22D3EE" if is_dark else "#0284C7"

    start_x = 485
    end_x = 1110
    start_y = 146
    line_h = 23

    # Exact monospace character width for JetBrains Mono at 14px
    char_w = 8.42

    row_elements = []
    for i, (label, val, link) in enumerate(ROWS):
        y = start_y + i * line_h

        lbl_end_x = start_x + len(label) * char_w + 10.0
        val_start_x = end_x - len(val) * char_w - 10.0
        leader_w = max(10.0, val_start_x - lbl_end_x)

        # Number of dots spaced across leader width
        num_dots = max(2, int(round(leader_w / 12.0)))
        dots_str = "." * num_dots

        # Label element
        lbl_svg = f'<text x="{start_x}" y="{y}" fill="{label_color}" font-family="\'JetBrains Mono\', monospace" font-size="14" font-weight="600">{label}</text>'

        # Dotted leader element with exact textLength constraint
        dots_svg = f'<text x="{lbl_end_x:.1f}" y="{y}" fill="{leader_color}" font-family="\'JetBrains Mono\', monospace" font-size="14" textLength="{leader_w:.1f}" lengthAdjust="spacing">{dots_str}</text>'

        # Value element
        is_accent = "Grid." in label or "Subject" in label
        c = accent_color if is_accent else val_color
        weight = "700" if is_accent else "500"

        if link:
            val_svg = f'<a href="{link}" target="_blank"><text x="{end_x}" y="{y}" text-anchor="end" fill="{c}" font-family="\'JetBrains Mono\', monospace" font-size="14" font-weight="{weight}" text-decoration="underline">{val}</text></a>'
        else:
            val_svg = f'<text x="{end_x}" y="{y}" text-anchor="end" fill="{c}" font-family="\'JetBrains Mono\', monospace" font-size="14" font-weight="{weight}">{val}</text>'

        row_elements.append(f'<g class="sys-row" id="row_{i}">{lbl_svg}{dots_svg}{val_svg}</g>')

    return "\n    ".join(row_elements)


def generate_svg_banner(theme="dark"):
    is_dark = (theme == "dark")
    portrait_data, trajectories = load_assets()

    # Theme colors
    bg_color = "#0A161F" if is_dark else "#F8FAFC"
    term_bg = "#0D1D29" if is_dark else "#FFFFFF"
    term_border = "#1E3A4F" if is_dark else "#CBD5E1"
    val_color = "#E2E8F0" if is_dark else "#0F172A"
    portrait_color = "#A78BFA" if is_dark else "#7C3AED"
    accent_cyan = "#22D3EE" if is_dark else "#0284C7"
    accent_secondary = "#689182" if is_dark else "#475569"
    accent_emerald = "#108981" if is_dark else "#0D9488"
    frame_bg = "#071017" if is_dark else "#F1F5F9"
    header_border = "#182E3E" if is_dark else "#E2E8F0"

    dots_list = portrait_data["dark_mode"]["coords"] if is_dark else portrait_data["light_mode"]["coords"]

    # 1. Build 60 intro groups (Animation A: 3.2s progressive intro)
    print(f"[{theme.upper()}] Building 60 intro reveal groups...")
    intro_groups = build_intro_groups(dots_list, num_groups=60)

    # 2. Build 94 drift bands (Animation B Layer 1)
    print(f"[{theme.upper()}] Building 94 drift bands (sigma=4 noise)...")
    drift_bands = build_drift_bands(dots_list, num_bands=94, target_dots=17000)

    # 3. Assemble SVG markup
    # Timing calculations for 14.2s morph loop:
    # k0=0.0 (0s), k1=0.2113 (3.0s), k2=0.3028 (4.3s), k3=0.4437 (6.3s),
    # k4=0.5352 (7.6s), k5=0.6761 (9.6s), k6=0.7676 (10.9s), k7=0.9085 (12.9s), k8=1.0 (14.2s)
    key_times = "0; 0.2113; 0.3028; 0.4437; 0.5352; 0.6761; 0.7676; 0.9085; 1"

    # SVG Elements:
    # A. Intro reveal layer (fades groups in over 3.2s, then fades out at 3.0s to hand off to looping layer)
    intro_layer_parts = []
    for g_id, path_d in enumerate(intro_groups):
        t_begin = round((g_id / 60.0) * 3.0, 3)
        t_dur = 0.35
        intro_layer_parts.append(
            f'<path d="{path_d}" fill="{portrait_color}" shape-rendering="crispEdges" opacity="0">'
            f'<animate attributeName="opacity" from="0" to="1" begin="{t_begin}s" dur="{t_dur}s" fill="freeze"/>'
            f'</path>'
        )
    intro_paths_joined = "\n          ".join(intro_layer_parts)

    intro_layer_svg = f"""      <!-- LAYER 0: Intro Reveal (Plays once over 3.2s across 60 interleaved groups) -->
      <g id="intro-layer" opacity="1">
        <animate attributeName="opacity" values="1; 1; 0" keyTimes="0; 0.2113; 0.3028" begin="0s" dur="14.2s" fill="freeze"/>
        <g transform="translate({PORTRAIT_X}, {PORTRAIT_Y})">
          {intro_paths_joined}
        </g>
      </g>"""

    # B. Looping Layer 1: Portrait Drift (94 bands translating by 42% toward logo centroid)
    drift_band_parts = []
    for b in drift_bands:
        dx, dy = b["dx"], b["dy"]
        drift_band_parts.append(
            f'<g class="drift-band" opacity="1">'
            f'<animateTransform attributeName="transform" type="translate" dur="14.2s" repeatCount="indefinite" '
            f'keyTimes="0; 0.2113; 0.3028; 0.9085; 1" values="0,0; 0,0; {dx},{dy}; {dx},{dy}; 0,0"/>'
            f'<animate attributeName="opacity" dur="14.2s" repeatCount="indefinite" '
            f'keyTimes="0; 0.2113; 0.3028; 0.9085; 1" values="1; 1; 0; 0; 1"/>'
            f'<path d="{b["d"]}" fill="{portrait_color}" shape-rendering="crispEdges"/>'
            f'</g>'
        )
    drift_bands_joined = "\n          ".join(drift_band_parts)

    looping_drift_svg = f"""      <!-- LAYER 1: Portrait Drift (14.2s repeat, 94 bands, 42% centroid translate) -->
      <g id="portrait-drift-layer" transform="translate({PORTRAIT_X}, {PORTRAIT_Y})">
        {drift_bands_joined}
      </g>"""

    # C. Looping Layer 2: 900 Travelling Morph Particles
    # Particle color: vibrant cyan accent for visibility and wow factor
    particle_color = accent_cyan
    particle_elements = []
    for p in trajectories:
        init_x, init_y = p["p_init"]
        py_x, py_y = p["p_python"]
        dk_x, dk_y = p["p_docker"]
        gt_x, gt_y = p["p_git"]

        # Trajectory values:
        # k0: init, k1: init, k2: python, k3: python, k4: docker, k5: docker, k6: git, k7: git, k8: init
        val_x = f"{init_x}; {init_x}; {py_x}; {py_x}; {dk_x}; {dk_x}; {gt_x}; {gt_x}; {init_x}"
        val_y = f"{init_y}; {init_y}; {py_y}; {py_y}; {dk_y}; {dk_y}; {gt_y}; {gt_y}; {init_y}"

        particle_elements.append(
            f'<circle cx="{init_x}" cy="{init_y}" r="1.5" fill="{particle_color}">'
            f'<animate attributeName="cx" dur="14.2s" repeatCount="indefinite" keyTimes="{key_times}" values="{val_x}"/>'
            f'<animate attributeName="cy" dur="14.2s" repeatCount="indefinite" keyTimes="{key_times}" values="{val_y}"/>'
            f'</circle>'
        )
    particles_joined = "\n          ".join(particle_elements)

    # Master opacity for travelling particles:
    # 0 during portrait (0s-3.0s), fades to 1 during transition (3.0-4.3s), stays 1, fades to 0 at return (12.9-14.2s)
    morph_layer_svg = f"""      <!-- LAYER 2: Travelling Morph Particles (900 particles, Optimal Transport) -->
      <g id="morph-particles-layer" transform="translate({PORTRAIT_X}, {PORTRAIT_Y})" opacity="0">
        <animate attributeName="opacity" dur="14.2s" repeatCount="indefinite"
                 keyTimes="{key_times}"
                 values="0; 0; 1; 1; 1; 1; 1; 1; 0"/>
        {particles_joined}
      </g>"""

    # D. Active Phase Label (below portrait frame)
    phase_label_svg = f"""      <!-- HUD Phase Indicator -->
      <g transform="translate({PORTRAIT_X + 150}, 498)">
        <rect x="-115" y="-16" width="230" height="23" rx="4" fill="{accent_cyan}12" stroke="{accent_secondary}" stroke-width="0.8" opacity="0.6"/>
        <text text-anchor="middle" y="0" font-family="'JetBrains Mono', monospace" font-size="10.5" font-weight="700" fill="{accent_cyan}" letter-spacing="1.5">
          VISUAL.MAP // PROFILE ACTIVE
        </text>
      </g>"""

    # E. Right Panel Rows
    system_info_rows_svg = build_system_info_rows(theme)

    full_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEWBOX_W} {VIEWBOX_H}" width="100%" height="auto" style="max-width:{VIEWBOX_W}px;display:block;margin:auto;">
  <defs>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&amp;display=swap');
      * {{
        box-sizing: border-box;
      }}
      text {{
        user-select: none;
      }}
      .sys-row text {{
        transition: fill 0.2s ease;
      }}
      .sys-row:hover text {{
        fill: {accent_cyan} !important;
      }}
    </style>
    <!-- Terminal Shadow Filter -->
    <filter id="term-shadow" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="0" dy="16" stdDeviation="24" flood-color="#000000" flood-opacity="0.45"/>
    </filter>
    <!-- Frame Clip Path -->
    <clipPath id="portrait-clip">
      <rect x="{PORTRAIT_X}" y="{PORTRAIT_Y}" width="{GRID_W}" height="{GRID_H}" rx="4"/>
    </clipPath>
  </defs>

  <!-- Banner Outer Canvas -->
  <rect width="{VIEWBOX_W}" height="{VIEWBOX_H}" fill="{bg_color}"/>

  <!-- Subtle Ambient Glow -->
  <circle cx="230" cy="280" r="240" fill="{accent_cyan}" opacity="{0.035 if is_dark else 0.05}"/>
  <circle cx="850" cy="300" r="320" fill="{portrait_color}" opacity="{0.025 if is_dark else 0.04}"/>

  <!-- Terminal Window Container -->
  <g id="terminal-window" filter="url(#term-shadow)">
    <!-- Terminal Outer Window -->
    <rect x="{TERM_X}" y="{TERM_Y}" width="{TERM_W}" height="{TERM_H}" rx="10" fill="{term_bg}" stroke="{term_border}" stroke-width="1.2"/>

    <!-- Terminal Titlebar -->
    <rect x="{TERM_X}" y="{TERM_Y}" width="{TERM_W}" height="46" rx="10" fill="{term_bg}"/>
    <line x1="{TERM_X}" y1="{TERM_Y + 46}" x2="{TERM_X + TERM_W}" y2="{TERM_Y + 46}" stroke="{header_border}" stroke-width="1"/>

    <!-- macOS Window Action Buttons -->
    <circle cx="{TERM_X + 22}" cy="{TERM_Y + 23}" r="5.5" fill="#EF4444"/>
    <circle cx="{TERM_X + 40}" cy="{TERM_Y + 23}" r="5.5" fill="#F59E0B"/>
    <circle cx="{TERM_X + 58}" cy="{TERM_Y + 23}" r="5.5" fill="#10B981"/>

    <!-- Terminal Window Title -->
    <text x="{TERM_X + 85}" y="{TERM_Y + 28}" fill="{accent_secondary}" font-family="'JetBrains Mono', monospace" font-size="13" font-weight="600" letter-spacing="0.5">
      profile.sh live
    </text>

    <!-- Pulsing LIVE Badge -->
    <g transform="translate({TERM_X + 910}, {TERM_Y + 23})">
      <circle cx="0" cy="0" r="4.5" fill="#EF4444">
        <animate attributeName="opacity" values="1;0.4;1" dur="1.8s" repeatCount="indefinite"/>
      </circle>
      <circle cx="0" cy="0" r="4.5" fill="none" stroke="#EF4444" stroke-width="1.2" opacity="0.8">
        <animate attributeName="r" values="4.5;10;4.5" dur="1.8s" repeatCount="indefinite"/>
        <animate attributeName="opacity" values="0.8;0;0.8" dur="1.8s" repeatCount="indefinite"/>
      </circle>
      <text x="14" y="4.5" fill="#EF4444" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="700" letter-spacing="1">LIVE</text>
    </g>

    <!-- User Handle Pill -->
    <g transform="translate({TERM_X + 990}, {TERM_Y + 12})">
      <rect x="0" y="0" width="138" height="24" rx="12" fill="{accent_cyan}18" stroke="{accent_cyan}" stroke-width="0.9"/>
      <a href="https://github.com/RakeshBabuGajula" target="_blank">
        <text x="69" y="16" text-anchor="middle" fill="{accent_cyan}" font-family="'JetBrains Mono', monospace" font-size="13" font-weight="700">@RakeshBabuGajula</text>
      </a>
    </g>

    <!-- Vertical Split Divider (38% left, 62% right) -->
    <line x1="{SPLIT_X}" y1="{TERM_Y + 46}" x2="{SPLIT_X}" y2="{TERM_Y + TERM_H}" stroke="{header_border}" stroke-width="1"/>

    <!-- ==================== LEFT PANEL: VISUAL.MAP ==================== -->
    <g id="left-panel">
      <!-- Section Header -->
      <text x="{PORTRAIT_X - 8}" y="98" fill="{accent_cyan}" font-family="'JetBrains Mono', monospace" font-size="13" font-weight="700" letter-spacing="1.5">
        VISUAL.MAP // SYNAPSE
      </text>
      <text x="{PORTRAIT_X + GRID_W + 8}" y="98" text-anchor="end" fill="{accent_secondary}" font-family="'JetBrains Mono', monospace" font-size="11" font-weight="500">
        [300x340]
      </text>

      <!-- Frame Box -->
      <rect x="{PORTRAIT_X - 10}" y="{PORTRAIT_Y - 10}" width="{GRID_W + 20}" height="{GRID_H + 20}" rx="6" fill="{frame_bg}" stroke="{term_border}" stroke-width="1"/>

      <!-- Corner Crosshair Ticks -->
      <path d="M{PORTRAIT_X - 14},{PORTRAIT_Y - 4}v-10h10 M{PORTRAIT_X + GRID_W + 14},{PORTRAIT_Y - 4}v-10h-10 M{PORTRAIT_X - 14},{PORTRAIT_Y + GRID_H + 4}v10h10 M{PORTRAIT_X + GRID_W + 14},{PORTRAIT_Y + GRID_H + 4}v10h-10" fill="none" stroke="{accent_cyan}" stroke-width="1.4"/>

      <!-- Portrait & Logo Morph Animation Container (Clipped) -->
      <g clip-path="url(#portrait-clip)">
        <!-- Background inside frame -->
        <rect x="{PORTRAIT_X}" y="{PORTRAIT_Y}" width="{GRID_W}" height="{GRID_H}" fill="{frame_bg}"/>

        {intro_layer_svg}
        {looping_drift_svg}
        {morph_layer_svg}
      </g>

      <!-- Bottom HUD readout -->
      {phase_label_svg}
      <text x="{PORTRAIT_X + 150}" y="534" text-anchor="middle" fill="{accent_secondary}" font-family="'JetBrains Mono', monospace" font-size="10.5" letter-spacing="0.5">
        VECTOR MORPH // 900 OPTIMAL TRANSPORT PARTICLES
      </text>
    </g>

    <!-- ==================== RIGHT PANEL: SYSTEM.INFO ==================== -->
    <g id="right-panel">
      <!-- Section Header -->
      <text x="485" y="98" fill="{accent_cyan}" font-family="'JetBrains Mono', monospace" font-size="13" font-weight="700" letter-spacing="1.5">
        SYSTEM.INFO // TELEMETRY &amp; PROFILE READOUT
      </text>
      <text x="{TERM_X + TERM_W - 20}" y="98" text-anchor="end" fill="{accent_emerald}" font-family="'JetBrains Mono', monospace" font-size="11" font-weight="600">
        ONLINE [2023-2027]
      </text>

      <line x1="485" y1="112" x2="{TERM_X + TERM_W - 20}" y2="112" stroke="{header_border}" stroke-width="0.8"/>

      <!-- Telemetry Rows -->
      {system_info_rows_svg}

      <!-- Bottom Status Bar -->
      <line x1="485" y1="504" x2="{TERM_X + TERM_W - 20}" y2="504" stroke="{header_border}" stroke-width="0.8"/>
      <g transform="translate(485, 526)" font-family="'JetBrains Mono', monospace" font-size="11" fill="{accent_secondary}">
        <text x="0" y="0">UPTIME: <tspan fill="{accent_emerald}" font-weight="700">99.98%</tspan></text>
        <text x="145" y="0">SECURITY: <tspan fill="{accent_cyan}" font-weight="700">TLS 1.3</tspan></text>
        <text x="300" y="0">TARGET: <tspan fill="{accent_cyan}" font-weight="700">SHIPPING</tspan></text>
        <text x="450" y="0">LOC: <tspan fill="{val_color}" font-weight="600">17.3850° N, 78.4867° E</tspan></text>
      </g>
    </g>
  </g>
</svg>"""

    out_file = OUTPUT_DIR / f"{theme}.svg"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(full_svg)
    print(f"[{theme.upper()}] Successfully generated banner: {out_file} ({len(full_svg):,} bytes)")
    return out_file


def main():
    print("Generating animated SVG GitHub Profile Banners for Rakesh Babu Gajula...")
    generate_svg_banner("dark")
    generate_svg_banner("light")
    print("All banners generated successfully!")


if __name__ == "__main__":
    main()
