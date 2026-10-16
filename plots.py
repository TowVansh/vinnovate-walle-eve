"""Matplotlib figures for missions and the benchmark (python walle.py run/bench write them automatically)."""
import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


def plot_mission(cfg, env, agent, trace, path):
    """Map with the path coloured by time, events marked, final belief heatmap, and battery over time."""
    W, H = cfg["map"]["width"], cfg["map"]["height"]
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.2])
    ax, bx, cx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    for a, title in ((ax, "Path taken (colour = time)"), (bx, "Final belief: where WALL-E thinks EVE is")):
        a.set_xlim(-0.5, W - 0.5); a.set_ylim(H - 0.5, -0.5); a.set_aspect("equal")
        a.set_xticks(range(0, W, 2)); a.set_yticks(range(0, H, 2)); a.set_title(title)
        for x, y in cfg["map"].get("walls", []):
            a.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, color="#555"))

    # belief heatmap
    grid = [[0.0] * W for _ in range(H)]
    for (x, y), p in agent.belief.items():
        grid[y][x] = p
    im = bx.imshow(grid, cmap="magma", origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5), zorder=0)
    fig.colorbar(im, ax=bx, fraction=0.035, label="P(EVE in cell)")

    # path as a time-coloured line
    pts = [tuple(r["obs"]["pos"]) for r in trace]
    pts = [tuple(cfg["start"])] + pts
    segs = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1) if pts[i] != pts[i + 1]]
    lc = LineCollection(segs, cmap="viridis", linewidths=3)
    lc.set_array(list(range(len(segs))))
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, fraction=0.035, label="move number")

    scans = [tuple(r["obs"]["pos"]) for r in trace if r["action"] == "scan"]
    fails = [tuple(r["obs"]["pos"]) for r in trace if r["action"] == "contact" and not r["obs"].get("contact")]
    for a in (ax, bx):
        for c in cfg.get("clues", []):
            a.plot(*c["pos"], "s", color="#3b82f6", ms=9)
        for d in cfg.get("decoys", []):
            a.plot(*d["pos"], "X", color="#ef4444", ms=13)
        for z in agent.decoy_zones:
            a.add_patch(plt.Circle(z, 3, fill=False, ls="--", color="#ef4444", lw=1.5))
        a.plot(*cfg["base"], "^", color="#22c55e", ms=13)
        a.plot(*cfg["eve"]["pos"], "*", color="#ec4899", ms=20, mec="white")
    if scans:
        ax.plot(*zip(*scans), "o", mfc="none", mec="#facc15", ms=6)
    if fails:
        ax.plot(*zip(*fails), "x", color="k", ms=10, mew=2.5)
    ax.legend(handles=[
        Line2D([], [], marker="^", color="#22c55e", ls="", ms=10, label="base / recharge"),
        Line2D([], [], marker="*", color="#ec4899", ls="", ms=14, label="EVE (truth)"),
        Line2D([], [], marker="s", color="#3b82f6", ls="", ms=8, label="clue"),
        Line2D([], [], marker="X", color="#ef4444", ls="", ms=10, label="decoy beacon"),
        Line2D([], [], marker="o", mfc="none", mec="#facc15", ls="", ms=7, label="scan"),
        Line2D([], [], marker="x", color="k", ls="", ms=8, label="failed contact"),
        Line2D([], [], color="#ef4444", ls="--", label="detected decoy zone"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=8, frameon=False)

    # battery and belief confidence over time
    steps = [r["t"] for r in trace]
    cx.plot(steps, [r["obs"]["battery"] for r in trace], color="#22c55e", label="battery")
    for r in trace:
        if r["action"] == "recharge":
            cx.axvline(r["t"], color="#22c55e", alpha=.3, lw=4)
    cx2 = cx.twinx()
    cx2.plot(steps, [r["top_belief"][0][1] for r in trace], color="#a855f7", label="max belief")
    cx2.set_ylim(0, 1.05); cx2.set_ylabel("max P(EVE)")
    cx.set_xlabel("step"); cx.set_ylabel("battery"); cx.set_title("Battery (green bands = recharges) and confidence over time")
    cx.legend(loc="upper left"); cx2.legend(loc="upper right")

    fig.suptitle(f"{cfg.get('name', 'mission')}: {'EVE FOUND' if env.found else 'EVE NOT FOUND'} in {len(trace)} steps", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_bench(result, path):
    """Success rate by agent and mission type."""
    agents = list(result)
    kinds = [("success_no_decoy", "no decoy"), ("success_with_decoy", "with decoy"), ("success_rate", "overall")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.38
    for i, a in enumerate(agents):
        vals = [result[a][k] * 100 for k, _ in kinds]
        bars = ax.bar([j + i * w for j in range(len(kinds))], vals, w, label=a, color=["#a855f7", "#94a3b8"][i % 2])
        ax.bar_label(bars, fmt="%.0f%%", fontsize=9)
    ax.set_xticks([j + w / 2 for j in range(len(kinds))], [k for _, k in kinds])
    ax.set_ylim(0, 110); ax.set_ylabel("missions where EVE was found (%)")
    ax.set_title(f"Bayesian agent vs reactive baseline, {result[agents[0]]['missions']} random missions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
