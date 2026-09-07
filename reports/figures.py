"""Presentation figures.

Rendered on the deck palette defined in custom.scss so the plots sit flush on the
slide background instead of reading as pasted-in grey boxes.
"""

import json
import os
from pathlib import Path

from cycler import cycler
from dotenv import load_dotenv
import lightgbm as lgb
import matplotlib as mpl
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import polars as pl

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
DATA = ROOT / "data"
MODELS = ROOT / "models"
ARCHIVE = ROOT / "archive"
SETUP = DATA / "setup"
FEAT = DATA / "features"
MEMBER_DIR = MODELS / "members"
COMBINER_DIR = MODELS / "combiner"
FIGS = ROOT / "reports"
FIGS.mkdir(parents=True, exist_ok=True)

W_TOTAL = 5.368096
TARGET_MEAN = 2.3312
SIGMA = float(np.sqrt(W_TOTAL))
FINAL_LB = 1.6460816563
CAP_AT = 180.0
EXPERIMENT_ID = "6"

# ── Palette ─────────────────────────────────────────────────
# Ground and ink are taken verbatim from custom.scss. The categorical slots are
# the dataviz reference dark ramp with slot 1 swapped for the deck accent; the
# set passes lightness band / chroma floor / adjacent CVD / normal-vision /
# contrast on the #04070e surface. Slots 1-3 additionally clear the all-pairs
# gate, so scatter forms never use more than three of them.
BG = "#04070e"  # slide background
PANEL = "#0b1322"  # card fill
GRID = "#182541"  # hairline
GRID_SOFT = "#101a2e"  # quieter hairline
FG = "#e8eefa"  # body ink
FG_HI = "#ffffff"  # headings
MUTED = "#8098bd"  # secondary ink
STEEL = "#5c82b8"  # axis furniture
ICE = "#a9cdff"  # callout ink

S1 = "#3b82f6"  # blue    (deck accent)
S2 = "#d95926"  # orange
S3 = "#199e70"  # aqua
S4 = "#c98500"  # yellow
S5 = "#d55181"  # magenta
CATEGORICAL = [S1, S2, S3, S4, S5]

BLUE_HI = "#86b6ef"  # blue ordinal ramp, light -> dark
BLUE_MID = "#3b82f6"
BLUE_LO = "#1f4b8f"

mpl.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": FG,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": FG,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "font.family": "sans-serif",
        # Manrope is the deck face; Inter is the closest installed stand-in.
        "font.sans-serif": ["Manrope", "Inter", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "600",
        "axes.titlepad": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "legend.labelcolor": FG,
        "figure.titlesize": 14,
        "figure.titleweight": "600",
        "axes.prop_cycle": cycler(color=CATEGORICAL),
        "axes.axisbelow": True,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "lines.solid_capstyle": "round",
    }
)


def style(ax, *, grid="both"):
    """Recessive grid on one or both axes."""
    ax.grid(True, axis=grid, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    return ax


CONFIG = json.loads((SETUP / "config.json").read_text())
LEVELS = json.loads((SETUP / "levels.json").read_text())
SPEC = json.loads((COMBINER_DIR / "members.json").read_text())
NNLS_KEPT = json.loads((COMBINER_DIR / "nnls_kept.json").read_text())
SCORES = json.loads((ARCHIVE / "scores.json").read_text())
MEMBER_ORDER = SPEC["members"]
KEPT_MEMBERS = SPEC["kept"]
MEMBER_CV = {n: json.loads((MEMBER_DIR / n / "cv.json").read_text()) for n in MEMBER_ORDER}
ANCHORS = CONFIG["train_anchors"]
ANCHOR_IDX = {a: i for i, a in enumerate(ANCHORS)}
VAL_ANCHORS = [
    "2025-09-24",
    "2025-10-08",
    "2025-10-22",
    "2025-12-03",
    "2025-12-17",
    "2025-12-31",
    "2026-01-14",
]
EVAL_ANCHORS = ["2025-12-17", "2025-12-31", "2026-01-14"]

load_dotenv(ROOT / ".env")
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", ""))
INTERPRET = {}
try:
    client = mlflow.tracking.MlflowClient()
    for run in client.search_runs([EXPERIMENT_ID], max_results=50):
        if run.info.run_name == "prod-interpret" and run.info.status == "FINISHED":
            INTERPRET = dict(run.data.metrics)
            break
    print(f"mlflow prod-interpret metrics: {len(INTERPRET)}")
except Exception as exc:
    print(f"mlflow unavailable, skipping mlflow-backed figures: {exc}")


def save(fig, name):
    fig.savefig(FIGS / name, facecolor=BG)
    plt.close(fig)
    print(f"wrote reports/{name}")


# ── Leaderboard timeline ────────────────────────────────────

LB_MILESTONES = {
    "catboost_v1": "v1 CatBoost",
    "catboost_v2": "v2, +141 features",
    "v3_seasonal": "v3, centring per anchor",
    "v6_vall": "v6, 21-member blend",
    "v11_spaced": "v11, spaced anchors",
    "v21_banked": "v21",
    "v32_seqopt": "v32, sequence direction",
    "v41_ridge": "v41, archive solve",
    "v43_calib": "v43, two-pass calibration",
}
lb_names = list(SCORES)
lb_vals = [float(SCORES[n][1]) for n in lb_names]
lb_names.append("v48_newdir")
lb_vals.append(FINAL_LB)
LB_MILESTONES["v48_newdir"] = "v48, shipped"
steps = np.arange(len(lb_vals))
best = np.minimum.accumulate(lb_vals)

# Milestone label placement. Early labels sit at a small pixel offset from their
# point; the four end-of-run milestones bunch into the same corner, so they are
# parked as a right-aligned column in the empty band under the plateau (explicit
# data coordinates) and joined to their points with thin leaders.
LB_LABEL_OFFSET = {
    "catboost_v1": (12, 6, "left"),
    "catboost_v2": (12, 6, "left"),
    "v3_seasonal": (12, -20, "left"),
    "v6_vall": (18, 11, "left"),
    "v11_spaced": (14, 9, "left"),
    "v21_banked": (14, 13, "left"),
}
LB_LABEL_ANCHOR = {
    "v32_seqopt": (49.0, 1.64700),
    "v41_ridge": (49.0, 1.64645),
    "v43_calib": (49.0, 1.64590),
    "v48_newdir": (49.0, 1.64535),
}

fig, ax = plt.subplots(1, 2, figsize=(14, 5.0), layout="constrained")
ax[0].plot(steps, lb_vals, "o", ms=4.5, color=S1, alpha=0.75, label="submission", zorder=3)
ax[0].plot(steps, best, "-", lw=2.2, color=S2, label="best so far", zorder=4)
ax[0].set_ylabel("public LB, RMSLE")
ax[0].set_xlabel("submission order")
ax[0].set_title(f"{len(lb_vals)} scored submissions, {max(lb_vals):.3f} to {min(lb_vals):.3f}")
ax[0].legend(loc="upper right")
ax[0].set_xlim(-2.5, steps.max() + 2.5)

zoom = np.array(lb_vals) < 1.6510
ax[1].plot(steps[zoom], np.array(lb_vals)[zoom], "o", ms=5, color=S1, alpha=0.75, zorder=3)
ax[1].plot(steps, best, "-", lw=2.2, color=S2, zorder=4)
ax[1].set_ylim(1.6452, 1.6513)
ax[1].set_xlim(steps[zoom].min() - 2, steps.max() + 3)
ax[1].set_xlabel("submission order")
ax[1].set_ylabel("public LB, RMSLE")
ax[1].set_title(f"the last 0.005, where {int(zoom.sum())} of the submissions went")

LEADER = {"arrowstyle": "-", "color": STEEL, "lw": 0.7, "shrinkA": 2, "shrinkB": 4}
for key, label in LB_MILESTONES.items():
    if key not in lb_names:
        continue
    i = lb_names.index(key)
    target = ax[1] if lb_vals[i] < 1.6510 else ax[0]
    common = {"fontsize": 9.5, "color": ICE, "va": "center", "zorder": 5}
    if key in LB_LABEL_ANCHOR:
        target.annotate(
            label,
            (i, lb_vals[i]),
            xytext=LB_LABEL_ANCHOR[key],
            textcoords="data",
            ha="right",
            arrowprops=LEADER,
            **common,
        )
    else:
        dx, dy, ha = LB_LABEL_OFFSET[key]
        target.annotate(
            label,
            (i, lb_vals[i]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            arrowprops=LEADER if abs(dy) > 12 else None,
            **common,
        )
for a in ax:
    style(a)
save(fig, "lb_timeline.png")

# ── Anchor levels ───────────────────────────────────────────

lv_anchor = [entry["anchor"] for entry in LEVELS]
lv_n = np.array([entry["n_cohort"] for entry in LEVELS], dtype=float)
lv_pos = np.array([entry["p_positive"] for entry in LEVELS])
lv_mu = np.array([entry["mean_log1p"] for entry in LEVELS])
xs = np.arange(len(LEVELS))
held_from = len(LEVELS) - 3.5

fig, ax = plt.subplots(1, 3, figsize=(15, 5.9), layout="constrained")
panels = [
    (lv_n / 1000.0, S1, "cohort size", "users, thousands"),
    (lv_pos, S3, "share of the cohort with positive target", ""),
    (lv_mu, S4, "mu(anchor) = mean log1p(gmv) over 30 days", ""),
]
for a, (series, color, title, ylabel) in zip(ax, panels, strict=True):
    a.axvspan(held_from, xs.max() + 0.6, color=STEEL, alpha=0.16, lw=0, zorder=0)
    a.plot(xs, series, "o-", ms=4, color=color, zorder=3)
    a.set_title(title)
    a.set_ylabel(ylabel)
    a.set_xlim(-0.6, xs.max() + 0.6)
    a.set_xticks(xs[::4])
    a.set_xticklabels([lv_anchor[i][2:] for i in xs[::4]], rotation=45, ha="right", fontsize=9)
    style(a)

# One horizontal callout for the held-out band, inside the shaded region.
ax[0].annotate(
    "3 held-out\nanchors",
    xy=((held_from + xs.max() + 0.6) / 2, 0.035),
    xycoords=("data", "axes fraction"),
    ha="center",
    va="bottom",
    fontsize=9.5,
    color=ICE,
    linespacing=1.35,
)
fig.suptitle(
    f"{len(LEVELS)} training anchors: the level moves with the calendar, "
    "the panel grows by selection"
)
save(fig, "anchor_levels.png")

# ── Member weights ──────────────────────────────────────────

# Colour follows the model family, so a member keeps its hue across both panels.
# The three gradient-boosting libraries are steps of one blue ordinal ramp; the
# scatter collapses to the three family hues, which is the all-pairs-safe set.
KIND_OF = {}
for name in MEMBER_ORDER:
    if name.startswith(("cat", "v3_cat")):
        KIND_OF[name] = "catboost"
    elif name.startswith("xgb"):
        KIND_OF[name] = "xgboost"
    elif name.startswith("lgb"):
        KIND_OF[name] = "lightgbm"
    elif name.startswith(("seq", "v3_seq")):
        KIND_OF[name] = "transformer"
    else:
        KIND_OF[name] = "neural tabular"
KIND_COLOR = {
    "catboost": BLUE_HI,
    "xgboost": BLUE_MID,
    "lightgbm": BLUE_LO,
    "transformer": S2,
    "neural tabular": S3,
}
FAMILY_OF = {
    "catboost": "gradient boosting",
    "xgboost": "gradient boosting",
    "lightgbm": "gradient boosting",
    "transformer": "transformer",
    "neural tabular": "neural tabular",
}
FAMILY_COLOR = {"gradient boosting": S1, "transformer": S2, "neural tabular": S3}
FAMILY_MARKER = {"gradient boosting": "o", "transformer": "s", "neural tabular": "^"}

w_kept = np.zeros(len(MEMBER_ORDER))
w_kept[np.array(NNLS_KEPT["columns"])] = np.array(NNLS_KEPT["weights"])
order = np.argsort(-w_kept)[: len(KEPT_MEMBERS)][::-1]
cv_all = np.array([MEMBER_CV[n]["POOLED_ALL"] for n in MEMBER_ORDER])

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained")
bar_colors = [KIND_COLOR[KIND_OF[MEMBER_ORDER[i]]] for i in order]
ax[0].barh(range(len(order)), w_kept[order], color=bar_colors, height=0.72)
ax[0].set_yticks(range(len(order)), [MEMBER_ORDER[i] for i in order], fontsize=10)
ax[0].set_ylim(-0.7, len(order) - 0.3)
ax[0].set_xlabel("non-negative least squares weight")
ax[0].set_title(f"{len(KEPT_MEMBERS)} of {len(MEMBER_ORDER)} members survive the solve")
ax[0].set_xlim(0, w_kept.max() * 1.14)
for rank, i in enumerate(order):
    ax[0].annotate(
        f"{w_kept[i]:.3f}",
        (w_kept[i], rank),
        xytext=(5, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=MUTED,
    )
kept_kinds = [k for k in KIND_COLOR if any(KIND_OF[MEMBER_ORDER[i]] == k for i in order)]
ax[0].legend(
    handles=[mpl.patches.Patch(facecolor=KIND_COLOR[k], label=k) for k in kept_kinds],
    loc="lower right",
    fontsize=9,
)
style(ax[0], grid="x")

for family, color in FAMILY_COLOR.items():
    m = np.array([FAMILY_OF[KIND_OF[n]] == family for n in MEMBER_ORDER])
    ax[1].scatter(
        cv_all[m],
        w_kept[m],
        s=70,
        marker=FAMILY_MARKER[family],
        facecolor=color,
        edgecolor=BG,
        linewidth=1.2,
        label=family,
        alpha=0.95,
        zorder=3,
    )
ax[1].set_xlabel("pooled out-of-fold RMSLE over 7 anchors")
ax[1].set_ylabel("weight in the blend")
ax[1].set_title("a better member is not a more useful one")
n_zero = int((w_kept == 0).sum())
ax[1].set_ylim(-0.034, w_kept.max() * 1.12)
ax[1].set_yticks(np.arange(0.0, w_kept.max() * 1.12, 0.025))
ax[1].annotate(
    f"{n_zero} members solve to exactly 0",
    xy=(cv_all[w_kept == 0].mean(), 0.0),
    xytext=(cv_all[w_kept == 0].mean(), -0.024),
    textcoords="data",
    ha="center",
    va="center",
    fontsize=9.5,
    color=ICE,
    arrowprops={"arrowstyle": "-", "color": STEEL, "lw": 0.7, "shrinkA": 3, "shrinkB": 6},
)
ax[1].legend(loc="upper right")
ax[1].margins(x=0.06)
style(ax[1])
save(fig, "member_weights.png")

# ── Out-of-fold panel ───────────────────────────────────────

OOF_Z, OOF_Y = {}, {}
COHORT = np.load(SETUP / "cohort.npy")
TARGETS = np.load(SETUP / "targets.npy")
for a in VAL_ANCHORS:
    m = COHORT[ANCHOR_IDX[a]]
    cols = []
    for n in MEMBER_ORDER:
        raw = np.load(MEMBER_DIR / n / f"oof_{a}.npy").astype(np.float64)[m]
        cols.append((raw - raw.mean()) / raw.std())
    OOF_Z[a] = np.column_stack(cols)
    OOF_Y[a] = np.log1p(TARGETS[ANCHOR_IDX[a]][m])

booster = lgb.Booster(model_file=str(COMBINER_DIR / SPEC["file"]["kept"]))
cols_kept = np.array(NNLS_KEPT["columns"])
w_nnls = np.array(NNLS_KEPT["weights"], dtype=np.float64)
OOF_PRED, BOOST_GAIN = {}, {}
for a in VAL_ANCHORS:
    z = OOF_Z[a][:, cols_kept]
    lz = z @ w_nnls
    p = (lz - lz.mean()) / lz.std()
    beta, level = np.polyfit(p, OOF_Y[a], 1)
    OOF_PRED[a] = np.clip(beta * p + level, 0.0, None)
    q = lz + booster.predict(np.column_stack([z, lz]))
    q = (q - q.mean()) / q.std()
    beta_q, level_q = np.polyfit(q, OOF_Y[a], 1)
    BOOST_GAIN[a] = float(
        np.sqrt(((OOF_PRED[a] - OOF_Y[a]) ** 2).mean())
        - np.sqrt(((np.clip(beta_q * q + level_q, 0.0, None) - OOF_Y[a]) ** 2).mean())
    )
print(
    "residual combiner, rmsle gain out of fold: "
    + str({k: round(v, 6) for k, v in BOOST_GAIN.items()})
)

# ── Blend calibration ───────────────────────────────────────

fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
for a, color in zip(EVAL_ANCHORS, CATEGORICAL, strict=False):
    y, pr = OOF_Y[a], OOF_PRED[a]
    edges = np.quantile(pr, np.linspace(0, 1, 21))
    idx = np.clip(np.digitize(pr, edges[1:-1]), 0, 19)
    centres = [pr[idx == i].mean() for i in range(20)]
    ax[0].plot(centres, [y[idx == i].mean() for i in range(20)], "o-", ms=4, color=color, label=a)
    ax[1].plot(centres, [(y[idx == i] == 0).mean() for i in range(20)], "o-", ms=4, color=color)
    ax[2].plot(centres, [y[idx == i].std() for i in range(20)], "o-", ms=4, color=color)
ax[0].plot([0, 6], [0, 6], "--", lw=1.2, color=STEEL, zorder=1, label="perfect calibration")
ax[0].set_xlabel("predicted log1p gmv")
ax[0].set_ylabel("realised log1p gmv")
ax[0].set_title("calibration by ventile")
ax[1].set_xlabel("predicted log1p gmv")
ax[1].set_ylabel("share with zero gmv")
ax[1].set_title("zero mass against the prediction")
ax[2].set_xlabel("predicted log1p gmv")
ax[2].set_ylabel("realised sd")
ax[2].set_title("irreducible spread")
for a_ in ax:
    style(a_)
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper right", ncols=4, fontsize=10)
save(fig, "blend_calibration.png")

# ── Error structure ─────────────────────────────────────────

a = EVAL_ANCHORS[-1]
y, pr = OOF_Y[a], OOF_PRED[a]
res = pr - y
BUCKETS = {
    "y = 0\npredicted low": (y == 0) & (pr <= 3.0),
    "y = 0\npredicted high": (y == 0) & (pr > 3.0),
    "y and prediction\nboth mid": (y > 0) & (y <= 5.0) & (pr > 1.0) & (pr <= 4.0),
    "y high\npredicted high": (y > 5.0) & (pr >= 4.0),
    "y high\npredicted zero": (y > 5.0) & (pr < 1.0),
}
total = float((res**2).sum())
share_err = [float((res[m] ** 2).sum()) / total for m in BUCKETS.values()]
share_n = [float(m.mean()) for m in BUCKETS.values()]

fig, ax = plt.subplots(figsize=(11, 5.0), layout="constrained")
pos = np.arange(len(BUCKETS))
ax.bar(pos - 0.21, share_n, 0.38, color=S1, label="share of users")
ax.bar(pos + 0.21, share_err, 0.38, color=S2, label="share of squared error")
for i, (sn, se) in enumerate(zip(share_n, share_err, strict=True)):
    ax.annotate(
        f"{sn:.1%}",
        (i - 0.21, sn),
        ha="center",
        fontsize=9.5,
        color=MUTED,
        xytext=(0, 4),
        textcoords="offset points",
    )
    ax.annotate(
        f"{se:.1%}",
        (i + 0.21, se),
        ha="center",
        fontsize=9.5,
        color=MUTED,
        xytext=(0, 4),
        textcoords="offset points",
    )
ax.set_xticks(pos, list(BUCKETS), fontsize=10)
ax.set_ylim(0, max(*share_n, *share_err) * 1.14)
ax.set_ylabel("share")
ax.set_title(f"where the error lives, anchor {a}, held-out RMSLE {np.sqrt((res**2).mean()):.4f}")
ax.legend(loc="upper right")
style(ax, grid="y")
save(fig, "error_structure.png")

# ── Feature families ────────────────────────────────────────

if INTERPRET:
    perm = {k[len("perm_group_") :]: v for k, v in INTERPRET.items() if k.startswith("perm_group_")}
    fam = {
        k[len("family_share_") :]: v for k, v in INTERPRET.items() if k.startswith("family_share_")
    }
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained")
    for a_, data, xlabel, title in (
        (
            ax[0],
            perm,
            "mean absolute shift of the blend, z units",
            "group permutation on 20000 users",
        ),
        (ax[1], fam, "share of mean absolute attribution", "attribution share, same families"),
    ):
        keys = sorted(data, key=lambda k: data[k])
        vals = [data[k] for k in keys]
        a_.barh(
            range(len(keys)),
            vals,
            height=0.7,
            color=[S2 if k == "daily_tensor" else S1 for k in keys],
        )
        a_.set_yticks(range(len(keys)), keys, fontsize=10)
        a_.set_ylim(-0.7, len(keys) - 0.3)
        a_.set_xlim(0, max(vals) * 1.06)
        a_.set_xlabel(xlabel)
        a_.set_title(title)
        style(a_, grid="x")
    ax[0].legend(
        handles=[
            mpl.patches.Patch(facecolor=S2, label="raw daily tensor"),
            mpl.patches.Patch(facecolor=S1, label="tabular family"),
        ],
        loc="lower right",
        fontsize=9,
    )
    save(fig, "feature_families.png")

# ── Exchange rate ───────────────────────────────────────────

names_a, rho_a, lb_a = [], [], []
for key, (stem, lb) in SCORES.items():
    path = ARCHIVE / f"{stem}.csv"
    if not path.exists():
        continue
    lg = np.log1p(
        np.clip(pl.read_csv(path).sort("user_id")["predict"].to_numpy().astype(np.float64), 0, None)
    )
    level, beta = float(lg.mean()), float(lg.std())
    names_a.append(key)
    lb_a.append(float(lb))
    rho_a.append(
        (beta * beta + W_TOTAL + (level - TARGET_MEAN) ** 2 - lb * lb) / (2 * beta * SIGMA)
    )
b_a = np.asarray(rho_a) * SIGMA
lb_a = np.asarray(lb_a)
b_grid = np.linspace(1.55, 1.68, 400)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained")
ax[0].plot(b_grid, np.sqrt(W_TOTAL - b_grid**2), "-", lw=2.2, color=S1, label=r"$\sqrt{W - B^2}$")
ax[0].scatter(
    b_a,
    lb_a,
    s=44,
    color=S2,
    alpha=0.9,
    edgecolor=BG,
    linewidth=0.8,
    label="scored submissions",
    zorder=3,
)
ax[0].set_xlabel("B = Cov(z, u), recovered from the score")
ax[0].set_ylabel("public LB, RMSLE")
ax[0].set_title(f"W = {W_TOTAL}, m = {TARGET_MEAN}, pinned by two constant probes")
ax[0].legend(loc="upper right")
ax[0].set_xlim(1.618, 1.6325)
ax[0].set_ylim(1.6455, 1.6575)

gains = np.array([0.0, 0.0025, 0.005, 0.01])
b0 = 1.6270516
gain_pct = gains * 100
gain_lb = np.sqrt(W_TOTAL - (b0 * (1 + gains)) ** 2)
ax[1].plot(gain_pct, gain_lb, "o-", lw=2.2, ms=7, color=S2, zorder=3)
for i, (gx, gy) in enumerate(zip(gain_pct, gain_lb, strict=True)):
    last = i == len(gains) - 1
    ax[1].annotate(
        f"{gy:.6f}",
        (gx, gy),
        textcoords="offset points",
        xytext=(-10, -16) if last else (10, 8),
        ha="right" if last else "left",
        fontsize=10,
        color=ICE,
    )
ax[1].set_xlabel("relative gain in B, percent")
ax[1].set_ylabel("public LB, RMSLE")
ax[1].set_title("exchange rate: 1 percent of B is worth 0.0165 RMSLE")
ax[1].margins(x=0.10, y=0.14)
for a_ in ax:
    style(a_)
save(fig, "exchange_rate.png")

# ── Segments ────────────────────────────────────────────────

x_names = json.loads((FEAT / "names_x.json").read_text())
e_names = json.loads((FEAT / "names_e.json").read_text())
keep_idx = np.load(FEAT / "keep_idx.npy")
base_names = [x_names[j] for j in keep_idx] + e_names
col = {n: i for i, n in enumerate(base_names)}
submit = CONFIG["submit_anchor"]
xb = np.asarray(np.load(FEAT / "x" / f"X_{submit}.npy", mmap_mode="r"))[:, keep_idx]
xb = np.concatenate([xb, np.asarray(np.load(FEAT / "e" / f"E_{submit}.npy", mmap_mode="r"))], 1)
ord365 = xb[:, col["has_order_sum_365d"]]
rec = np.minimum(xb[:, col["days_since_last_order"]], CAP_AT)
prev = xb[:, col["days_since_prev_order"]]
act30 = xb[:, col["active_sum_30d"]]
sea30 = xb[:, col["searches_sum_30d"]]
ten = np.minimum(xb[:, col["tenure_days"]], CAP_AT)
g365 = xb[:, col["gmv_sum_365d"]]
SEGMENTS = {
    "whale": g365 >= np.quantile(g365, 0.99),
    "regular": (ord365 >= 6) & (rec <= 30),
    "lapsing": (ord365 >= 3) & (rec >= 90),
    "never_ordered": ord365 == 0,
    "search_only_no_order": (ord365 == 0) & (sea30 >= np.quantile(sea30[sea30 > 0], 0.5)),
    "reactivated": (rec <= 14) & (prev >= 90),
    "thin_history": ten <= 90,
    "barely_active": act30 <= 2,
    "cart_no_order": (xb[:, col["has_cart_sum_90d"]] > 0) & (xb[:, col["has_order_sum_90d"]] == 0),
    "single_order": ord365 == 1,
}
shares = {s: float(m.mean()) for s, m in SEGMENTS.items()}
(FIGS / "segments.json").write_text(json.dumps(shares, indent=1))
print("wrote reports/segments.json")

if INTERPRET:
    seg = {k[len("seg_rmsle_") :]: v for k, v in INTERPRET.items() if k.startswith("seg_rmsle_")}
    keys = sorted(seg, key=lambda k: seg[k])
    pooled = INTERPRET["eval_rmsle"]
    # Labels live in a reserved column to the right of every bar, so the pooled
    # rule never crosses text.
    label_x = max(seg.values()) * 1.14
    fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
    ax.barh(
        range(len(keys)),
        [seg[k] for k in keys],
        height=0.72,
        color=[S1 if seg[k] < pooled else S2 for k in keys],
        zorder=2,
    )
    ax.set_yticks(range(len(keys)), keys, fontsize=11)
    ax.set_ylim(-0.7, len(keys) + 0.5)
    for i, k in enumerate(keys):
        ax.annotate(
            f"{seg[k]:.3f}",
            (label_x, i),
            va="center",
            ha="left",
            fontsize=11,
            color=FG,
        )
        ax.annotate(
            f"{shares[k]:.1%}",
            (label_x + 0.34, i),
            va="center",
            ha="left",
            fontsize=11,
            color=MUTED,
        )
    for x, header in ((label_x, "RMSLE"), (label_x + 0.34, "of cohort")):
        ax.annotate(
            header,
            (x, len(keys) - 0.35),
            va="center",
            ha="left",
            fontsize=9.5,
            color=STEEL,
        )
    ax.axvline(pooled, color=ICE, ls="--", lw=1.4, zorder=3)
    ax.annotate(
        f"pooled {pooled:.4f}",
        (pooled, len(keys) - 0.55),
        xytext=(-8, 0),
        textcoords="offset points",
        color=ICE,
        fontsize=10,
        ha="right",
        va="center",
    )
    ax.set_xlim(0, label_x + 0.72)
    ax.set_xticks(np.arange(0, 2.1, 0.5))
    ax.set_xlabel("held-out RMSLE over the 3 evaluation anchors")
    ax.set_title("error by behavioural segment, segments overlap")
    # outside the axes: inside, the pooled rule cuts straight through the labels
    fig.legend(
        handles=[
            mpl.patches.Patch(facecolor=S2, label="worse than pooled"),
            mpl.patches.Patch(facecolor=S1, label="better than pooled"),
        ],
        loc="outside upper left",
        ncols=2,
        fontsize=9.5,
    )
    style(ax, grid="x")
    save(fig, "segment_rmsle.png")
