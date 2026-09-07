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

BG = "#0d0d10"
FG = "#e8e8ec"
GRID = "#2a2a33"
ACCENT = "#4cc9f0"
WARM = "#f7768e"
GOLD = "#e0af68"
GREEN = "#9ece6a"
VIOLET = "#bb9af7"

mpl.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": FG,
        "axes.titlecolor": FG,
        "text.color": FG,
        "xtick.color": FG,
        "ytick.color": FG,
        "grid.color": GRID,
        "font.size": 9,
        "axes.titlesize": 11,
        "legend.frameon": False,
        "legend.labelcolor": FG,
        "axes.prop_cycle": cycler(color=[ACCENT, GOLD, GREEN, WARM, VIOLET]),
        "axes.axisbelow": True,
    }
)

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
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote reports/{name}")


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

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].plot(steps, lb_vals, "o", ms=3.5, color=ACCENT, alpha=0.55, label="submission")
ax[0].plot(steps, best, "-", lw=2, color=WARM, label="best so far")
ax[0].set_ylabel("public LB, RMSLE")
ax[0].set_xlabel("submission order")
ax[0].set_title("64 scored submissions, 1.840 to 1.646")
ax[0].legend(loc="upper right")
zoom = np.array(lb_vals) < 1.6510
ax[1].plot(steps[zoom], np.array(lb_vals)[zoom], "o", ms=4, color=ACCENT, alpha=0.55)
ax[1].plot(steps, best, "-", lw=2, color=WARM)
ax[1].set_ylim(1.6455, 1.6510)
ax[1].set_xlim(steps[zoom].min() - 1, steps.max() + 1)
ax[1].set_xlabel("submission order")
ax[1].set_title(f"the last 0.005, where {int(zoom.sum())} of the submissions went")
for key, label in LB_MILESTONES.items():
    if key not in lb_names:
        continue
    i = lb_names.index(key)
    target = ax[1] if lb_vals[i] < 1.6510 else ax[0]
    target.annotate(
        label,
        (i, lb_vals[i]),
        textcoords="offset points",
        xytext=(6, 7),
        fontsize=7,
        color=GOLD,
    )
for a in ax:
    a.grid(alpha=0.25)
save(fig, "lb_timeline.png")

lv_anchor = [entry["anchor"] for entry in LEVELS]
lv_n = np.array([entry["n_cohort"] for entry in LEVELS], dtype=float)
lv_pos = np.array([entry["p_positive"] for entry in LEVELS])
lv_mu = np.array([entry["mean_log1p"] for entry in LEVELS])
xs = np.arange(len(LEVELS))

fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
ax[0].plot(xs, lv_n / 1000.0, "o-", ms=3, color=ACCENT)
ax[0].set_title("cohort size, thousands")
ax[0].set_ylabel("users with 30d activity and 60d history")
ax[1].plot(xs, lv_pos, "o-", ms=3, color=GOLD)
ax[1].set_title("share of the cohort with positive target")
ax[2].plot(xs, lv_mu, "o-", ms=3, color=WARM)
ax[2].set_title("mu(anchor) = mean log1p(gmv) over 30 days")
for a in ax:
    a.set_xticks(xs[::4])
    a.set_xticklabels([lv_anchor[i][2:] for i in xs[::4]], rotation=45, fontsize=7)
    a.grid(alpha=0.25)
    a.axvline(len(LEVELS) - 3.5, color=VIOLET, ls="--", lw=1)
ax[0].annotate(
    "3 held-out anchors",
    (len(LEVELS) - 3.4, lv_n.min() / 1000.0),
    color=VIOLET,
    fontsize=7,
    rotation=90,
)
fig.suptitle(
    f"{len(LEVELS)} training anchors: the level moves with the calendar, "
    "the panel grows by selection",
    y=1.04,
)
save(fig, "anchor_levels.png")

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
    "catboost": ACCENT,
    "xgboost": GREEN,
    "lightgbm": GOLD,
    "transformer": WARM,
    "neural tabular": VIOLET,
}
w_kept = np.zeros(len(MEMBER_ORDER))
w_kept[np.array(NNLS_KEPT["columns"])] = np.array(NNLS_KEPT["weights"])
order = np.argsort(-w_kept)[: len(KEPT_MEMBERS)][::-1]
cv_all = np.array([MEMBER_CV[n]["POOLED_ALL"] for n in MEMBER_ORDER])

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].barh(
    range(len(order)),
    w_kept[order],
    color=[KIND_COLOR[KIND_OF[MEMBER_ORDER[i]]] for i in order],
)
ax[0].set_yticks(range(len(order)), [MEMBER_ORDER[i] for i in order], fontsize=8)
ax[0].set_xlabel("non-negative least squares weight")
ax[0].set_title(f"{len(KEPT_MEMBERS)} of {len(MEMBER_ORDER)} members survive the solve")
for kind, color in KIND_COLOR.items():
    m = np.array([KIND_OF[n] == kind for n in MEMBER_ORDER])
    ax[1].scatter(cv_all[m], w_kept[m], s=42, color=color, label=kind, alpha=0.9)
ax[1].set_xlabel("pooled out-of-fold RMSLE over 7 anchors")
ax[1].set_ylabel("weight in the blend")
ax[1].set_title("a better member is not a more useful one")
ax[1].legend(fontsize=8, loc="upper right")
for a in ax:
    a.grid(alpha=0.25)
save(fig, "member_weights.png")

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

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1))
for a in EVAL_ANCHORS:
    y, pr = OOF_Y[a], OOF_PRED[a]
    edges = np.quantile(pr, np.linspace(0, 1, 21))
    idx = np.clip(np.digitize(pr, edges[1:-1]), 0, 19)
    ax[0].plot(
        [pr[idx == i].mean() for i in range(20)],
        [y[idx == i].mean() for i in range(20)],
        "o-",
        ms=3,
        label=a,
    )
    ax[1].plot(
        [pr[idx == i].mean() for i in range(20)],
        [(y[idx == i] == 0).mean() for i in range(20)],
        "o-",
        ms=3,
    )
    ax[2].plot(
        [pr[idx == i].mean() for i in range(20)],
        [y[idx == i].std() for i in range(20)],
        "o-",
        ms=3,
    )
ax[0].plot([0, 6], [0, 6], "--", lw=1, color=GRID)
ax[0].set_xlabel("predicted log1p gmv")
ax[0].set_ylabel("realised")
ax[0].set_title("calibration by ventile")
ax[0].legend(fontsize=7)
ax[1].set_xlabel("predicted log1p gmv")
ax[1].set_ylabel("share with zero gmv")
ax[1].set_title("zero mass against the prediction")
ax[2].set_xlabel("predicted log1p gmv")
ax[2].set_ylabel("realised sd")
ax[2].set_title("irreducible spread")
for a in ax:
    a.grid(alpha=0.25)
save(fig, "blend_calibration.png")

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

fig, ax = plt.subplots(figsize=(9.5, 4.3))
pos = np.arange(len(BUCKETS))
ax.bar(pos - 0.2, share_n, 0.4, color=ACCENT, label="share of users")
ax.bar(pos + 0.2, share_err, 0.4, color=WARM, label="share of squared error")
for i, (sn, se) in enumerate(zip(share_n, share_err, strict=True)):
    ax.annotate(
        f"{sn:.1%}",
        (i - 0.2, sn),
        ha="center",
        fontsize=7.5,
        xytext=(0, 3),
        textcoords="offset points",
    )
    ax.annotate(
        f"{se:.1%}",
        (i + 0.2, se),
        ha="center",
        fontsize=7.5,
        xytext=(0, 3),
        textcoords="offset points",
    )
ax.set_xticks(pos, list(BUCKETS), fontsize=8)
ax.set_ylabel("share")
ax.set_title(f"where the error lives, anchor {a}, held-out RMSLE {np.sqrt((res**2).mean()):.4f}")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="y")
save(fig, "error_structure.png")

if INTERPRET:
    perm = {k[len("perm_group_") :]: v for k, v in INTERPRET.items() if k.startswith("perm_group_")}
    fam = {
        k[len("family_share_") :]: v for k, v in INTERPRET.items() if k.startswith("family_share_")
    }
    keys = sorted(perm, key=lambda k: perm[k])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].barh(
        range(len(keys)),
        [perm[k] for k in keys],
        color=[WARM if k == "daily_tensor" else ACCENT for k in keys],
    )
    ax[0].set_yticks(range(len(keys)), keys, fontsize=8)
    ax[0].set_xlabel("mean absolute shift of the blend, z units")
    ax[0].set_title("group permutation on 20000 users")
    keys2 = sorted(fam, key=lambda k: fam[k])
    ax[1].barh(
        range(len(keys2)),
        [fam[k] for k in keys2],
        color=[WARM if k == "daily_tensor" else GREEN for k in keys2],
    )
    ax[1].set_yticks(range(len(keys2)), keys2, fontsize=8)
    ax[1].set_xlabel("share of mean absolute attribution")
    ax[1].set_title("attribution share, same families")
    for a_ in ax:
        a_.grid(alpha=0.25, axis="x")
    save(fig, "feature_families.png")

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

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].plot(b_grid, np.sqrt(W_TOTAL - b_grid**2), "-", lw=2, color=ACCENT, label="sqrt(W - B^2)")
ax[0].scatter(b_a, lb_a, s=26, color=GOLD, alpha=0.85, label="scored submissions")
ax[0].set_xlabel("B = Cov(z, u), recovered from the score")
ax[0].set_ylabel("public LB, RMSLE")
ax[0].set_title(f"W = {W_TOTAL}, m = {TARGET_MEAN}, pinned by two constant probes")
ax[0].legend(fontsize=8)
ax[0].set_xlim(1.618, 1.6325)
ax[0].set_ylim(1.6455, 1.6575)
gains = np.array([0.0, 0.0025, 0.005, 0.01])
b0 = 1.6270516
ax[1].plot(gains * 100, np.sqrt(W_TOTAL - (b0 * (1 + gains)) ** 2), "o-", lw=2, color=WARM)
for g in gains:
    s = float(np.sqrt(W_TOTAL - (b0 * (1 + g)) ** 2))
    ax[1].annotate(
        f"{s:.6f}", (g * 100, s), textcoords="offset points", xytext=(8, 4), fontsize=8, color=GOLD
    )
ax[1].set_xlabel("relative gain in B, percent")
ax[1].set_ylabel("public LB, RMSLE")
ax[1].set_title("exchange rate: 1 percent of B is worth 0.0165 RMSLE")
for a_ in ax:
    a_.grid(alpha=0.25)
save(fig, "exchange_rate.png")

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
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.barh(
        range(len(keys)),
        [seg[k] for k in keys],
        color=[GOLD if seg[k] < INTERPRET["eval_rmsle"] else WARM for k in keys],
    )
    ax.set_yticks(range(len(keys)), keys, fontsize=9)
    for i, k in enumerate(keys):
        ax.annotate(
            f"{seg[k]:.3f}   {shares[k]:.1%} of the cohort",
            (seg[k], i),
            va="center",
            fontsize=8,
            xytext=(6, 0),
            textcoords="offset points",
        )
    ax.axvline(INTERPRET["eval_rmsle"], color=FG, ls="--", lw=1.2)
    ax.annotate(
        f"pooled {INTERPRET['eval_rmsle']:.4f}",
        (INTERPRET["eval_rmsle"], len(keys) - 0.4),
        color=FG,
        fontsize=8,
        xytext=(-72, 0),
        textcoords="offset points",
    )
    ax.set_xlim(0, 2.45)
    ax.set_xlabel("held-out RMSLE over the 3 evaluation anchors")
    ax.set_title("error by behavioural segment, segments overlap")
    ax.grid(alpha=0.25, axis="x")
    save(fig, "segment_rmsle.png")
