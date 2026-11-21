"""Q1: schema graph of the 8 tables (nodes) and every foreign key (edge FK -> referenced PK).

    python src/schema_graph.py

Writes results/figs/schema_graph.png (matplotlib) and results/schema.dot (Graphviz source;
render with `dot -Tpng results/schema.dot -o schema.png` if Graphviz is installed).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"

# table: (primary key, [foreign keys], row-time column)
TABLES = {
    "Users": ("user_id", [], "first_seen_at"),
    "Repositories": ("repo_id", [], "first_seen_at"),
    "PullRequests": ("pr_id", ["repo_id", "author_id", "merged_by_id"], "first_seen_at"),
    "Issues": ("issue_id", ["repo_id", "author_id", "pull_request_id", "assignee_id"], "first_seen_at"),
    "IssueComments": ("comment_id", ["issue_id", "repo_id", "author_id"], "created_at"),
    "PRReviews": ("review_id", ["pr_id", "repo_id", "reviewer_id"], "first_seen_at"),
    "PRReviewComments": ("comment_id", ["pr_id", "repo_id", "author_id", "pull_request_review_id",
                                        "in_reply_to_id"], "created_at"),
    "Pushes": ("push_id", ["user_id", "repo_id"], "created_at"),
}
# (table, fk column) -> referenced table
FKS = {
    ("PullRequests", "repo_id"): "Repositories", ("PullRequests", "author_id"): "Users",
    ("PullRequests", "merged_by_id"): "Users",
    ("Issues", "repo_id"): "Repositories", ("Issues", "author_id"): "Users",
    ("Issues", "pull_request_id"): "PullRequests", ("Issues", "assignee_id"): "Users",
    ("IssueComments", "issue_id"): "Issues", ("IssueComments", "repo_id"): "Repositories",
    ("IssueComments", "author_id"): "Users",
    ("PRReviews", "pr_id"): "PullRequests", ("PRReviews", "repo_id"): "Repositories",
    ("PRReviews", "reviewer_id"): "Users",
    ("PRReviewComments", "pr_id"): "PullRequests", ("PRReviewComments", "repo_id"): "Repositories",
    ("PRReviewComments", "author_id"): "Users",
    ("PRReviewComments", "pull_request_review_id"): "PRReviews",
    ("PRReviewComments", "in_reply_to_id"): "PRReviewComments",
    ("Pushes", "user_id"): "Users", ("Pushes", "repo_id"): "Repositories",
}

ROW = ["Pushes", "IssueComments", "Issues", "PullRequests", "PRReviews", "PRReviewComments"]
W, LINE, GAP = 3.0, 0.36, 0.7
Y_ROW, Y_USERS, Y_REPOS = 0.0, 4.2, -4.6


def box_h(t):
    return LINE * (2 + len(TABLES[t][1])) + 0.3


def dot():
    lines = ["digraph schema {", "  rankdir=LR; node [shape=plaintext, fontname=Helvetica];"]
    for t, (pk, fks, rt) in TABLES.items():
        rows = f'<tr><td bgcolor="#dbe7fb"><b>{t}</b></td></tr><tr><td align="left"><b>PK</b> {pk}</td></tr>'
        rows += "".join(f'<tr><td align="left" port="{f}">FK {f}</td></tr>' for f in fks)
        lines.append(f'  {t} [label=<<table border="1" cellborder="0">{rows}</table>>];')
    for (t, f), ref in FKS.items():
        lines.append(f'  {t}:{f} -> {ref} [label="{f}", fontsize=9];')
    lines.append("}")
    (RES / "schema.dot").write_text("\n".join(lines))


def table_box(ax, t, x, y, w, h):
    pk, fks, rt = TABLES[t]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04",
                                fc="white", ec="#2a3f5f", lw=1.5, zorder=2))
    ax.add_patch(FancyBboxPatch((x - w / 2, y + h / 2 - LINE - 0.08), w, LINE + 0.08,
                                boxstyle="round,pad=0.04", fc="#dbe7fb", ec="#2a3f5f", lw=1.5, zorder=3))
    ax.text(x, y + h / 2 - LINE / 2 - 0.04, t, ha="center", va="center", weight="bold", fontsize=11, zorder=4)
    yy = y + h / 2 - LINE * 1.5 - 0.1
    ax.text(x - w / 2 + 0.12, yy, f"PK  {pk}", va="center", fontsize=9.5, color="#b03030", weight="bold", zorder=4)
    for f in fks:
        yy -= LINE
        ax.text(x - w / 2 + 0.12, yy, f"FK  {f}", va="center", fontsize=9.5, color="#1f5fa8", zorder=4)
    ax.text(x + w / 2 - 0.08, y - h / 2 + 0.08, f"row time: {rt}", ha="right", va="bottom",
            fontsize=7.5, color="#666", style="italic", zorder=4)


def arrow(ax, p0, p1, col, rad=0.0, lw=1.3):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                 mutation_scale=13, color=col, lw=lw, zorder=1, shrinkA=0, shrinkB=0))


def draw():
    n = len(ROW)
    xs = {t: i * (W + GAP) for i, t in enumerate(ROW)}
    hmax = max(box_h(t) for t in ROW)
    left, right = -W / 2, xs[ROW[-1]] + W / 2
    fig, ax = plt.subplots(figsize=(17, 10))
    wide = right - left
    table_box(ax, "Users", (left + right) / 2, Y_USERS, wide, 0.95)
    table_box(ax, "Repositories", (left + right) / 2, Y_REPOS, wide, 0.95)
    for t in ROW:
        table_box(ax, t, xs[t], Y_ROW + (hmax - box_h(t)) / 2, W, box_h(t))
    top = {t: Y_ROW + (hmax - box_h(t)) / 2 + box_h(t) / 2 for t in ROW}
    bot = {t: Y_ROW + (hmax - box_h(t)) / 2 - box_h(t) / 2 for t in ROW}
    red, green, orange, purple = "#c43c5c", "#3a9e5f", "#e07b24", "#7d5ab5"
    for t in ROW:
        to_u = [f for (tt, f), r in FKS.items() if tt == t and r == "Users"]
        to_r = [f for (tt, f), r in FKS.items() if tt == t and r == "Repositories"]
        for k, f in enumerate(to_u):
            x = xs[t] - 0.9 + k * 1.0
            arrow(ax, (x, top[t] + 0.12), (x, Y_USERS - 0.5), red)
            ax.text(x + 0.08, (top[t] + Y_USERS - 0.4) / 2 + 0.25 * (k % 2), f, fontsize=8.5, color=red,
                    rotation=90, va="center", ha="left")
        for f in to_r:
            x = xs[t]
            arrow(ax, (x, bot[t] - 0.08), (x, Y_REPOS + 0.5), green)
            ax.text(x + 0.08, (bot[t] + Y_REPOS) / 2, f, fontsize=8.5, color=green, rotation=90,
                    va="center", ha="left")
    # edges between neighbouring tables in the row (horizontal, at the height of the FK line)
    for (t, f), ref in FKS.items():
        if t in ("Users", "Repositories") or ref in ("Users", "Repositories") or t == ref:
            continue
        i, j = ROW.index(t), ROW.index(ref)
        ytxt = top[t] - LINE * (2.5 + TABLES[t][1].index(f)) + 0.08 - 0.12
        if abs(i - j) == 1:
            x0 = xs[t] + (W / 2 if j > i else -W / 2) + 0.05
            x1 = xs[ref] + (-W / 2 if j > i else W / 2) - 0.05
            y1 = min(ytxt, bot[ref] + 0.4) if ytxt < bot[ref] + 0.4 else ytxt
            arrow(ax, (x0, ytxt), (x1, max(y1, bot[ref] + 0.3)), orange)
            label = f.replace("pull_request_review_id", "pull_request_\nreview_id")
            ax.text((x0 + x1) / 2, ytxt + 0.12, label, fontsize=7.5, color=orange, ha="center", va="bottom",
                    bbox=dict(fc="white", ec="none", pad=0.3), zorder=6)
        else:   # skips one table: curve below the row
            x0, x1 = xs[t] - 0.6, xs[ref] + 0.6
            arrow(ax, (x0, bot[t] - 0.08), (x1, bot[ref] - 0.08), orange, rad=-0.35)
            ax.text(x0 - 0.3 * (x0 - x1), min(bot[t], bot[ref]) - 1.0, f, fontsize=8.5, color=orange, ha="center",
                    bbox=dict(fc="white", ec="none", pad=0.5))
    # self reference: reply to another review comment
    t = "PRReviewComments"
    yr = top[t] - LINE * (2.5 + TABLES[t][1].index("in_reply_to_id")) - 0.04
    xr = xs[t] + W / 2 + 0.05
    arrow(ax, (xr, yr), (xr, top[t] - LINE * 1.5), purple, rad=0.9)
    ax.text(xr + 0.45, yr + 0.5, "in_reply_to_id\n(self)", fontsize=8, color=purple, va="center")
    ax.set_xlim(left - 0.4, right + 1.6)
    ax.set_ylim(Y_REPOS - 0.8, Y_USERS + 0.8)
    ax.axis("off")
    ax.set_title("Schema graph: 8 tables (nodes), 20 foreign keys (edges, FK -> referenced PK)\n"
                 "red: -> Users.user_id   green: -> Repositories.repo_id   orange: -> other tables   "
                 "purple: self reference", fontsize=12)
    fig.tight_layout()
    fig.savefig(RES / "figs" / "schema_graph.png", dpi=150)


if __name__ == "__main__":
    (RES / "figs").mkdir(parents=True, exist_ok=True)
    dot()
    draw()
    print(len(FKS), "foreign keys")
