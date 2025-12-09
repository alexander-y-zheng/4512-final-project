import matplotlib.pyplot as plt

# --------------------------------------
# Common geometry for the 2-period tree
# --------------------------------------

# x-coordinates correspond to time steps
#   0 -> t=0, 1 -> t=5 weeks, 2 -> t=10 weeks
# y-coordinates just for visual separation
node_positions = {
    "root": (0, 0),
    "u":    (1, 1),
    "d":    (1, -1),
    "uu":   (2, 2),
    "ud":   (2, 0),
    "dd":   (2, -2),
}

edges = [
    ("root", "u"),
    ("root", "d"),
    ("u", "uu"),
    ("u", "ud"),
    ("d", "ud"),
    ("d", "dd"),
]

# --------------------------------------
# Node values from your data
# --------------------------------------

stock_vals = {
    "root": 75.0,
    "u":    82.5,
    "d":    71.25,
    "uu":   90.75,
    "ud":   78.375,
    "dd":   67.6875,
}

bond_vals = {
    "root": 1.0,
    "u":    1.009661761,
    "d":    1.004819268,
    "uu":   1.009661761,
    "ud":   1.009661761,
    "dd":   1.009661761,
}

call_vals = {
    "root": 3.633818811,
    "u":    7.859711539,
    "d":    1.227517789,
    "uu":   15.75,
    "ud":   3.375,
    "dd":   0.0,
}

# --------------------------------------
# Helper function to plot a clean tree
# --------------------------------------


def plot_tree(values, title, filename, value_fmt="{:.2f}", prefix="", unit=""):
    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Draw all edges in a light, consistent color
    for src, dst in edges:
        x0, y0 = node_positions[src]
        x1, y1 = node_positions[dst]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="black",
            linewidth=1,
        )

    # Draw nodes and labels
    for node, (x, y) in node_positions.items():
        v = values[node]
        ax.scatter(x, y, s=40, color="black")

        # Choose a small label offset so it doesn't sit on the node or edge
        # Move labels slightly to the right and up/down depending on level
        if y >= 0:
            dy = 0.25
        else:
            dy = -0.25

        label_text = value_fmt.format(v)
        if prefix:
            label_text = f"{prefix} = {label_text}"
        if unit:
            label_text = f"{label_text} {unit}"

        ax.text(
            x + 0.08,
            y + dy,
            label_text,
            ha="left",
            va="center",
            fontsize=8,
        )

    # Time labels along the bottom
    time_labels = ["t = 0", "t = 5 weeks", "t = 10 weeks"]
    for i, lbl in enumerate(time_labels):
        ax.text(
            i,
            -2.8,
            lbl,
            ha="center",
            va="top",
            fontsize=9,
        )

    # Clean up axes
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-3.2, 3.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------
# Generate the three trees
# --------------------------------------

plot_tree(
    stock_vals,
    title="Two-Period Stock Price Tree",
    filename="stock_tree.pdf",
    value_fmt="{:.2f}",
    prefix="S",
    unit="USD",
)

plot_tree(
    bond_vals,
    title="Bond (Cash Account) Value Tree",
    filename="bond_tree.pdf",
    value_fmt="{:.6f}",
    prefix="B",
    unit="(discounted units)",
)

plot_tree(
    call_vals,
    title="Call Value Tree",
    filename="call_tree.pdf",
    value_fmt="{:.4f}",
    prefix="C",
    unit="USD",
)

