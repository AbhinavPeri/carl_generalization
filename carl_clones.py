import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import marimo as mo
    import numpy as np
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    import optax
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, jax, jnp, make_subplots, mo, nn, np, optax


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # CARL representation in two identical gridworlds with different state identities

    Two copies of the same empty 6×6 gridworld (**environment A** and **environment B**): same layout, same dynamics, but
    different states. A state is observed as a one-hot **identity** over the 72 cells of both copies, so the networks cannot
    tell that "row 2, column 3" is the same place in A and in B. Every episode starts in the **top-left** cell of its copy,
    lasts `k = 36` steps, and episodes alternate between the copies.

    **One** `φ(s, g)` and **one** `ψ(a_τ)` (unit vectors) are trained on all episodes with symmetric InfoNCE on
    `cos / τ`; positives are the `(s, g, a_τ)` of one episode, negatives are the other episodes of the batch
    (`s` = start, `g` = final state, `a_τ` = the episode's actions, one-hot and zero-padded). No RL. The only thing the
    copies share is the action sequence.

    **Data policies.** Every policy is **ε-greedy toward a goal**: a goal cell is sampled uniformly (it is always reachable
    within `k` steps); at every step, with probability `ε` the action is uniformly random, otherwise it moves toward the goal
    (first along its preferred axis, then the other) and stands still (zero padding) once it is there.
    `ε = 0` is the deterministic **optimal** policy, `ε = 1` is the **random** policy.

    | data set | environment A | environment B |
    |---|---|---|
    | random | random (ε = 1) | random (ε = 1) |
    | same optimal | optimal, vertical first | optimal, vertical first |
    | different optimal | optimal, vertical first | optimal, horizontal first |
    | **random A, ε-greedy B** | random (ε = 1) | **ε-greedy** with the ε set below, vertical first |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    DEFAULTS = dict(dim=8, temperature=0.1, iterations=3000, learning_rate=1e-3, batch_size=1024, hidden_dim=256,
                    n_hidden_layers=2, epsilon=0.0, seed=0)
    form = mo.md("""
    ## Setup
    {dim} {temperature} {iterations}

    {learning_rate} {batch_size} {hidden_dim} {n_hidden_layers}

    {epsilon} {seed}
    """).batch(
        dim=mo.ui.slider(2, 64, value=DEFAULTS["dim"], label="embedding dim D", show_value=True),
        temperature=mo.ui.dropdown({"0.03": 0.03, "0.05": 0.05, "0.1": 0.1, "0.2": 0.2, "0.5": 0.5, "1.0": 1.0},
                                   value="0.1", label="temperature τ"),
        iterations=mo.ui.slider(500, 100_000, step=500, value=DEFAULTS["iterations"], label="updates", show_value=True),
        learning_rate=mo.ui.dropdown({"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3, "3e-3": 3e-3}, value="1e-3",
                                     label="learning rate"),
        batch_size=mo.ui.dropdown({str(n): n for n in (128, 256, 512, 1024, 2048, 4096)},
                                  value=str(DEFAULTS["batch_size"]), label="episodes per update (= contrastive batch)"),
        hidden_dim=mo.ui.dropdown({str(n): n for n in (16, 32, 64, 128, 256, 512, 1024)},
                                  value=str(DEFAULTS["hidden_dim"]), label="network width"),
        n_hidden_layers=mo.ui.slider(0, 6, value=DEFAULTS["n_hidden_layers"], label="network depth", show_value=True),
        epsilon=mo.ui.slider(0.0, 1.0, step=0.05, value=DEFAULTS["epsilon"], show_value=True,
                             label="ε of the ε-greedy policy in B  (random A, ε-greedy B)"),
        seed=mo.ui.number(0, 1000, value=DEFAULTS["seed"], label="seed"),
    ).form(submit_button_label="train")
    form
    return DEFAULTS, form


@app.cell(hide_code=True)
def _(DEFAULTS, form, jnp):
    cfg = {**DEFAULTS, **(form.value or {})}

    # data set name -> (epsilon in A and B, vertical-first in A and B)
    DATASETS = {
        "random": (jnp.array([1.0, 1.0]), jnp.array([True, True])),
        "same optimal": (jnp.array([0.0, 0.0]), jnp.array([True, True])),
        "different optimal": (jnp.array([0.0, 0.0]), jnp.array([True, False])),
        f"random A, ε-greedy B (ε={cfg['epsilon']:g})": (jnp.array([1.0, cfg["epsilon"]]), jnp.array([True, True])),
    }
    COLORS = dict(zip(DATASETS, ["#6b7280", "#2563eb", "#ea6a2a", "#16a34a"]))
    return COLORS, DATASETS, cfg


@app.cell(hide_code=True)
def _(jax, jnp):
    # ---- the world: two copies (0 = A, 1 = B) of an empty H x W gridworld, always starting top-left
    H = W = 6
    N_CELLS = H * W
    K = N_CELLS                                  # steps per episode: enough to reach any cell
    UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, -1
    MOVES = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]])        # (row, column) change of up, down, left, right

    def step(position, action):
        """position (n, 2), action (n,); STAY (-1) does not move; the border blocks."""
        moved = jnp.clip(position + MOVES[jnp.maximum(action, 0)], 0, jnp.array([H - 1, W - 1]))
        return jnp.where((action < 0)[:, None], position, moved)

    def identity(copy, position):
        """Index of a state among the 2 * H * W state identities."""
        return copy * N_CELLS + position[..., 0] * W + position[..., 1]

    def episodes(key, n, epsilon, vertical_first):
        """n ε-greedy episodes of K steps; episode i is in copy i % 2. epsilon and vertical_first have one entry per copy.
        Returns copy (n,), positions (n, K + 1, 2), actions (n, K) with STAY = -1 for padding."""
        copy = jnp.arange(n) % 2
        goal_key, scan_key = jax.random.split(key)
        goal_index = jax.random.randint(goal_key, (n,), 1, N_CELLS)                         # any cell but the start (0)
        goal = jnp.stack([goal_index // W, goal_index % W], axis=-1)

        def act(position, step_key):
            delta = goal - position
            vertical = jnp.where(delta[:, 0] > 0, DOWN, UP)
            horizontal = jnp.where(delta[:, 1] > 0, RIGHT, LEFT)
            vertical_left, horizontal_left = delta[:, 0] != 0, delta[:, 1] != 0
            vertical_first_action = jnp.where(vertical_left, vertical, jnp.where(horizontal_left, horizontal, STAY))
            horizontal_first_action = jnp.where(horizontal_left, horizontal, jnp.where(vertical_left, vertical, STAY))
            greedy = jnp.where(vertical_first[copy], vertical_first_action, horizontal_first_action)
            explore_key, random_key = jax.random.split(step_key)
            explore = jax.random.uniform(explore_key, (n,)) < epsilon[copy]
            action = jnp.where(explore, jax.random.randint(random_key, (n,), 0, 4), greedy)
            return step(position, action), (position, action)

        final, (positions, actions) = jax.lax.scan(act, jnp.zeros((n, 2), jnp.int32), jax.random.split(scan_key, K))
        positions = jnp.concatenate([positions, final[None]], axis=0).swapaxes(0, 1)
        return copy, positions, actions.T

    return H, K, N_CELLS, W, episodes, identity


@app.cell(hide_code=True)
def _(N_CELLS, K, cfg, identity, jax, jnp, nn, np, optax):
    # ---- the representation: phi(s, g) and psi(a_tau), unit vectors, symmetric InfoNCE
    class MLP(nn.Module):
        out: int

        @nn.compact
        def __call__(self, x):
            for _ in range(cfg["n_hidden_layers"]):
                x = nn.relu(nn.Dense(cfg["hidden_dim"])(x))
            return nn.Dense(self.out)(x)

    def unit(vectors):
        return vectors / jnp.sqrt((vectors ** 2).sum(-1, keepdims=True) + 1e-12)

    net = MLP(cfg["dim"])

    def phi(params, state_id, goal_id):
        inputs = jnp.concatenate([jax.nn.one_hot(state_id, 2 * N_CELLS), jax.nn.one_hot(goal_id, 2 * N_CELLS)], axis=-1)
        return unit(net.apply(params["phi"], inputs))

    def psi(params, actions):
        sequence = jax.nn.one_hot(actions, 4).reshape(actions.shape[0], -1)            # padding (-1) -> all zeros
        return unit(net.apply(params["psi"], sequence))

    def init_params(key):
        phi_key, psi_key = jax.random.split(key)
        return {"phi": net.init(phi_key, jnp.zeros((1, 4 * N_CELLS))), "psi": net.init(psi_key, jnp.zeros((1, 4 * K)))}

    def loss_fn(params, state_id, goal_id, actions):
        cosines = phi(params, state_id, goal_id) @ psi(params, actions).T               # (B, B), diagonal = positives
        logits = cosines / cfg["temperature"]
        labels = jnp.arange(logits.shape[0])
        loss = 0.5 * (optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
                      + optax.softmax_cross_entropy_with_integer_labels(logits.T, labels).mean())
        angles = jnp.degrees(jnp.arccos(jnp.clip(jax.lax.stop_gradient(cosines), -1.0, 1.0)))
        positive = jnp.eye(logits.shape[0], dtype=bool)
        pair = state_id * 2 * N_CELLS + goal_id                                          # which (s, g) pair a row is
        same_pair = pair[:, None] == pair[None]
        return loss, {
            "retrieval_accuracy": same_pair[logits.argmax(0), labels].mean(),            # best (s, g) for each sequence
            "positive_pair_angle": (angles * positive).sum() / positive.sum(),
            "negative_pair_angle": (angles * ~positive).sum() / (~positive).sum()}

    def train(episodes, dataset, key, n_updates, progress):
        """Train phi and psi on the data of `dataset` for n_updates; returns (params, history), one point per update."""
        epsilon, vertical_first = dataset
        optimizer = optax.adam(cfg["learning_rate"])

        def update(carry, update_key):
            params, opt_state = carry
            copy, positions, actions = episodes(update_key, cfg["batch_size"], epsilon, vertical_first)
            state_id, goal_id = identity(copy, positions[:, 0]), identity(copy, positions[:, -1])
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state_id, goal_id, actions)
            updates, opt_state = optimizer.update(grads, opt_state)
            return (optax.apply_updates(params, updates), opt_state), {"loss": loss, **metrics}

        chunk = 50                                                                       # updates per jit call
        run_chunk = jax.jit(lambda carry, chunk_key: jax.lax.scan(update, carry, jax.random.split(chunk_key, chunk)))
        params = init_params(key)
        carry, history = (params, optimizer.init(params)), []
        for i in range(n_updates // chunk):
            carry, metrics = run_chunk(carry, jax.random.fold_in(key, i))
            history.append({name: np.asarray(values) for name, values in metrics.items()})
            progress()
        return carry[0], {name: np.concatenate([h[name] for h in history]) for name in history[0]}

    return phi, train


@app.cell(hide_code=True)
def _(mo):
    import pathlib, textwrap

    def _cell_body(marker):
        """Source of the notebook cell that contains `marker`, without its signature and return line."""
        chunk = next(c for c in pathlib.Path(__file__).read_text().split("@app.cell")
                     if marker in c and "pathlib" not in c)
        return textwrap.dedent(chunk[chunk.index("):\n") + 3:chunk.rindex("\n    return")])

    def _code(marker):
        return mo.md(f"```python\n{_cell_body(marker)}\n```")

    mo.vstack([
        mo.md("## Implementation"),
        mo.accordion({
            "Data sets (code)": _code("# data set name ->"),
            "World and ε-greedy data policies (code)": _code("# ---- the world"),
            "Representation, loss and training loop (code)": _code("# ---- the representation"),
        })])
    return


@app.cell(hide_code=True)
def _(DATASETS, H, W, episodes, go, jax, make_subplots, mo):
    _figure = make_subplots(rows=2, cols=len(DATASETS), horizontal_spacing=0.07, vertical_spacing=0.16,
                            subplot_titles=[f"{_name}<br>environment {_copy}" for _copy in "AB" for _name in DATASETS])
    for _column, (_name, (_epsilon, _vertical_first)) in enumerate(DATASETS.items(), start=1):
        _copy, _positions, _ = episodes(jax.random.PRNGKey(0), 4096, _epsilon, _vertical_first)
        for _row in (1, 2):
            _in_copy = _positions[_copy == _row - 1]                                           # (m, K + 1, 2)
            _cell = _in_copy[..., 0] * W + _in_copy[..., 1]
            _visited = jax.nn.one_hot(_cell, H * W).max(axis=1).mean(0)                        # fraction of episodes per cell
            _figure.add_trace(go.Heatmap(
                z=_visited.reshape(H, W), zmin=0, zmax=1, colorscale="YlOrRd", texttemplate="%{z:.2f}",
                showscale=(_row, _column) == (1, len(DATASETS)), colorbar=dict(title="P(visited)"),
                hovertemplate="row %{y}, column %{x}: %{z:.2f}<extra></extra>"), _row, _column)
            _axis = len(DATASETS) * (_row - 1) + _column
            _figure.update_yaxes(autorange="reversed", scaleanchor=f"x{'' if _axis == 1 else _axis}", row=_row, col=_column)
    _figure.update_annotations(font_size=13)
    _figure.update_layout(height=800, width=1150, margin=dict(t=70, b=10))
    mo.vstack([mo.md("## The data: which cells does an episode visit?  (fraction of episodes visiting each cell)"), _figure])
    return


@app.cell(hide_code=True)
def _(DATASETS, cfg, episodes, jax, mo, train):
    results = {}
    with mo.status.progress_bar(total=len(DATASETS) * (cfg["iterations"] // 50), title="training") as _bar:
        for _name, _dataset in DATASETS.items():
            results[_name] = train(episodes, _dataset, jax.random.PRNGKey(cfg["seed"]), cfg["iterations"],
                                   progress=_bar.update)
    return (results,)


@app.cell(hide_code=True)
def _(COLORS, cfg, go, make_subplots, mo, np, results):
    mo.md("## Training: loss, retrieval accuracy, and angles of positive vs negative pairs")

    def _smooth(values, weight=0.95):
        smoothed = np.array(values, dtype=float)
        for i in range(1, len(smoothed)):
            smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * smoothed[i]
        return smoothed

    curves_figure = make_subplots(rows=2, cols=2, horizontal_spacing=0.1, vertical_spacing=0.2,
                                  subplot_titles=("InfoNCE loss", "retrieval accuracy ψ(a_τ) → φ(s, g)",
                                                  "positive-pair angle (degrees)", "negative-pair angle (degrees)"))
    for _name, (_, _history) in results.items():
        for _row, _column, _key in [(1, 1, "loss"), (1, 2, "retrieval_accuracy"),
                                    (2, 1, "positive_pair_angle"), (2, 2, "negative_pair_angle")]:
            _y = _smooth(_history[_key])
            _step = max(1, len(_y) // 1500)
            curves_figure.add_trace(go.Scatter(
                x=np.arange(1, len(_y) + 1)[::_step], y=_y[::_step], mode="lines", name=_name, legendgroup=_name,
                line=dict(color=COLORS[_name]), showlegend=(_row, _column) == (1, 1)), _row, _column)
    curves_figure.add_hline(y=1 / cfg["batch_size"], line=dict(color="#aaa", dash="dot"), row=1, col=2)    # chance
    curves_figure.update_xaxes(title_text="update")
    curves_figure.update_yaxes(range=[0, 180], row=2)
    curves_figure.update_annotations(font_size=13)
    curves_figure.update_layout(template="plotly_white", height=800, width=1100, margin=dict(l=60, r=20, t=60, b=50))
    curves_figure
    return


@app.cell(hide_code=True)
def _(H, W, identity, jnp, np, phi):
    CELLS = np.array([(r, c) for r in range(H) for c in range(W)])                              # row-major

    def embed(params, copy, state_cells, goal_cells):
        """φ(s, g) in environment `copy` (0 = A, 1 = B) for arrays of (row, column) cells: (n, D)."""
        return phi(params, identity(copy, jnp.asarray(state_cells)), identity(copy, jnp.asarray(goal_cells)))

    return CELLS, embed


@app.cell(hide_code=True)
def _(CELLS, embed, mo, np, results):
    # does the same (start, goal) in A and in B get the same representation?  (start = top-left, the only trained start)
    _start = np.zeros_like(CELLS)
    _rows = []
    for _name, (_params, _) in results.items():
        _a, _b = embed(_params, 0, _start, CELLS), embed(_params, 1, _start, CELLS)
        _angle = np.degrees(np.arccos(np.clip(np.asarray(_a @ _b.T), -1, 1)))                  # (goal in A, goal in B)
        _other = (_angle.sum() - np.trace(_angle)) / (_angle.size - len(_angle))
        _rows.append(f"| {_name} | {np.diag(_angle).mean():.1f}° | {_other:.1f}° |")
    mo.md("## Are the two environments aligned?\n"
          "Angle between `φ_A(s₀, g)` and `φ_B(s₀, g′)` over the goals `g`, `g′` (`s₀` = top-left start). "
          "Aligned copies have a small angle for the **same** goal and ≈ 90° for different goals.\n\n"
          "| data | same goal in A and B | different goals |\n|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(H, W, mo):
    mo.md("""
    ## Choose an (s, g) in environment A and look at environment B
    Each row is one data set. The **left** panel shades every goal cell `g'` of environment A by the **angle** (degrees,
    bright = close) between `φ_A(s, g)` and `φ_A(s, g')`; the **right** panel does the same for environment B: the angle
    between `φ_A(s, g)` and `φ_B(s_B, g')`, with `s_B` the start in B (by default the same cell as `s`).
    A bright cell in B that is not the matching goal is a pair that the representation treats like the chosen pair of A.
    Only `s` = top-left was ever seen in training.
    """)
    ref_state_row = mo.ui.slider(0, H - 1, value=0, label="s (in A): row", show_value=True)
    ref_state_column = mo.ui.slider(0, W - 1, value=0, label="s (in A): column", show_value=True)
    ref_goal_row = mo.ui.slider(0, H - 1, value=1, label="g (in A): row", show_value=True)
    ref_goal_column = mo.ui.slider(0, W - 1, value=2, label="g (in A): column", show_value=True)
    same_state = mo.ui.checkbox(value=True, label="s in B = same cell as s in A")
    other_state_row = mo.ui.slider(0, H - 1, value=0, label="s in B: row", show_value=True)
    other_state_column = mo.ui.slider(0, W - 1, value=0, label="s in B: column", show_value=True)
    mo.vstack([mo.hstack([ref_state_row, ref_state_column], justify="start"),
               mo.hstack([ref_goal_row, ref_goal_column], justify="start"),
               mo.hstack([same_state, other_state_row, other_state_column], justify="start")])
    return (other_state_column, other_state_row, ref_goal_column, ref_goal_row, ref_state_column, ref_state_row,
            same_state)


@app.cell(hide_code=True)
def _(CELLS, H, W, embed, go, make_subplots, mo, np, other_state_column, other_state_row, ref_goal_column,
      ref_goal_row, ref_state_column, ref_state_row, results, same_state):
    _state_a = (ref_state_row.value, ref_state_column.value)
    _goal_a = (ref_goal_row.value, ref_goal_column.value)
    _state_b = _state_a if same_state.value else (other_state_row.value, other_state_column.value)

    alias_figure = make_subplots(rows=len(results), cols=2, horizontal_spacing=0.08, vertical_spacing=0.07,
                                 subplot_titles=[f"{_name}<br>environment {_copy}" for _name in results for _copy in "AB"])
    for _row, (_name, (_params, _)) in enumerate(results.items(), start=1):
        _reference = embed(_params, 0, [_state_a], [_goal_a])[0]                               # φ_A(s, g)
        for _column, (_copy, _start) in enumerate(((0, _state_a), (1, _state_b)), start=1):
            _others = embed(_params, _copy, np.tile(_start, (len(CELLS), 1)), CELLS)
            _angle = np.degrees(np.arccos(np.clip(np.asarray(_others @ _reference), -1, 1))).reshape(H, W)
            alias_figure.add_trace(go.Heatmap(
                z=_angle, zmin=0, zmax=180, colorscale="Viridis_r", showscale=(_row, _column) == (1, 2),
                colorbar=dict(title="degrees"), texttemplate="%{z:.0f}",
                hovertemplate="g' = (row %{y}, column %{x}): %{z:.1f}°<extra></extra>"), _row, _column)
            alias_figure.add_trace(go.Scatter(
                x=[_start[1]], y=[_start[0]], mode="markers", showlegend=False, name="s",
                marker=dict(symbol="star", size=16, color="white", line=dict(color="#111", width=2))), _row, _column)
            if _copy == 0:
                alias_figure.add_trace(go.Scatter(
                    x=[_goal_a[1]], y=[_goal_a[0]], mode="markers", showlegend=False, name="g",
                    marker=dict(symbol="circle-open", size=22, color="#e11d48", line=dict(width=3))), _row, _column)
            _axis = 2 * (_row - 1) + _column
            alias_figure.update_yaxes(autorange="reversed", scaleanchor=f"x{'' if _axis == 1 else _axis}", row=_row, col=_column)
    alias_figure.update_annotations(font_size=13)
    alias_figure.update_layout(height=420 * len(results), width=1000, margin=dict(t=60, b=10))
    mo.vstack([mo.md(f"Reference: **φ_A(s = {_state_a}, g = {_goal_a})**; right panels use **s in B = {_state_b}**. "
                     "Star = s, red circle = g."),
               alias_figure])
    return


if __name__ == "__main__":
    app.run()
