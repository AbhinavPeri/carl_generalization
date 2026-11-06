# carl_generalization

One marimo notebook, `carl_clones.py`: a CARL-style contrastive representation `φ(s, g)` / `ψ(a_τ)` trained on **two
identical gridworlds whose states have different identities**, with random, optimal, and ε-greedy data policies.
Everything (world, data, networks, training, plots) is inside the notebook; the code is under its *Implementation*
section.

## Setup

Requires [Nix](https://nixos.org/download) with flakes enabled and, optionally, an NVIDIA GPU.

```sh
nix run . -- -c 'uv sync --extra cuda && uv run marimo edit carl_clones.py'
```

`nix run .` opens an FHS shell (uv + Python 3.12, so the pip wheels for JAX/CUDA work); `-c '…'` runs a command in it.
Use `uv sync` instead of `uv sync --extra cuda` on a machine without an NVIDIA GPU. For a read-only app use
`uv run marimo run carl_clones.py`, and for a static HTML copy `uv run marimo export html carl_clones.py -o out.html`.
`nix develop` puts the same shell on your PATH as `carl-generalization`.

Reproducibility: `flake.lock` pins nixpkgs (uv, Python, system libraries) and `uv.lock` pins every Python package.

Without Nix: install [uv](https://docs.astral.sh/uv/) and Python 3.12, then run the `uv` commands above. On NixOS the
flake adds `/run/opengl-driver/lib` to the library path so JAX finds the NVIDIA driver; on other distributions the
driver is found as usual.
