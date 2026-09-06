# Installing Cosmos

Cosmos is built on top of [NGSolve/Netgen](https://ngsolve.org/), a finite-element
library distributed as its own compiled package outside of PyPI's normal wheel
ecosystem on some platforms. Getting a fully reproducible environment means
pinning three separate things: the Python version, NGSolve/Netgen itself, and
the pure-Python dependencies (numpy, scipy, ...). This file covers all three.

## 1. Python version

This project targets **Python 3.10+** (`pyproject.toml`); the environment it
was developed and tested against is pinned exactly via `Pipfile`
(`python_version = "3.14"`). Using the same minor version as `Pipfile` is the
safest way to reproduce results bit-for-bit.

## 2. Install NGSolve/Netgen

Pick the route for your platform — **do this before** setting up the
Python environment in step 3, since NGSolve is not managed by `Pipfile`/`pip`.

### macOS

Download and install the official app bundle from
[ngsolve.org/downloads](https://ngsolve.org/downloads) (`Netgen.app`, drag into
`/Applications`). It ships its own Python framework's site-packages; expose it
to your own Python by setting, in your shell profile:

```bash
export NETGENDIR=/Applications/Netgen.app/Contents/MacOS
export PYTHONPATH="/Applications/Netgen.app/Contents/Resources/lib/python3.14/site-packages:$PYTHONPATH"
```

(Adjust the `python3.14` path component to match the Python version bundled in
the `Netgen.app` you installed.) This is the exact mechanism used by the
reference development environment for this repository — a venv created with
`include-system-site-packages = false` still sees `ngsolve`/`netgen` purely
because of this `PYTHONPATH` entry, not because they were `pip install`ed.

### Linux / other platforms

Prebuilt wheels are published on PyPI for common platforms:

```bash
pip install ngsolve
```

If that doesn't have a wheel for your platform, see the
[NGSolve installation docs](https://docu.ngsolve.org/latest/install/install.html)
for building from source or using conda-forge.

### Verify the install

```bash
python3 -c "import ngsolve, netgen; print(ngsolve.__file__)"
```

should succeed and print a path — do this **before** moving on, since a
broken NGSolve install is the most common source of confusing failures later.

## 3. Install Cosmos and its Python dependencies

This project uses [`pipenv`](https://pipenv.pypa.io/) with a committed
`Pipfile.lock` for exact, reproducible dependency versions.

```bash
pip install pipenv          # if you don't already have it
git clone <this-repo-url>
cd cosmos-fix
pipenv install --dev        # reads Pipfile.lock; installs cosmos itself editably too
pipenv shell                # activate the environment
```

`pipenv install --dev` installs exactly the versions recorded in
`Pipfile.lock` (reproducible across machines) rather than re-resolving
`Pipfile`'s open-ended `"*"` version specifiers — use `pipenv install
--dev --skip-lock` only if you deliberately want the latest compatible
versions instead of the locked ones.

Prefer plain `pip` / a virtualenv you manage yourself? That works too, since
`cosmos` is a normal `pyproject.toml`/hatchling package:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[tests,docs]"
```

— just note this route does **not** give you the pinned versions from
`Pipfile.lock`, only pyproject.toml's looser `>=`/unconstrained requirements,
and you're still responsible for step 2 (NGSolve) separately either way.

## 4. Verify the full install

```bash
python3 -c "import cosmos; print(cosmos.__version__)"
pytest tests/unit tests/pde_examples -q
```

Both should succeed in well under a minute. If they do, you're set up
identically to the reference environment; `tutorials/` is a good next stop.
