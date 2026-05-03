"""Modal app: run Lean 4 / Lake builds for the EchoRH formalization.

The local machine never runs `lean` or `lake`.  We edit Lean files
locally and execute every build, check, and toolchain operation
inside a Modal container.

Why no `modal.Volume` for `.lake`
---------------------------------
We tried storing `.lake/packages/mathlib` on a `modal.Volume`.  Even
no-op `lake env true` invocations took ~3.5 minutes because Lake
stats thousands of `.olean`/`.trace` files and Volume I/O is too
slow for that pattern.  Instead we **bake Mathlib into the image**
at build time:

1. We copy `lakefile.toml`, `lean-toolchain`, and `lake-manifest.json`
   into a *warmup* directory at `/opt/lean-warmup/`.
2. We run `lake update && lake exe cache get` there during image
   build.  This downloads ~6.4 GB of prebuilt Mathlib `.olean` files
   into the image's overlay filesystem.
3. At function runtime, the project at `/root/echo-rh-lean/` (whose
   own source files are mounted via `add_local_dir`) symlinks
   `.lake/packages -> /opt/lean-warmup/.lake/packages`.  EchoRH's
   own ~10 MB of build output then lands on the container's local
   disk and is rebuilt from scratch each invocation -- cheap because
   it only depends on already-compiled Mathlib oleans.

Image cache invalidation
------------------------
The image rebuilds (and re-pulls Mathlib, ~5-10 min) **only** when
one of `lakefile.toml`, `lean-toolchain`, or `lake-manifest.json`
changes.  Day-to-day edits to `EchoRH/*.lean` reuse the cached image.

Usage
-----
First-time setup is implicit -- the first `modal run` triggers the
image build.

Build the whole library::

    modal run echo-rh-lean/modal_lean.py::build

Build a single module::

    modal run echo-rh-lean/modal_lean.py::build --target EchoRH.Basic

Sanity-check the toolchain::

    modal run echo-rh-lean/modal_lean.py::check

Inspect axiom dependencies of a declaration::

    modal run echo-rh-lean/modal_lean.py::axioms \\
        --target EchoRH.gradeFourEigenvalue_ne_one
"""

from __future__ import annotations

from pathlib import Path

import modal

THIS_DIR = Path(__file__).resolve().parent

REMOTE_PROJECT = "/root/echo-rh-lean"
WARMUP_DIR = "/opt/lean-warmup"

# Read the toolchain spec from the local lean-toolchain so we can
# pre-install it at image build time.
#
# This file is also imported by the container itself when functions
# run, but at that point the image is already built and we don't
# need the value -- so we tolerate a missing file there.
_TOOLCHAIN_FILE = THIS_DIR / "lean-toolchain"
LEAN_TOOLCHAIN = (
    _TOOLCHAIN_FILE.read_text().strip()
    if _TOOLCHAIN_FILE.is_file()
    else "leanprover/lean4:v4.30.0-rc2"
)


# ---------------------------------------------------------------------------
# Image construction
# ---------------------------------------------------------------------------
#
# Order of operations matters:
#
#   1. apt + elan + toolchain (independent of project files)
#   2. copy lakefile.toml / lean-toolchain / lake-manifest.json into
#      /opt/lean-warmup with `copy=True` so subsequent `run_commands`
#      can see them
#   3. run lake update + lake exe cache get inside /opt/lean-warmup
#   4. mount the live project source at /root/echo-rh-lean (overlays
#      at runtime; doesn't trigger image rebuilds when source changes)

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "ca-certificates", "build-essential")
    .run_commands(
        # Install elan + the toolchain pinned by lean-toolchain.
        "curl -fsSL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh "
        f"| sh -s -- -y --default-toolchain {LEAN_TOOLCHAIN} --no-modify-path",
        f"/root/.elan/bin/elan toolchain install {LEAN_TOOLCHAIN}",
        "/root/.elan/bin/lean --version",
        "/root/.elan/bin/lake --version",
    )
    .env(
        {
            "PATH": "/root/.elan/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "ELAN_HOME": "/root/.elan",
        }
    )
    # Stage 2: copy manifest-determining files into the warmup dir.
    # `copy=True` materializes them on the image filesystem so the
    # following run_commands can use them.  Image rebuilds when these
    # contents change, but NOT when our EchoRH/*.lean files change.
    .add_local_file(
        THIS_DIR / "lakefile.toml",
        f"{WARMUP_DIR}/lakefile.toml",
        copy=True,
    )
    .add_local_file(
        THIS_DIR / "lean-toolchain",
        f"{WARMUP_DIR}/lean-toolchain",
        copy=True,
    )
    .add_local_file(
        THIS_DIR / "lake-manifest.json",
        f"{WARMUP_DIR}/lake-manifest.json",
        copy=True,
    )
    # Stage 3: pre-warm Mathlib oleans inside the warmup dir.
    .run_commands(
        f"mkdir -p {WARMUP_DIR}/EchoRH",
        f"echo 'def stub := 0' > {WARMUP_DIR}/EchoRH/Basic.lean",
        f"echo 'import EchoRH.Basic' > {WARMUP_DIR}/EchoRH.lean",
        f"cd {WARMUP_DIR} && lake update",
        f"cd {WARMUP_DIR} && lake exe cache get",
        f"du -sh {WARMUP_DIR}/.lake/packages/mathlib || true",
    )
    # Stage 4: live project source, overlaid at runtime.
    .add_local_dir(
        str(THIS_DIR),
        REMOTE_PROJECT,
        ignore=[".lake", ".lake/**", "__pycache__", "*.pyc", "modal_lean.py"],
    )
)

app = modal.App("echo-rh-lean", image=_image)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bash(cmd: str, *, quiet: bool = False) -> tuple[int, str]:
    """Run `cmd` in bash at REMOTE_PROJECT, capture combined output."""
    import subprocess

    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        cwd=REMOTE_PROJECT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        if not quiet:
            print(line, end="")
        chunks.append(line)
    proc.wait()
    return proc.returncode, "".join(chunks)


def _wire_up_packages() -> None:
    """Symlink `.lake/packages` from the warmup dir into the live
    project so lake can find pre-built Mathlib oleans without copying
    or re-resolving."""
    _bash(
        # Ensure the directory exists (overlay FS, fast).
        f"mkdir -p {REMOTE_PROJECT}/.lake && "
        # Idempotent symlink (force-replace if it already exists from
        # a previous container or a stray file).
        f"ln -sfn {WARMUP_DIR}/.lake/packages {REMOTE_PROJECT}/.lake/packages && "
        # Use the warmup's manifest exactly so dep versions agree.
        f"cp -f {WARMUP_DIR}/lake-manifest.json {REMOTE_PROJECT}/lake-manifest.json && "
        f"ls -la {REMOTE_PROJECT}/.lake/",
        quiet=True,
    )


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


@app.function(cpu=4.0, memory=8 * 1024, timeout=30 * 60)
def check() -> dict:
    """Verify the toolchain installs and `lake`/`lean` resolve."""
    rc1, _ = _bash("which elan && elan --version && lake --version && lean --version")
    rc2, _ = _bash(f"ls -la {WARMUP_DIR}/.lake/packages/ 2>&1 | head")
    _wire_up_packages()
    rc3, _ = _bash("ls -la .lake/")
    rc4, _ = _bash("time lake env true")
    return {
        "toolchain_rc": rc1,
        "warmup_rc": rc2,
        "wired_rc": rc3,
        "lake_env_rc": rc4,
    }


@app.function(cpu=8.0, memory=16 * 1024, timeout=60 * 60)
def build(target: str = "EchoRH") -> dict:
    """Build a Lake target.  Default is the whole `EchoRH` library.

    Args:
        target: Lake target name, e.g. `EchoRH`, `EchoRH.Basic`,
            `EchoRH.PoleShift`.
    """
    _wire_up_packages()

    print(f"\n=== building target: {target} ===")
    rc_ver, _ = _bash("lake --version && lean --version")
    if rc_ver != 0:
        return {"ok": False, "stage": "toolchain", "rc": rc_ver}

    rc, output = _bash(f"lake build {target}")

    sorries = output.count("uses 'sorry'")
    errors = output.count("error:")
    last_chunk = "".join(output.splitlines(keepends=True)[-30:])
    return {
        "ok": rc == 0,
        "target": target,
        "rc": rc,
        "sorry_warnings": sorries,
        "error_lines": errors,
        "last_lines": last_chunk,
    }


@app.function(cpu=8.0, memory=16 * 1024, timeout=30 * 60)
def axioms_of(decl: str = "EchoRH.gradeFourEigenvalue_ne_one") -> dict:
    """Print the axiom dependencies of a single declaration.

    Useful for verifying we don't sneak in non-classical axioms or
    leave `sorry` in load-bearing theorems.
    """
    _wire_up_packages()

    print("=== building EchoRH (so the decl exists) ===")
    rc_build, _ = _bash("lake build EchoRH 2>&1 | tail -5")
    if rc_build != 0:
        return {"ok": False, "stage": "build", "rc": rc_build}

    print(f"\n=== #print axioms {decl} ===")
    # Write a small Lean script and feed it via `lake env lean`.
    # `lake env` injects the right LEAN_PATH so `import EchoRH`
    # resolves to the just-built oleans.
    script = (
        "import EchoRH\n"
        f"#print axioms {decl}\n"
    )
    cmd = (
        f"cat > /tmp/axioms.lean <<'__EOF__'\n{script}__EOF__\n"
        "lake env lean /tmp/axioms.lean"
    )
    rc, output = _bash(cmd)
    return {"ok": rc == 0, "decl": decl, "rc": rc, "stdout_tail": output[-2000:]}


@app.local_entrypoint()
def main(action: str = "build", target: str = "EchoRH") -> None:
    """CLI dispatcher.

    Examples::

        modal run modal_lean.py
        modal run modal_lean.py --action check
        modal run modal_lean.py --action build --target EchoRH.Basic
        modal run modal_lean.py --action axioms --target EchoRH.gradeFourEigenvalue_ne_one
    """
    if action == "check":
        result = check.remote()
    elif action == "build":
        result = build.remote(target=target)
    elif action == "axioms":
        result = axioms_of.remote(decl=target)
    else:
        raise SystemExit(
            f"unknown action: {action!r}; "
            "use one of {check, build, axioms}"
        )
    print("\n=== result ===")
    print(result)
