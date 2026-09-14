"""
fetch_robot.py
---------------
pull the UR5e model from MuJoCo Menagerie into this repo.

Menagerie (github.com/google-deepmind/mujoco_menagerie) hosts ~60 robots,
~1 GB total. We only want one folder: universal_robots_ur5e/. Cloning the
whole repo just to keep one folder wastes bandwidth and disk, so this script
uses git's sparse-checkout to pull only that path into a scratch directory,
copies it into assets/robots/ur5e/, and discards the rest of the clone (the
scratch clone's .git never touches this repo).

The model is committed to this repo, not re-fetched on every setup, so a
fresh clone of THIS project works offline. Re-run this script only when you
want to pull an update from upstream Menagerie.

    python scripts/fetch_robot.py            fetch (or re-fetch) the model
    python scripts/fetch_robot.py --check    verify the already-fetched
                                              model compiles -- no network
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco

REPO_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
SUBFOLDER = "universal_robots_ur5e"
DEST_DIR = Path(__file__).resolve().parent.parent / "assets" / "robots" / "ur5e"


# ----------------------------------------------------------------------
# Fetch: sparse-clone into a scratch dir, copy the one folder out, record
# which upstream commit it came from.
# ----------------------------------------------------------------------

def fetch_robot() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)

        # NOTE: `git clone --sparse` (the one-flag shortcut) is buggy on
        # git < 2.35 (e.g. Ubuntu 20.04's stock 2.25.1) -- it runs
        # `git -C <repo-url> sparse-checkout init` internally, passing the
        # URL as if it were the local clone path, and fails. Doing the same
        # three steps by hand works on every git version.
        subprocess.run(
            [
                "git", "clone",
                "--filter=blob:none", "--no-checkout", "--depth", "1",
                REPO_URL, str(scratch_path),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "init", "--cone"],
            cwd=scratch_path,
            check=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", SUBFOLDER],
            cwd=scratch_path,
            check=True,
        )
        subprocess.run(
            ["git", "checkout"],
            cwd=scratch_path,
            check=True,
        )

        fetched = scratch_path / SUBFOLDER
        if not fetched.is_dir():
            sys.exit(f"error: expected {fetched} after sparse-checkout, but it is missing")

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=scratch_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        if DEST_DIR.exists():
            shutil.rmtree(DEST_DIR)
        DEST_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(fetched, DEST_DIR)

    provenance = DEST_DIR / "PROVENANCE.md"
    provenance.write_text(
        f"Fetched from: {REPO_URL}\n"
        f"Subfolder:    {SUBFOLDER}\n"
        f"Commit:       {commit}\n"
    )
    print(f"fetched {SUBFOLDER} @ {commit[:8]} -> {DEST_DIR}")


# ----------------------------------------------------------------------
# Verify: the model actually compiles. This is what --check runs, and what
# fetch_robot() also runs at the end -- exit 0 alone doesn't prove the XML
# is valid, only that git didn't error.
# ----------------------------------------------------------------------

def verify() -> None:
    xml_path = DEST_DIR / "ur5e.xml"
    if not xml_path.exists():
        sys.exit(f"error: {xml_path} not found -- run without --check to fetch it first")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    print(f"ok: {xml_path} compiles -- {model.nbody} bodies, {model.ngeom} geoms, {model.nq} DOF")


def main() -> None:
    if "--check" in sys.argv:
        verify()
        return

    fetch_robot()
    verify()


if __name__ == "__main__":
    main()
