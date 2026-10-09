"""Classroom entry point matching GW170817_tutorial.ipynb.

--mode fast --sample: reduced-band real-data conditional diagnostic.
--mode preview: display paths and quantiles from a bundled previous fast run.
--mode full --sample: long 17-parameter PE (22 parameters for hyperbolic).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["preview", "fast", "full"], default="fast")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--joint", action="store_true", help="run shared-distance GW–EM photosphere inference")
    p.add_argument("--lightcurve", action="store_true", help="also plot AT2017gfo photometry and fit its decline")
    p.add_argument("--likelihood", choices=["gaussian", "hyperbolic"], default="gaussian")
    p.add_argument("--steps", type=int)
    p.add_argument("--burn", type=int)
    p.add_argument("--outdir", type=Path)
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    if args.joint:
        if args.mode == "preview" or args.likelihood != "gaussian":
            p.error("--joint requires fast/full mode and Gaussian GW likelihood")
        command = [sys.executable, str(here / "gw170817_joint.py"), "--mode", args.mode]
        if args.sample:
            command.append("--sample")
        for name in ["steps", "burn", "outdir"]:
            if getattr(args, name) is not None:
                value = getattr(args, name)
                command += [f"--{name}", str(value.resolve() if name == "outdir" else value)]
        subprocess.run(command, cwd=root, check=True)
        return
    if args.lightcurve:
        subprocess.run([sys.executable, str(here / "at2017gfo_lightcurve.py"),
                        "--outdir", str(root / "results/at2017gfo_lightcurve")], cwd=root, check=True)
    if args.mode == "preview":
        demo = here / "gw170817_demo"
        print("Saved Gaussian fast run: no new inference is performed.")
        print((demo / "pilot_quantiles.json").read_text())
        print(f"Recovery plot: {demo / 'recovery.png'}")
        print(f"Corner plot: {demo / 'pilot_corner.png'}")
        return
    script = here / ("gw170817_pilot.py" if args.mode == "fast" else "gw170817_pe.py")
    command = [sys.executable, str(script), "--likelihood", args.likelihood]
    if args.mode == "fast":
        command.append("--fast")
    # Separate output directories preserve runs with different modes/likelihoods.
    output = args.outdir or root / "results" / f"gw170817_tutorial_{args.mode}_{args.likelihood}"
    command += ["--outdir", str(output.resolve())]
    if args.sample:
        command.append("--sample")
    for name in ["steps", "burn"]:
        if getattr(args, name) is not None:
            command += [f"--{name}", str(getattr(args, name))]
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
