"""Multidimensional low-spin BNS PE driver; convergence must be assessed externally.

Default: data setup and likelihood smoke check. --sample launches Eryn.
Uses the same cleaned-data/PSD setup as gw170817_pilot.py, with all CBC
parameters below sampled. This is an independent reanalysis configuration,
not an exact replication of a particular LVK release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import bilby
import numpy as np

from gw170817_pilot import TC, IFOS, build
from hyperwave.detectors.lvk import GW
from hyperwave.inference import LVKinference
from hyperwave.likelihoods import GWLikelihoods


def priors():
    uniform = bilby.core.prior.Uniform
    specs = dict(chirp_mass=(1.18, 1.22), mass_ratio=(0.5, 1.0),
                 psi=(0, np.pi), phase=(0, 2*np.pi), ra=(0, 2*np.pi),
                 a_1=(0, 0.05), a_2=(0, 0.05), cos_theta_jn=(-1, 1),
                 cos_tilt_1=(-1, 1), cos_tilt_2=(-1, 1),
                 phi_12=(0, 2*np.pi), phi_jl=(0, 2*np.pi),
                 geocent_time=(TC-0.1, TC+0.1), lambda_1=(0, 5000), lambda_2=(0, 5000))
    out = {name:uniform(lo, hi, name=name) for name, (lo, hi) in specs.items()}
    out["dec"] = bilby.core.prior.Cosine(name="dec")
    out["luminosity_distance"] = bilby.core.prior.PowerLaw(
        alpha=2, minimum=5, maximum=100, name="luminosity_distance")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("results/gw170817_full_gaussian"))
    p.add_argument("--likelihood", choices=["gaussian", "hyperbolic"], default="gaussian")
    p.add_argument("--approximant", default="IMRPhenomPv2_NRTidal")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--burn", type=int, default=1000)
    p.add_argument("--walkers", type=int, default=64)
    p.add_argument("--temperatures", type=int, default=4)
    p.add_argument("--seed", type=int, default=170817)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    # Reuse the pilot's cache without downloading another copy of the release.
    pilot, _, metadata = build(Path("results/gw170817"))
    prior = priors()
    gw = GW(pilot.noise, approximant=args.approximant, reference_frequency=50,
            parameters=list(prior), static_parameters={}, n_jobs=1)
    f = gw.frequency_array()
    psd = np.array([i.power_spectral_density_array[gw.mask] for i in pilot.noise.ifos])
    data = np.array([gw.detector_data_fd(i) for i in range(3)])
    like = GWLikelihoods(data, f, IFOS, psd, gw, ddims=False, nsegs=4,
                         gpu=False, cpu_cores=1)
    # A plausible point validates every sampled parameter, including tides.
    point = dict(chirp_mass=1.1975, mass_ratio=.85, luminosity_distance=45,
                 psi=0.2, phase=2.5, ra=3.44616, dec=-.40808,
                 a_1=.005, a_2=.005, cos_theta_jn=np.cos(.4),
                 cos_tilt_1=.9, cos_tilt_2=.9, phi_12=.2, phi_jl=.2,
                 geocent_time=TC, lambda_1=400, lambda_2=400)
    nuisance = {}
    if args.likelihood == "hyperbolic":
        nuisance = {name:bilby.core.prior.Uniform(1e-3, 30, name=name)
                    for name in ["alpha", *[f"delta_{i}" for i in range(4)]]}
    theta = np.r_[[point[name] for name in prior], [5]*len(nuisance)]
    fn = like.gaussian if args.likelihood == "gaussian" else like.hyperbolic_classic
    value = np.asarray(fn(np.stack([theta, theta])))
    if not np.all(np.isfinite(value)) or np.any(value <= -1e299):
        raise ValueError("Likelihood smoke check failed")
    metadata.update(approximant=args.approximant, parameters=list(prior), fixed={},
                    bounds={k:[v.minimum, v.maximum] for k,v in prior.items()},
                    priors={k:repr(v) for k,v in prior.items()},
                    noise_priors={k:repr(v) for k,v in nuisance.items()},
                    likelihood=args.likelihood, seed=args.seed,
                    calibration="fixed nominal response; no calibration marginalization",
                    setup_log_likelihood=value.tolist(),
                    convergence="not assessed; default step counts do not guarantee convergence")
    metadata["hyperwave_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata["local_diff"] = subprocess.check_output(["git", "diff"], text=True)
    metadata["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata["data_loader_sha256"] = hashlib.sha256(Path(__file__).with_name("gw170817_pilot.py").read_bytes()).hexdigest()
    metadata["environment"] = subprocess.check_output([".venv/bin/python", "-m", "pip", "freeze"], text=True).splitlines()
    (args.outdir / "manifest.json").write_text(json.dumps(metadata, indent=2))
    print(f"Setup OK: {len(prior)} science + {len(nuisance)} noise parameters, likelihood {value}")
    if not args.sample:
        return
    if args.walkers < 2*len(theta):
        raise ValueError("Use at least twice as many walkers as dimensions")
    bilby.core.utils.random.seed(args.seed)
    np.random.seed(args.seed)
    # Prior initialization intentionally explores the specified prior.
    # A known-event warm start can accelerate sampling, but is not a convergence check.
    inf = LVKinference(fn, "eryn", prior, nuisance,
                       dict(save_dir=str(args.outdir), TAG=args.likelihood, like=args.likelihood),
                       sampler_kwargs=dict(nwalkers=args.walkers, ntemps=args.temperatures,
                                           burn=args.burn, nsteps=args.steps),
                       periodic=["phase", "psi", "ra", "phi_12", "phi_jl"])
    inf.run()
    np.save(args.outdir / "samples.npy", inf.get_samples(thin=1))


if __name__ == "__main__":
    main()
