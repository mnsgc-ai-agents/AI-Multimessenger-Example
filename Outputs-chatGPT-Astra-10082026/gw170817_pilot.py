"""GW170817 recovery and conditional PE using version-pinned cleaned GWOSC data.

Run from the repository root using .venv/bin/python. Default is setup/recovery;
--sample runs a short conditional Eryn diagnostic, not a converged LVK posterior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.optimize import minimize
from scipy.signal import welch

from hyperwave.detectors.lvk import DetectorNoise, GW
from hyperwave.detectors.psd import PowerSpectralDensity
from hyperwave.likelihoods import GWLikelihoods

EVENT_API = "https://gwosc.org/eventapi/json/O1_O2-Preliminary/GW170817/v2/"
TC = 1187008882.43
FS = 4096
DURATION = 256
IFOS = ["H1", "L1", "V1"]
NAMES = ["chirp_mass", "luminosity_distance", "phase", "geocent_time"]
# Deliberately conditional: host position, mass ratio, spins, tides and
# orientation fixed. Detector-frame chirp mass; this is NOT source-frame mass.
FIXED = dict(mass_ratio=0.85, ra=3.44616, dec=-0.40808, psi=0.0,
             cos_theta_jn=np.cos(0.4), chi_1=0.0, chi_2=0.0,
             lambda_1=400.0, lambda_2=400.0)
BOUNDS = [(1.195, 1.200), (5.0, 150.0), (0.0, 2*np.pi), (TC-0.1, TC+0.1)]
FAST_CONFIG = dict(duration=32, fs=2048, fmin=64, fmax=512)


def fetch(url, path):
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=180) as src, tmp.open("wb") as dst:
            while block := src.read(1024 * 1024):
                dst.write(block)
        tmp.replace(path)


def build(outdir, *, duration=DURATION, fs=FS, fmin=23, fmax=1024, cache_dir=None):
    cache = Path(cache_dir) if cache_dir is not None else outdir / "data"
    cache.mkdir(parents=True, exist_ok=True)
    fetch(EVENT_API, cache / "event.json")
    event = json.loads((cache / "event.json").read_text())["events"]["GW170817-v2"]
    noise = DetectorNoise(duration, fs, TC, IFOS, minimum_frequency=fmin,
                          maximum_frequency=fmax, post_trigger_duration=2)
    # Integer-aligned data cuts, with merger safely inside the segment.
    start = np.floor(TC) + 2 - duration
    noise._start_time, noise._end_time = start, start + duration
    manifest = []
    for ifo in noise.ifos:
        entry, = [s for s in event["strain"] if s["detector"] == ifo.name
                  and s["sampling_rate"] == FS and s["format"] == "hdf5"]
        path = cache / entry["url"].rsplit("/", 1)[-1]
        print(f"> {ifo.name}: fetching/reading cleaned v2 data", flush=True)
        fetch(entry["url"], path)
        ts = TimeSeries.read(path, format="hdf5.gwosc")
        # 512 s of off-source data ending 32 s before the analysis segment.
        # Avoid both the BNS signal and the corrupted final H1 data interval.
        off = ts.crop(start-544, start-32)
        segment = ts.crop(start, start+duration)
        if fs != FS:
            off, segment = off.resample(fs), segment.resample(fs)
        if len(segment) != duration*fs or len(off) != 512*fs:
            raise ValueError("Unexpected strain length")
        if not np.all(np.isfinite(segment.value)) or not np.all(np.isfinite(off.value)):
            raise ValueError("Nonfinite strain")
        freq, psd = welch(off.value, fs=fs, window="hann", nperseg=16*fs,
                          noverlap=8*fs, detrend="constant", average="median")
        ifo.power_spectral_density = PowerSpectralDensity(freq, psd)
        ifo.strain_data.set_from_gwpy_timeseries(segment)
        manifest.append(dict(detector=ifo.name, url=entry["url"],
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    gw = GW(noise, approximant="TaylorF2", reference_frequency=50,
            parameters=NAMES, static_parameters=FIXED, n_jobs=1)
    freq = gw.frequency_array()
    psd = np.array([i.power_spectral_density_array[gw.mask] for i in noise.ifos])
    data = np.array([gw.detector_data_fd(i) for i in range(3)])
    if not np.all(np.isfinite(psd) & (psd > 0)):
        raise ValueError("Invalid analysis PSD")
    like = GWLikelihoods(data, freq, IFOS, psd, gw, ddims=False, nsegs=4,
                         gpu=False, cpu_cores=1)
    metadata = dict(event_api=EVENT_API, files=manifest, fs=fs, duration=duration,
                    analysis_start=start, fmin=fmin, fmax=fmax,
                    psd_interval=[start-544, start-32], psd_method="median Welch, 16 s Hann, 50% overlap",
                    approximant="TaylorF2", parameters=NAMES, fixed=FIXED,
                    bounds=BOUNDS, convergence="not assessed")
    return gw, like, metadata


def recover(gw, like, outdir):
    # Search a small chirp-mass grid around the known event, with fixed q/tides.
    # Phase-maximized single-template correlations; no template-bank/background.
    n = int(gw.duration * gw.sampling_rate)
    lag = np.fft.fftfreq(n) * gw.duration  # signed time lags in seconds
    select = np.abs(lag) < 0.1
    order = np.argsort(lag[select])
    lags = lag[select][order]
    best = None
    df = 1 / gw.duration
    for mc in np.linspace(1.196, 1.199, 31):
        theta = np.array([mc, 40.0, 0.0, TC])
        h = gw.make_injections_to_ifo_batch(theta[None])[0]
        curves = []
        for j in range(3):
            product = like.data[j] * h[j].conj() / like.psd[j]
            spectrum = np.zeros(n, complex)
            spectrum[np.flatnonzero(gw.mask)] = product
            sigma = np.sqrt(4*df*np.sum(np.abs(h[j])**2 / like.psd[j]))
            z = 4*df*n*np.fft.ifft(spectrum) / sigma
            curves.append(np.abs(z[select])[order])
        network = np.sqrt(np.sum(np.array(curves)**2, axis=0))
        index = int(np.argmax(network))
        if best is None or network[index] > best[0]:
            best = (float(network[index]), mc, float(lags[index]), curves)
    snr, mc, dt, curves = best
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, curve in zip(IFOS, curves):
        ax.plot(lags, curve, label=name)
    ax.set(xlabel="Time offset from reference template (s)", ylabel="Matched-filter |SNR|",
           title="GW170817 cleaned data: targeted template recovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "recovery.png", dpi=160)
    plt.close(fig)
    result = dict(targeted_quadrature_snr=snr, detector_frame_chirp_mass_grid=mc,
                  lag_seconds=dt, significance="not estimated; targeted recovery only")
    (outdir / "recovery.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    # Optimize conditional Gaussian likelihood with time represented as offset
    # for numerical conditioning (the waveform still receives absolute GPS).
    def unpack(x):
        return np.array([x[0], x[1], x[2] % (2*np.pi), TC+x[3]])
    def objective(x):
        return -float(like.gaussian(unpack(x)))
    bounds = [BOUNDS[0], BOUNDS[1], (-2*np.pi, 4*np.pi), (-0.1, 0.1)]
    trials = [minimize(objective, [mc, 40, phase, dt], method="Nelder-Mead",
                       bounds=bounds, options=dict(maxiter=500, xatol=1e-7, fatol=1e-4))
              for phase in [0, np.pi/2, np.pi, 3*np.pi/2]]
    opt = min(trials, key=lambda r:r.fun)
    theta = unpack(opt.x)
    diagnostics = dict(conditional_best_fit=dict(zip(NAMES, theta.tolist())),
                       optimizer_success=bool(opt.success),
                       gaussian_log_likelihood=float(like.gaussian(theta)),
                       gaussian_log_likelihood_gain_vs_noise=float(like.gaussian(theta)-np.sum(like.yy_noise)))
    (outdir / "setup.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2), flush=True)
    return theta


def sample(like, theta, args):
    import bilby
    from hyperwave.inference import LVKinference
    bilby.core.utils.random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    priors = {name:bilby.core.prior.Uniform(lo, hi, name=name)
              for name, (lo, hi) in zip(NAMES, BOUNDS)}
    nuisance = {}
    if args.likelihood == "hyperbolic":
        nuisance = {"alpha":bilby.core.prior.Uniform(1e-3, 30, name="alpha")}
        nuisance.update({f"delta_{i}":bilby.core.prior.Uniform(1e-3, 30, name=f"delta_{i}")
                         for i in range(4)})
    center = np.r_[theta, [5.0]*len(nuisance)]
    scales = np.r_[[1e-5, 1, 0.05, 0.001], [0.2]*len(nuisance)]
    walkers = max(24, 2*len(center)+2)
    coords = center + rng.normal(size=(2, walkers, len(center)))*scales
    function = like.gaussian if args.likelihood == "gaussian" else like.hyperbolic_classic
    inf = LVKinference(function, "eryn", priors, nuisance,
                       dict(save_dir=str(args.outdir), TAG=args.likelihood, like=args.likelihood),
                       sampler_kwargs=dict(nwalkers=walkers, ntemps=2, burn=args.burn,
                                           nsteps=args.steps, init_coords=coords), periodic=["phase"])
    inf.run()
    samples = inf.get_samples(thin=1)
    np.save(args.outdir / "pilot_samples.npy", samples)
    summary = {name:np.quantile(samples[:, i], [0.05, 0.5, 0.95]).tolist()
               for i, name in enumerate([*priors, *nuisance])}
    (args.outdir / "pilot_quantiles.json").write_text(json.dumps(dict(
        warning="Short conditional diagnostic; convergence not established; not LVK PE reproduction",
        likelihood=args.likelihood, seed=args.seed, burn=args.burn, steps=args.steps,
        quantiles_5_50_95=summary), indent=2))
    plot_samples(samples, args.outdir)


def plot_samples(samples, outdir):
    import corner
    display = samples[:, :4].copy()
    display[:, 3] -= TC
    labels = [r"$\mathcal{M}_{\rm det}\ (M_\odot)$", r"$d_L\ ({\rm Mpc})$",
              r"$\phi\ ({\rm rad})$", r"$t_c-t_0\ ({\rm s})$"]
    with plt.rc_context({"font.size":10, "xtick.labelsize":9, "ytick.labelsize":9}):
        fig = corner.corner(display, labels=labels, label_kwargs={"fontsize":12})
        fig.subplots_adjust(top=.92)
        fig.suptitle("GW170817 conditional pilot — convergence not established", fontsize=12)
        fig.savefig(outdir / "pilot_corner.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--data-cache", type=Path, help="shared directory containing downloaded strain files")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--fast", action="store_true", help="32 s, 64–512 Hz classroom demonstration")
    parser.add_argument("--reuse-fit", action="store_true",
                        help="reuse an existing Gaussian fit after verifying the analysis settings")
    parser.add_argument("--likelihood", choices=["gaussian", "hyperbolic"], default="gaussian")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--burn", type=int)
    parser.add_argument("--seed", type=int, default=170817)
    args = parser.parse_args()
    args.outdir = args.outdir or Path("results/gw170817_fast" if args.fast else "results/gw170817")
    if args.fast and args.data_cache is None:
        args.data_cache = Path("results/gw170817/data")
    args.steps = args.steps if args.steps is not None else (20 if args.fast else 100)
    args.burn = args.burn if args.burn is not None else (10 if args.fast else 50)
    args.outdir.mkdir(parents=True, exist_ok=True)
    previous = json.loads((args.outdir / "manifest.json").read_text()) if args.reuse_fit else None
    gw, like, metadata = build(args.outdir, cache_dir=args.data_cache,
                               **(FAST_CONFIG if args.fast else {}))
    metadata["hyperwave_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata["local_diff"] = subprocess.check_output(["git", "diff"], text=True)
    metadata["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata["environment"] = subprocess.check_output([".venv/bin/python", "-m", "pip", "freeze"], text=True).splitlines()
    if args.reuse_fit:
        keys = ["event_api", "fs", "duration", "analysis_start", "fmin", "fmax",
                "psd_interval", "approximant", "parameters", "fixed", "bounds", "files"]
        for key in keys:
            if previous[key] != json.loads(json.dumps(metadata[key])):
                raise ValueError(f"Cached fit has different analysis setting: {key}")
        fit = json.loads((args.outdir / "setup.json").read_text())["conditional_best_fit"]
        theta = np.array([fit[name] for name in NAMES])
    else:
        theta = recover(gw, like, args.outdir)
    (args.outdir / "manifest.json").write_text(json.dumps(metadata, indent=2))
    if args.sample:
        sample(like, theta, args)


if __name__ == "__main__":
    main()
