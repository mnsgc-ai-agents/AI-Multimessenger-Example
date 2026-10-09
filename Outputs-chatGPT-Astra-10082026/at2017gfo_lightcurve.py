"""AT2017gfo classroom photometry and phenomenological decline fits.

Uses a bundled, version-pinned NMMA example data file. No NMMA installation or
network access is required. Fits are descriptive; they do not infer ejecta,
distance, inclination, an EOS, or a joint GW-EM posterior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

DATA_DIR = Path(__file__).resolve().parent / "at2017gfo_data"
GW_GPS = 1187008882.43
OPTICAL = ["ps1::g", "ps1::r", "ps1::i", "ps1::z", "ps1::y"]
INFRARED = ["2massj", "2massh", "2massks"]
LABELS = {**{f"ps1::{b}":b for b in "grizy"}, "2massj":"J", "2massh":"H", "2massks":"Ks"}


def load_photometry(gps_time=GW_GPS):
    path = DATA_DIR / "AT2017gfo.dat"
    provenance = json.loads((DATA_DIR / "provenance.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["sha256"]:
        raise ValueError("Bundled photometry checksum does not match its provenance")
    table = pd.read_csv(path, sep=r"\s+", names=["utc", "band", "magnitude", "sigma_mag"])
    if not np.all(np.isfinite(table.magnitude)) or np.any(table.sigma_mag <= 0) or table.sigma_mag.isna().any():
        raise ValueError("Invalid photometry")
    # Convert GPS to UTC before subtracting astronomical MJD timestamps.
    merger_mjd_utc = float(Time(gps_time, format="gps").utc.mjd)
    table["mjd_utc"] = Time(table.utc.to_numpy(dtype=str), format="isot", scale="utc").mjd
    table["time_days"] = table.mjd_utc - merger_mjd_utc
    table["is_upper_limit"] = ~np.isfinite(table.sigma_mag)
    return table.sort_values(["time_days", "band"]).reset_index(drop=True), provenance


def fit_decline(table, bands=OPTICAL[:4], tmin=1., tmax=5., systematic_mag=.15):
    """Weighted m(t)=a+b log10(t/day); alpha=b/2.5 for F_nu proportional to t^-alpha.

    Formal covariance assumes this model and fixed independent Gaussian errors.
    The chosen systematic floor is an illustrative assumption, not inferred.
    """
    if not (0 < tmin < tmax) or systematic_mag < 0:
        raise ValueError("Require 0 < tmin < tmax and a nonnegative systematic floor")
    rows = []
    for band in bands:
        data = table[(table.band == band) & ~table.is_upper_limit
                     & table.time_days.between(tmin, tmax)]
        if len(data) < 3:
            continue
        sigma = np.hypot(data.sigma_mag.to_numpy(), systematic_mag)
        design = np.column_stack([np.ones(len(data)), np.log10(data.time_days)])
        weighted = design / sigma[:, None]
        coeff, _, rank, _ = np.linalg.lstsq(weighted, data.magnitude.to_numpy()/sigma, rcond=None)
        if rank != 2:
            raise ValueError(f"Insufficient distinct times for {band}")
        covariance = np.linalg.inv(weighted.T @ weighted)
        residual = (data.magnitude.to_numpy() - design @ coeff) / sigma
        rows.append(dict(band=band, n_points=len(data), magnitude_at_1day=coeff[0],
                         slope_mag_per_dex=coeff[1], alpha=coeff[1]/2.5,
                         alpha_formal_sigma=np.sqrt(covariance[1,1])/2.5,
                         reduced_chi2=float(residual@residual/(len(data)-2)),
                         fit_start_days=tmin, fit_end_days=tmax, systematic_mag=systematic_mag))
    return pd.DataFrame(rows)


def plot_photometry(table, fits=None, outdir=None):
    with plt.rc_context({"font.size":10, "axes.labelsize":11, "axes.titlesize":12}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
        for ax, bands, title in zip(axes, [OPTICAL, INFRARED], ["Optical photometry", "Near-infrared photometry"]):
            for band in bands:
                data = table[table.band == band]
                detections = data[~data.is_upper_limit]
                line = ax.errorbar(detections.time_days, detections.magnitude,
                                  yerr=detections.sigma_mag, fmt="o", markersize=4,
                                  capsize=2, label=LABELS[band])
                limits = data[data.is_upper_limit]
                ax.scatter(limits.time_days, limits.magnitude, marker="v", color=line[0].get_color())
                if fits is not None and not fits.empty:
                    row = fits[fits.band == band]
                    if not row.empty:
                        r = row.iloc[0]
                        t = np.linspace(r.fit_start_days, r.fit_end_days, 150)
                        ax.plot(t, r.magnitude_at_1day+r.slope_mag_per_dex*np.log10(t),
                                color=line[0].get_color(), linewidth=1.5)
            ax.invert_yaxis()
            ax.set(xscale="log", xlabel="Observer-frame days since GW merger", title=title)
            ax.legend(fontsize=9)
            ax.grid(alpha=.2)
        axes[0].set_ylabel("Magnitude as supplied (no additional correction)")
        fig.suptitle("AT2017gfo: real photometry; lines = descriptive decline fits", fontsize=12)
        fig.tight_layout()
        if outdir is not None:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(outdir) / "at2017gfo_lightcurve.png", dpi=160)
        return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("results/at2017gfo_lightcurve"))
    p.add_argument("--fit-start", type=float, default=1.)
    p.add_argument("--fit-end", type=float, default=5.)
    p.add_argument("--systematic-mag", type=float, default=.15)
    args = p.parse_args()
    table, provenance = load_photometry()
    fits = fit_decline(table, tmin=args.fit_start, tmax=args.fit_end, systematic_mag=args.systematic_mag)
    args.outdir.mkdir(parents=True, exist_ok=True)
    fits.to_csv(args.outdir / "decline_fits.csv", index=False)
    provenance.update(gw_reference_gps=GW_GPS, merger_mjd_utc=float(Time(GW_GPS, format="gps").utc.mjd),
                      fit_start=args.fit_start, fit_end=args.fit_end, systematic_mag=args.systematic_mag,
                      interpretation="Descriptive detection-only magnitude fits; not joint GW-EM PE")
    (args.outdir / "manifest.json").write_text(json.dumps(provenance, indent=2))
    plt.close(plot_photometry(table, fits, args.outdir))
    print(fits.to_string(index=False))
    print("Flux upper limits are displayed but excluded from the Gaussian decline fits.")


if __name__ == "__main__":
    main()
