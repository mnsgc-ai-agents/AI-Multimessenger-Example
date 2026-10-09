"""Conditional GW170817–AT2017gfo inference with a cooling blackbody photosphere.

Fast: four GW + four EM parameters, real data, shared distance and merger epoch.
Full: seventeen GW + four EM parameters; the SAME approximate EM model.
Neither route is an EOS/ejecta-mass analysis. Step counts do not ensure convergence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.constants import h, c, k, parsec

from at2017gfo_lightcurve import DATA_DIR, load_photometry
from gw170817_pilot import TC, NAMES, BOUNDS, FAST_CONFIG, IFOS, build, recover

EM_NAMES = ['log10_R1_cm', 'log10_T1_K', 'radius_index', 'cooling_index']
EM_BOUNDS = [(14., 15.7), (3.5, 4.3), (0., 1.5), (0., 1.5)]
WARNING = 'Teaching diagnostic; convergence not established; conditional on an approximate single-photosphere EM model.'


class PhotosphereLikelihood:
    """Band-integrated AB magnitudes, observer-frame dust, fixed z and scatter.

    R1 and T1 refer to one REST-FRAME day. No heating, opacity, ejecta-mass,
    viewing-angle or EOS relation is supplied by this phenomenological model.
    """
    def __init__(self, tmin=.4, tmax=2.5, systematic_mag=.20, ebv=.105, redshift=.0098):
        if not (0 < tmin < tmax) or systematic_mag < 0 or ebv < 0 or redshift < 0:
            raise ValueError('Invalid time window, scatter, extinction or redshift')
        table, self.provenance = load_photometry()
        selected = table.band.isin(['ps1::'+b for b in 'grizy']) & table.time_days.between(tmin, tmax) & ~table.is_upper_limit
        self.data = table[selected].copy().reset_index(drop=True)
        if self.data.empty:
            raise ValueError('No detections selected')
        self.excluded_rows = int((~selected).sum())
        self.config = dict(tmin=tmin, tmax=tmax, systematic_mag=systematic_mag, ebv=ebv, redshift=redshift)
        self.sigma = np.hypot(self.data.sigma_mag.to_numpy(), systematic_mag)
        folder = DATA_DIR / 'bandpasses'
        self.band_provenance = json.loads((folder / 'provenance.json').read_text())
        self.bands = {}
        for band in 'grizy':
            path = folder / (band+'.txt')
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.band_provenance['sha256'][path.name]:
                raise ValueError('Bandpass checksum mismatch')
            wave, response, dust = np.loadtxt(path).T
            # Photon-counting AB average: integral(f_nu T d lambda/lambda) / integral(T d lambda/lambda).
            # Trapezoidal endpoint weights on the bundled uniform wavelength grid.
            weight = response / wave
            weight[[0, -1]] *= .5
            weight /= weight.sum()
            self.bands['ps1::'+band] = (wave*1e-10, weight, 10**(-.4*ebv*dust))

    def magnitudes(self, em_theta, distance, geocent_time=TC, times=None, bands=None):
        pars = np.atleast_2d(em_theta)
        n = len(pars)
        distance = np.broadcast_to(np.asarray(distance), (n,))
        epoch = np.broadcast_to(np.asarray(geocent_time), (n,))
        if times is None:
            times = self.data.time_days.to_numpy()
            bands = self.data.band.to_numpy()
        times, bands = np.asarray(times), np.asarray(bands)
        z = self.config['redshift']
        rest = (times[None, :] - (epoch[:, None]-TC)/86400.)/(1+z)
        if np.any(rest <= 0) or np.any(distance <= 0):
            raise ValueError('Photosphere requires positive phase and distance')
        radius_m = 10**pars[:, 0, None]*.01 * rest**pars[:, 2, None]
        temperature = 10**pars[:, 1, None] * rest**(-pars[:, 3, None])
        result = np.empty_like(rest)
        for band in np.unique(bands):
            idx = np.flatnonzero(bands == band)
            wavelength, weight, dust = self.bands[band]
            frequency = (1+z)*c/wavelength
            exponent = h*frequency[None, None, :]/(k*temperature[:, idx, None])
            planck = 2*h*frequency[None, None, :]**3/c**2 / np.expm1(np.clip(exponent, 1e-12, 700))
            # L_nu=4*pi^2*R^2*B_nu; f_nu_obs=(1+z)*L_nu_em/(4*pi*D_L^2).
            fnu = (1+z)*np.pi*(radius_m[:, idx, None]/(distance[:, None, None]*1e6*parsec))**2 * planck
            jy = np.sum(fnu*dust[None, None, :]*weight[None, None, :], axis=-1)/1e-26
            result[:, idx] = -2.5*np.log10(jy/3631.)
        return result

    def __call__(self, em_theta, distance, geocent_time=TC):
        prediction = self.magnitudes(em_theta, distance, geocent_time)
        residual = (self.data.magnitude.to_numpy()[None, :]-prediction)/self.sigma
        return -.5*np.sum(residual**2 + np.log(2*np.pi*self.sigma**2), axis=1)

    def fit(self, distance, epoch=TC):
        fit = least_squares(lambda x: ((self.magnitudes(x, distance, epoch)[0]-self.data.magnitude)/self.sigma).to_numpy(),
                            [14.85, 3.85, .6, .5], bounds=np.array(EM_BOUNDS).T)
        if not fit.success:
            raise RuntimeError(f'EM initialization failed: {fit.message}')
        return fit.x


class JointLikelihood:
    def __init__(self, gw_likelihood, gw_names, em):
        self.gw_likelihood, self.gw_names, self.em = gw_likelihood, list(gw_names), em
        self.ndim_gw = len(gw_names)
        self.distance_index = self.gw_names.index('luminosity_distance')
        self.time_index = self.gw_names.index('geocent_time')

    def components(self, theta):
        theta = np.atleast_2d(theta)
        if theta.shape[1] != self.ndim_gw+len(EM_NAMES):
            raise ValueError('Unexpected number of joint parameters')
        gw = np.asarray(self.gw_likelihood(theta[:, :self.ndim_gw])).reshape(-1)
        em = self.em(theta[:, self.ndim_gw:], theta[:, self.distance_index], theta[:, self.time_index])
        return gw, em

    def __call__(self, theta):
        gw, em = self.components(theta)
        return gw+em  # Eryn supplies the ONE joint prior; never add a GW posterior here.


def fast_priors():
    from bilby.core.prior import Uniform
    return {name:Uniform(lo, hi, name=name) for name, (lo, hi) in zip(NAMES, BOUNDS)}


def run_joint(gw_likelihood, gw_priors, gw_center, outdir, *, burn=10, steps=40, seed=170818,
              walkers=None, temperatures=2, sample=True, em=None):
    import bilby
    from hyperwave.inference import LVKinference
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    em = em if em is not None else PhotosphereLikelihood()
    joint = JointLikelihood(gw_likelihood, list(gw_priors), em)
    priors = dict(gw_priors)
    priors.update({name:bilby.core.prior.Uniform(lo, hi, name=name) for name, (lo, hi) in zip(EM_NAMES, EM_BOUNDS)})
    center = np.r_[gw_center, em.fit(gw_center[joint.distance_index], gw_center[joint.time_index])]
    lg, le = joint.components(center)
    if not np.all(np.isfinite(lg+le)):
        raise ValueError('Nonfinite joint likelihood')
    walkers = walkers or max(24, 2*len(priors)+2)
    if walkers < 2*len(priors) or burn < 0 or steps < 1 or temperatures < 1:
        raise ValueError('Require walkers >= 2*ndim, burn >= 0, steps >= 1, temperatures >= 1')
    info = dict(warning=WARNING, parameters=list(priors), priors={n:repr(p) for n,p in priors.items()},
                initialization='Local warm start; not evidence of mode exploration or convergence',
                em_model='R=R1*t_rest^radius_index; T=T1*t_rest^-cooling_index; isotropic blackbody',
                shared_parameters=['luminosity_distance', 'geocent_time'], em_config=em.config,
                em_rows=len(em.data), excluded_em_rows=em.excluded_rows,
                photometry=em.provenance, bandpasses=em.band_provenance,
                logL_gw_at_initial_center=float(lg[0]), logL_em_at_initial_center=float(le[0]),
                seed=seed, burn=burn, steps=steps, walkers=walkers, temperatures=temperatures,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (outdir/'joint_manifest.json').write_text(json.dumps(info, indent=2))
    em.data.to_csv(outdir/'selected_photometry.csv', index=False)
    if not sample:
        return joint, center, None
    bilby.core.utils.random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    scale = np.array([.002*(p.maximum-p.minimum) for p in priors.values()])
    scale[joint.time_index] = .0003
    scale[list(priors).index('chirp_mass')] = 1e-5
    scale[-4:] = [.01, .005, .02, .02]
    coords = center + rng.normal(size=(temperatures, walkers, len(priors)))*scale
    low, high = np.array([(p.minimum, p.maximum) for p in priors.values()]).T
    coords = np.clip(coords, low+1e-6*(high-low), high-1e-6*(high-low))
    periodic = [name for name in ['phase', 'psi', 'ra', 'phi_12', 'phi_jl'] if name in priors]
    inf = LVKinference(joint, 'eryn', priors, {}, dict(save_dir=str(outdir), TAG='joint', like='gaussian'),
                      sampler_kwargs=dict(nwalkers=walkers, ntemps=temperatures, burn=burn,
                                          nsteps=steps, init_coords=coords), periodic=periodic)
    inf.run()
    samples = inf.get_samples(thin=1)
    np.save(outdir/'joint_samples.npy', samples)
    summary = {name:np.quantile(samples[:, i], [.05,.5,.95]).tolist() for i,name in enumerate(priors)}
    (outdir/'joint_quantiles.json').write_text(json.dumps(dict(warning=WARNING, quantiles_5_50_95=summary), indent=2))
    plot_joint(samples, joint, outdir)
    return joint, center, samples


def plot_joint(samples, joint, outdir):
    import matplotlib.pyplot as plt
    import corner
    with plt.rc_context({'font.size':10, 'axes.labelsize':10, 'axes.titlesize':11,
                         'xtick.labelsize':9, 'ytick.labelsize':9, 'legend.fontsize':9,
                         'figure.titlesize':12}):
        outdir = Path(outdir)
        indices = [joint.distance_index, *range(joint.ndim_gw, joint.ndim_gw+4)]
        fig = corner.corner(samples[:, indices], labels=['D_L [Mpc]', 'log10 R1 [cm]', 'log10 T1 [K]', 'Radius index', 'Cooling index'],
                            quiet=True, show_titles=False)
        fig.suptitle('Joint GW–EM diagnostic — convergence not established', fontsize=12)
        fig.savefig(outdir/'joint_corner.png', dpi=130)
        plt.close(fig)
        em = joint.em
        fig, axes = plt.subplots(1,2,figsize=(11,4))
        draws = samples[np.linspace(0,len(samples)-1,min(100,len(samples))).astype(int)]
        for band in em.bands:
            rows = em.data[em.data.band==band]
            if rows.empty:
                continue
            line = axes[0].errorbar(rows.time_days,rows.magnitude,yerr=np.hypot(rows.sigma_mag,em.config['systematic_mag']),fmt='o',ms=3,label=band[-1])
            times = np.geomspace(em.config['tmin'],em.config['tmax'],80)
            models = em.magnitudes(draws[:,joint.ndim_gw:],draws[:,joint.distance_index],draws[:,joint.time_index],times=times,bands=np.repeat(band,len(times)))
            lo,mid,hi = np.quantile(models,[.05,.5,.95],axis=0)
            color=line[0].get_color()
            axes[0].plot(times,mid,color=color)
            axes[0].fill_between(times,lo,hi,color=color,alpha=.15)
        prediction = em.magnitudes(draws[:,joint.ndim_gw:],draws[:,joint.distance_index],draws[:,joint.time_index])
        residual = (em.data.magnitude.to_numpy()-np.median(prediction,axis=0))/em.sigma
        axes[1].scatter(em.data.time_days,residual,s=15)
        axes[1].axhline(0,color='k',lw=.7)
        axes[0].invert_yaxis()
        axes[0].legend(ncol=5)
        axes[0].set(xlabel='Observer-frame days since merger',ylabel='AB magnitude (observed)')
        axes[1].set(xlabel='Observer-frame days since merger',ylabel='Residual / total sigma')
        fig.suptitle('Early optical photosphere: model draws, not a converged credible band')
        fig.tight_layout()
        fig.savefig(outdir/'joint_lightcurve.png',dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['fast','full'],default='fast')
    parser.add_argument('--sample',action='store_true')
    parser.add_argument('--steps',type=int)
    parser.add_argument('--burn',type=int)
    parser.add_argument('--seed',type=int,default=170818)
    parser.add_argument('--outdir',type=Path)
    args = parser.parse_args()
    out = args.outdir or Path('results')/('gw170817_joint_'+args.mode)
    out.mkdir(parents=True,exist_ok=True)
    gw, like, metadata = build(out,cache_dir=Path('results/gw170817/data'),**(FAST_CONFIG if args.mode=='fast' else {}))
    if args.mode=='fast':
        priors = fast_priors()
        center = recover(gw,like,out)
    else:
        from gw170817_pe import priors as full_priors
        from hyperwave.detectors.lvk import GW
        from hyperwave.likelihoods import GWLikelihoods
        priors = full_priors()
        gw = GW(gw.noise,approximant='IMRPhenomPv2_NRTidal',reference_frequency=50,parameters=list(priors),static_parameters={},n_jobs=1)
        psd = np.array([ifo.power_spectral_density_array[gw.mask] for ifo in gw.noise.ifos])
        like = GWLikelihoods(np.array([gw.detector_data_fd(i) for i in range(3)]),gw.frequency_array(),IFOS,psd,gw,ddims=False,nsegs=4,gpu=False,cpu_cores=1)
        point = dict(chirp_mass=1.1975,mass_ratio=.85,psi=.2,phase=2.5,ra=3.44616,a_1=.005,a_2=.005,cos_theta_jn=np.cos(.4),cos_tilt_1=.9,cos_tilt_2=.9,phi_12=.2,phi_jl=.2,geocent_time=TC,lambda_1=400,lambda_2=400,dec=-.40808,luminosity_distance=45.)
        center = np.array([point[n] for n in priors])
        metadata.update(approximant='IMRPhenomPv2_NRTidal',parameters=list(priors),fixed={},bounds={n:[p.minimum,p.maximum] for n,p in priors.items()})
    metadata.update(likelihood='Gaussian GW + independent Gaussian magnitude errors',mode=args.mode)
    (out/'gw_manifest.json').write_text(json.dumps(metadata,indent=2))
    run_joint(like.gaussian,priors,center,out,burn=args.burn if args.burn is not None else (10 if args.mode=='fast' else 1000),
              steps=args.steps if args.steps is not None else (40 if args.mode=='fast' else 1000),seed=args.seed,
              walkers=24 if args.mode=='fast' else 64,temperatures=2 if args.mode=='fast' else 4,sample=args.sample)
    print(WARNING)
    print(f'Outputs: {out.resolve()}')


if __name__=='__main__':
    main()
