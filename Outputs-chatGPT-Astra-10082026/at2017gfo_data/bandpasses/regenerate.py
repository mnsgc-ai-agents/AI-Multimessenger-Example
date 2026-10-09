"""Optional regeneration of bundled bandpasses; ordinary tutorial runs do not need this.

Requires sncosmo==2.13.1 and dust_extinction==1.7 plus network access.
Run this file explicitly to replace the tables and provenance hashes.
"""
import sncosmo, numpy as np, json, hashlib
from pathlib import Path
from dust_extinction.parameter_averages import F99
import astropy.units as u
p=Path(__file__).resolve().parent
files={}
for b in 'grizy':
 bp=sncosmo.get_bandpass('ps1::'+b)
 wave=np.linspace(bp.minwave(),bp.maxwave(),160)
 values=np.column_stack((wave,bp(wave),F99(Rv=3.1)(wave*u.AA)*3.1))
 path=p/(b+'.txt')
 np.savetxt(path,values,header='wavelength_AA transmission A_lambda_per_E_BV_F99_Rv3.1')
 files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
(p/'provenance.json').write_text(json.dumps(dict(source='SNCosmo built-in PS1 bandpasses (Tonry et al. 2012)',reference='https://sncosmo.readthedocs.io/en/stable/bandpass-list.html',sncosmo_version=sncosmo.__version__,dust_extinction_version='1.7',extinction='Fitzpatrick (1999), R_V=3.1; stored A_lambda/E(B-V)',sampling='160 uniformly spaced wavelengths per filter; linear interpolation of native throughput',sha256=files),indent=2))
