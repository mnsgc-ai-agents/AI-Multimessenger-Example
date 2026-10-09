# AT2017gfo observational photometry

`AT2017gfo.dat` is an unmodified copy of the NMMA example photometry file at
commit `b62084fa0f59faae0b9bccc4ac63fe46a232e836`. The source URL, checksum and
interpretation are recorded in `provenance.json`. This small file is bundled
so that the light-curve tutorial works offline without installing NMMA.

Columns: UTC ISO timestamp, native filter name, supplied magnitude, magnitude
uncertainty. There are 141 rows in nine filters. Three rows have infinite
uncertainties, NMMA's convention for upper limits. They are shown separately
and are excluded from the classroom Gaussian decline fits.

The tutorial retains the supplied magnitudes without additional extinction,
zero-point, K-correction or bolometric conversion. It uses the explicit
`AT2017gfo.dat` variant, not the different `AT2017gfo_corrected.dat` variant.
Do not treat the plots as homogenized intrinsic luminosities or infer
cross-system colors without checking the relevant calibration conventions.

References:

- [Pinned NMMA data](https://github.com/nuclear-multimessenger-astronomy/nmma/blob/b62084fa0f59faae0b9bccc4ac63fe46a232e836/example_files/lightcurves/AT2017gfo.dat)
- [NMMA upper-limit convention](https://github.com/nuclear-multimessenger-astronomy/nmma/blob/b62084fa0f59faae0b9bccc4ac63fe46a232e836/nmma/em/em_likelihood.py)
- [Pang et al., NMMA framework](https://arxiv.org/abs/2205.08513)
- [Villar et al., combined AT2017gfo light curves](https://arxiv.org/abs/1710.11576)

No NMMA code or trained model is imported by the classroom light-curve script.
Cite the data source and the
relevant observational papers when extending this teaching example.
