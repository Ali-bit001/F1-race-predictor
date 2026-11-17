# M09 car-form control (Task A) - pre-statement, written before the runs

Label: DIAGNOSTIC, POST HOC, NO DECISION WEIGHT. Not a declared component; no attribution rows. Dev only (2021-2023).
Motivation: the skeptic found pace_hist_x has within-race rank correlation 0.963 with con_ew32 (0.969 at half-life 50) and q1off_x
0.865-0.923, so the +0.016 rho of latent_111 + pace_hist_x may be car-form dispersion rather than race pace.

Control column `form_x`: decayed mean (half-life 50, earlier races only, same lineage mapping, same min_prior_obs, prior 0, scaled by
the SD over seasons <= 2020) of the car's field-relative finishing position (the quantity behind con_ew*: car mean of the normalised
finishing position y, per lineage and race; centred by subtracting the race mean of y).
Rungs (latent_111, lam_x tuned over [0.3, 1, 3, 10] at refit_every=4, final at refit_every=1):
  C1 = reference + form_x ; C2 = reference + form_x + pace_hist_x. Budget: 15 (M09) + 8 = 23 <= 30.
Paired gains (10,000 race resamples, seed 0, driverId tie-break): C1 vs ref, hist vs C1, C2 vs C1.

Decision rule, fixed now:
* If the paired interval of (hist vs C1) on rho includes 0, the gain of hist over the reference is credited to car form, not race pace.
* Race pace adds value beyond car form only if the interval of (C2 vs C1) on rho excludes 0 (positive side).
* Nothing here changes any M09 verdict or enters the attribution table.
