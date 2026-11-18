# Plan-3 freeze (R14)

Date: 2026-11-18

SHA-256 of frozen_config.yaml: `24ef86b6271b9df4d12a4d08e5fff587d2f2d5bb877e79213c82dad1e19abe00`
(hash of the file with CRLF turned into LF, `pitlane.plan3.final.sha256_file`)

- Last race included in the data: 2026 round 20 (race_id 271). The fresh holdout is every race run after it.
- Components in the final pre-race model (H2): M11_reverse, M09_car_pace_hist; simulator S3 re-fitted on dev; decoder D0.
  H1 is the plan-2 frozen H1 model unchanged.
- Dev run that produced the final numbers: `20261118-0705_p3_final_model_dev` (methodologies/P3_final/results/).
- Git: committed in the commit tagged **`plan3-freeze`** (annotated, dated 2026-11-18), before any session of 2026 R21 (Las Vegas,
  FP1 2026-11-20 00:30 UTC). The run folders carry code stamp `8063606+dirty-6f6240706de9`: the uncommitted tree that is this commit.
  No file in this folder named `frozen_config.yaml` or `HOLDOUT_PREREG.md` may be edited after the tag (`tests/test_plan3_final.py`
  checks the config hash above against the file).
- Holdout tests (what is scored, references, T1-T3, minimum 20 races): `HOLDOUT_PREREG.md`,
  SHA-256 `f6031e581389f79b5dd3f46520a7ef2bc7985466d85d16654cfdbdf1b83b1acd`.
  `prestatement.md` SHA-256 `9dd86fda9a3baf481f2cd992a4e1fefa367aa9a077425a18dfbfae495f4a9adf`.
- Data at the freeze (git-ignored, SHA-256 of the files as they stand with data through 2026 R20; holdout rows are appended later,
  and rows with race_id <= 271 must stay equal):
  - `data/processed/entries2.parquet` `310293a78909b5a557ec0f47634c0a321f006bae7910f57bc466a284d2e28065`
  - `data/processed/entries.parquet` `ee0977bd13cb37eeba3806a3a28bdfd8583ec2b2aad7d49131b8256c03d9ff22`
  - `data/interim/plan3/laps.parquet` `7b61cc4f39a803e35a06953de31d40a632cb65b55800c90fc3f5a8ef6868f5f2`
  - `data/interim/plan3/pits.parquet` `e8792396200fe70aa07cc9353f7d67f421033f4cd2c4b3082c23dbc64d8750d6`
  - `data/external/lineage.csv` `857f228214f4da8596557af977480ad85f8c88e80829f89de9fea461210625f8`
  - `data/external/circuits.csv` `7ea7e667492d016bc82070e559eaf0c9416300ef1d1398b8b0b889ba4deafcf9`
  - `data/external/power_units.csv` `fd3701b99dc55979b63dec295e3c8f93a08eef06f4d1b6f06240663268b8cfba`
  - `data/external/scheduled_laps.csv` `1b0e3b72b342517717a2f07ca4eb70956eee0be96281a8461ef683e76edee093`
- Packages: Python 3.10.11, pandas 2.3.3, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, lightgbm 4.7.0.
- Hyper-parameters were tuned on 2021-2023 only. The re-fit on data through 2026 round 20 happens at prediction time
  (every race is predicted from a model fitted on all earlier races); nothing is re-tuned on 2024-2026.
- Predict a race: `python methodologies/P3_final/predict.py --season S --round R` (refuses to run if the hash above does not match).
