# data/

The datasets and the preprocessed ZIP are **not included** in this repository because of file size (the preprocessed archive is about 95 MB when extracted) and the datasets' own terms of use.

## 1. Download the original data

| Dataset | Source |
|---|---|
| Dataset A: HetRec 2011 Last.fm 2K | https://grouplens.org/datasets/hetrec-2011/ (`hetrec2011-lastfm-2k.zip`) |
| Dataset B: Last.fm 360K | https://doi.org/10.5281/zenodo.6090214 (`lastfm-dataset-360K.tar.gz`) |

## 2. Run the preprocessing scripts

Run A first, then B. Pass A's output folder to `--a-output` of B. Output folders must be empty.

```bash
python ../../01_preprocessing/preprocess_lastfm_2k_2.py \
  --input hetrec2011-lastfm-2k.zip --output data_a

python ../../01_preprocessing/preprocess_lastfm_360k_1.py \
  --input lastfm-dataset-360K.tar.gz --output data_b \
  --a-output data_a --sample-users 10000
```

## 3. Build the preprocessed ZIP

Bundle the results into one ZIP and place it in this folder (`notebooks/data/`). The notebooks find the ZIP automatically. The ZIP must contain:

```
Lastfm.2k-a/
  train.csv, validation.csv, test.csv      (from data_a/cf/)
  artists.csv, artist_tag_counts.csv       (from data_a/)
Lastfm.360k-b/
  train.csv, validation.csv, test.csv      (from data_b/cf/)
  artists.csv, content_candidate_artists.csv
  content_external_candidates/             (train.csv, validation.csv, test.csv)
```

## 4. File format notes

- CSV files are UTF-8 with BOM; read them with `encoding="utf-8-sig"`.
- Read `user_id` and `artist_id` as strings so that leading zeros of B's SHA-1 user IDs are preserved.
- `artist_index` is not a shared numbering between A and B; always join on `user_id` and `artist_id`.
