# Dataset A preprocessing: save only the CSVs to hand over and processing statistics
import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import unicodedata

# Unify case and whitespace of names and tags
def normalize(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())

# Save as CSV
def write_csv(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)

# Report
def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


# Check that the folder is empty
def prepare_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError('Output folder must be empty: ' + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path

# Split the same user’s history into train, validation, and test
def export_interactions(rows, output, *, seed, min_history,
                        val_fraction, test_fraction, artist_order=None):
    if min_history < 3 or not (0 < val_fraction < 1 and 0 < test_fraction < 1
                                and val_fraction + test_fraction < 1):
        raise ValueError('Need min-history >= 3 and positive split fractions summing to < 1')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    histories = collections.defaultdict(list)
    seen = set()
    for user, artist, plays in rows:
        if (user, artist) in seen or plays <= 0:
            raise ValueError('Input interactions must be positive and unique')
        seen.add((user, artist))
        histories[user].append((artist, plays))
    eligible = sorted(u for u, history in histories.items() if len(history) >= min_history)
    if not eligible:
        raise ValueError('No eligible users: reduce min-history or increase sample size')
    artists = list(artist_order) if artist_order is not None else sorted(
        {a for u in eligible for a, _ in histories[u]})
    artist_index = {a: i for i, a in enumerate(artists)}
    if len(artists) != len(artist_index):
        raise ValueError('Duplicate artist vocabulary')
    user_index = {u: i for i, u in enumerate(eligible)}
    parts = {'train': [], 'validation': [], 'test': []}
    for user in eligible:
        history = sorted(histories[user])
        # Use the same seed per user so a rerun gives the same split
        digest = hashlib.sha256((str(seed) + '|' + user).encode('utf-8')).digest()
        rng = random.Random(int.from_bytes(digest[:8], 'big'))
        rng.shuffle(history)
        n = len(history)
        # At least one item each for validation and test
        ntest = max(1, int(n * test_fraction))
        nval = max(1, int(n * val_fraction))
        if ntest + nval >= n:
            raise ValueError('Split leaves no training artist for a user')
        chunks = {'test': history[:ntest], 'validation': history[ntest:ntest+nval],
                  'train': history[ntest+nval:]}
        for part, values in chunks.items():
            parts[part].extend((user_index[user], artist_index[a], user, a, p,
                                math.log1p(p)) for a, p in values)
    shape = (len(eligible), len(artists))
    for part, values in parts.items():
        write_csv(output / (part + '.csv'),
                  ['user_index', 'artist_index', 'user_id', 'artist_id', 'plays', 'log1p_plays'],
                  sorted(values))
    popularity = collections.Counter(r[1] for r in parts['train'])
    train_artists = set(popularity)
    return {'users_before_history_filter': len(histories), 'eligible_users': len(eligible),
            'excluded_short_history_users': len(histories) - len(eligible),
            'excluded_short_history_rows': sum(len(v) for u, v in histories.items()
                                               if u not in user_index),
            'artist_vocabulary_size': len(artists), 'matrix_shape': list(shape),
            'split_rows': {k: len(v) for k, v in parts.items()},
            'heldout_rows_without_train_artist': {
                k: sum(r[1] not in train_artists for r in parts[k])
                for k in ['validation', 'test']},
            'min_history': min_history, 'seed': seed,
            'validation_fraction': val_fraction, 'test_fraction': test_fraction,
            'split_policy': 'per-user random artist holdout; floor with minimum 1 per heldout split'}

# Use the same split criteria for both datasets
def split_options(parser):
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-history', type=int, default=5)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--test-fraction', type=float, default=0.2)

# Convert the given split settings into function arguments
def split_kwargs(args):
    return dict(seed=args.seed, min_history=args.min_history,
                val_fraction=args.val_fraction, test_fraction=args.test_fraction)

import zipfile

# Read only the needed files without extracting the archive
def read_zip(z, name, encoding='utf-8-sig'):
    with z.open(name) as raw:
        yield from csv.DictReader(io.TextIOWrapper(raw, encoding=encoding, newline=''),
                                  delimiter='\t', quoting=csv.QUOTE_NONE)

def main():
    parser = argparse.ArgumentParser(description='Last.fm 2K preprocessing')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tag-encoding', default='latin-1',
                        help='character encoding used to read tags.dat')
    parser.add_argument('--min-tags', type=int, default=2,
                        help='minimum number of tags required for content-based recommendation')
    split_options(parser)
    args = parser.parse_args()
    if args.min_tags < 1:
        parser.error('--min-tags must be positive')
    output = prepare_output(args.output)
    stats = collections.Counter()
    with zipfile.ZipFile(args.input) as z:
        artists = {}
        for row in read_zip(z, 'artists.dat'):
            aid = str(int(row['id']))
            if aid in artists:
                raise ValueError('Duplicate artist ID: ' + aid)
            artists[aid] = row
        tags = {}
        for row in read_zip(z, 'tags.dat', args.tag_encoding):
            tid = str(int(row['tagID']))
            if tid in tags:
                raise ValueError('Duplicate tag ID: ' + tid)
            tags[tid] = normalize(row['tagValue'])
        plays = {}
        for row in read_zip(z, 'user_artists.dat'):
            stats['raw_listening_rows'] += 1
            try:
                uid, aid, count = str(int(row['userID'])), str(int(row['artistID'])), int(row['weight'])
                if int(uid) <= 0 or int(aid) <= 0 or count <= 0:
                    raise ValueError()
            except (ValueError, TypeError):
                stats['invalid_listening_rows'] += 1
                continue
            if aid not in artists:
                stats['unknown_listening_artist_rows'] += 1
                continue
            key = uid, aid
            if key in plays:
                stats['duplicate_listening_rows'] += 1
            # If cumulative play counts are duplicated, keep the larger value instead of adding
            plays[key] = max(plays.get(key, 0), count)
        # When cleaning tags, check whether the user is in the listening records
        known_users = {u for u, _ in plays}
        assignments = set()
        tag_counts = collections.Counter()
        for row in read_zip(z, 'user_taggedartists.dat'):
            stats['raw_tag_assignment_rows'] += 1
            uid, aid, tid = row['userID'], row['artistID'], row['tagID']
            if aid not in artists:
                stats['unknown_tagging_artist_rows'] += 1
                continue
            if uid not in known_users or tid not in tags or not tags[tid]:
                stats['invalid_tag_assignment_rows'] += 1
                continue
            key = uid, aid, tags[tid]
            if key in assignments:
                stats['duplicate_normalized_tag_assignments'] += 1
                continue
            assignments.add(key)
            # Count the users who assigned each tag, per artist
            tag_counts[aid, tags[tid]] += 1
    artist_order = sorted(artists, key=int)
    artist_index = {a: i for i, a in enumerate(artist_order)}
    tag_sets = collections.defaultdict(set)
    for aid, tag in tag_counts:
        tag_sets[aid].add(tag)
    # Use only artists with enough tags for content-based recommendation
    content_artists = {a for a in artists if len(tag_sets[a]) >= args.min_tags}
    # Count the usable tags for the report
    vocabulary = sorted({tag for a, tag in tag_counts if a in content_artists})
    if not vocabulary:
        raise ValueError('No eligible artist tags; lower --min-tags')
    write_csv(output / 'artist_tag_counts.csv', ['artist_id', 'tag', 'distinct_user_count'],
              ((a, t, count) for (a, t), count in sorted(tag_counts.items())))
    write_csv(output / 'artists.csv',
              ['artist_index', 'artist_id', 'name', 'normalized_name', 'url', 'distinct_tags', 'content_eligible'],
              ((artist_index[a], a, artists[a]['name'], normalize(artists[a]['name']),
                artists[a].get('url', ''), len(tag_sets[a]), int(a in content_artists)) for a in artist_order))
    report = {'dataset': 'HetRec 2011 Last.fm 2K', 'input': str(args.input.resolve()),
              'audit': dict(stats), 'clean_listening_rows': len(plays),
              'catalog_artists': len(artists), 'artists_without_tags': sum(not tag_sets[a] for a in artists),
              'content_eligible_artists': len(content_artists), 'min_tags': args.min_tags,
              'tag_vocabulary_size': len(vocabulary), 'tag_encoding': args.tag_encoding,
              'tfidf_policy': 'Not generated here; hand off artist_tag_counts.csv for the content model',
              'metadata_policy': 'Static aggregated catalog tags from all supplied tag assignments; no per-user tagging features. Not a temporal or strict user-independent metadata evaluation.',
              'split': export_interactions(((u, a, p) for (u, a), p in plays.items()),
                                           output / 'cf', artist_order=artist_order, **split_kwargs(args))}
    write_json(output / 'report.json', report)
    print('A preprocessing finished:', output.resolve(), flush=True)
    print(json.dumps({'clean_rows': len(plays), 'eligible_users': report['split']['eligible_users'],
                      'content_artists': len(content_artists)}, indent=2))

if __name__ == '__main__':
    main()
