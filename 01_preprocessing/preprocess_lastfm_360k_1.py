# Dataset B preprocessing: save only the CSVs to hand over and processing statistics.
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata

# Unify case and whitespace of names and tags
def normalize(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())

# Save as CSV (UTF-8 BOM so that non-ASCII text is not garbled)
def write_csv(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)

# For the report
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

import tarfile

UID = re.compile(rb'[0-9a-fA-F]{40}')
MBID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
ZERO_MBID = '00000000-0000-0000-0000-000000000000'

# Check that the artist ID has a valid format
def mbid_or_empty(value):
    value = value.strip().lower()
    return value if MBID.fullmatch(value) and value != ZERO_MBID else ''

# Read the compressed file line by line
def listening_lines(path):
    with tarfile.open(path, 'r|gz') as tar:
        found = False
        for member in tar:
            if member.isfile() and Path(member.name).name == 'usersha1-artmbid-artname-plays.tsv':
                if found:
                    raise ValueError('Multiple listening files in archive')
                found = True
                with tar.extractfile(member) as stream:
                    yield from stream
        if not found:
            raise ValueError('Listening TSV not found')

# Check the ID, name, and play count in one line of B
def parse_line(raw):
    try:
        parts = raw.decode('utf-8').rstrip('\r\n').split('\t')
    except UnicodeDecodeError:
        return None, 'invalid_utf8'
    if len(parts) != 4:
        return None, 'wrong_column_count'
    uid, raw_mid, name, count = parts
    if not UID.fullmatch(uid.encode('utf-8')):
        return None, 'invalid_user_id'
    uid = uid.lower()
    name = ' '.join(unicodedata.normalize('NFKC', name).split())
    if not name:
        return None, 'blank_artist_name'
    try:
        count = int(count)
        if count <= 0:
            raise ValueError()
    except ValueError:
        return None, 'invalid_or_nonpositive_plays'
    # If the MBID is missing or malformed, make a temporary key from the name
    mid = mbid_or_empty(raw_mid)
    normalized = normalize(name)
    key = 'mbid:' + mid if mid else 'name:' + normalized
    return (uid, key, name, normalized, count, mid, raw_mid), None

def main():
    parser = argparse.ArgumentParser(description='Last.fm 360K sampling and preprocessing')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-users', type=int, default=10000,
                        help='number of users to select at random')
    parser.add_argument('--a-output', type=Path,
                        help='output folder of A’s preprocessing; required for the content external evaluation')
    split_options(parser)
    args = parser.parse_args()
    if args.sample_users < 1:
        parser.error('--sample-users must be positive')
    output = prepare_output(args.output)
    a_rows = []
    a_by_name = collections.defaultdict(list)
    if args.a_output:
        with (args.a_output / 'artists.csv').open(encoding='utf-8-sig', newline='') as stream:
            a_rows = list(csv.DictReader(stream))
        for row in a_rows:
            a_by_name[normalize(row['name'])].append(row)
    users = set()
    name_keys = collections.defaultdict(set)
    stats = collections.Counter()
    print('1/2 Checking all records and identifying users and artist names.', flush=True)
    for raw in listening_lines(args.input):
        stats['raw_rows'] += 1
        row, error = parse_line(raw)
        if error:
            stats[error] += 1
            continue
        uid, key, name, normalized, count, mid, raw_mid = row
        users.add(uid)
        stats['valid_rows_before_duplicate_aggregation'] += 1
        if not raw_mid.strip():
            stats['missing_mbid_rows'] += 1
        elif not mid:
            stats['invalid_mbid_after_strip_rows'] += 1
        elif raw_mid != mid:
            stats['mbid_normalized_rows'] += 1
        if normalized in a_by_name:
            name_keys[normalized].add(key)
        if stats['raw_rows'] % 3000000 == 0:
            print('Scanned', stats['raw_rows'], 'rows', flush=True)
    # Draw users at random, not rows
    selected = set(random.Random(args.seed).sample(sorted(users), min(args.sample_users, len(users))))
    if not selected:
        raise ValueError('No valid users')
    # Link only when a name corresponds to exactly one artist on both sides
    proposed = collections.defaultdict(set)
    for name, keys in name_keys.items():
        if len(a_by_name[name]) == 1 and len(keys) == 1:
            proposed[next(iter(keys))].add(a_by_name[name][0]['artist_id'])
    b_to_a = {key: next(iter(ids)) for key, ids in proposed.items() if len(ids) == 1}
    a_lookup = {r['artist_id']: r for r in a_rows}
    write_csv(output / 'artist_links_candidates.csv',
              ['b_artist_key', 'a_artist_id', 'a_name', 'a_content_eligible', 'status'],
              ((key, aid, a_lookup[aid]['name'], a_lookup[aid]['content_eligible'],
                'conservative_name_candidate_not_identity_verified') for key, aid in sorted(b_to_a.items())))
    print('2/2 Collecting the selected users’ records and cleaning duplicates.', flush=True)
    plays = {}
    artist_names = {}
    sample_stats = collections.Counter()
    for raw in listening_lines(args.input):
        uid_bytes = raw.partition(b'\t')[0]
        if uid_bytes.decode('ascii', errors='replace').lower() not in selected:
            continue
        row, error = parse_line(raw)
        if error:
            sample_stats['invalid_rows_for_selected_users'] += 1
            continue
        uid, key, name, normalized, count, mid, raw_mid = row
        sample_stats['valid_rows_before_duplicate_aggregation'] += 1
        pair = uid, key
        if pair in plays:
            sample_stats['duplicate_user_artist_rows'] += 1
        # If the same user and artist repeat, use the largest play count.
        plays[pair] = max(plays.get(pair, 0), count)
        artist_names[key] = min(artist_names.get(key, name), name)
    write_csv(output / 'artists.csv', ['artist_id', 'name', 'normalized_name'],
              ((a, n, normalize(n)) for a, n in sorted(artist_names.items())))
    report = {'dataset': 'Last.fm 360K', 'input': str(args.input.resolve()),
              'full_data_audit': dict(stats), 'valid_users_before_sampling': len(users),
              'requested_sample_users': args.sample_users, 'sampled_users': len(selected),
              'sample_audit': dict(sample_stats), 'sample_clean_rows': len(plays),
              'sample_artists': len(artist_names), 'seed': args.seed,
              'duplicate_policy': 'max cumulative plays per user/artist key; never sum unverified snapshots',
              'artist_key_policy': 'valid stripped lowercase MBID; otherwise NFKC/casefold/whitespace-normalized name',
              'profile_policy': 'Profile attributes unused; no filtering by age, gender or country',
              'cf': export_interactions(((u, a, p) for (u, a), p in plays.items()),
                                         output / 'cf', **split_kwargs(args))}
    if a_rows:
        mapped_a = set(b_to_a.values())
        # Content evaluation only for linked artists that also have tags
        eligible_a = {a for a in mapped_a if a_lookup[a]['content_eligible'] == '1'}
        external = {}
        linked_count = 0
        for (user, key), count in plays.items():
            aid = b_to_a.get(key)
            if aid is not None:
                linked_count += 1
            if aid in eligible_a:
                pair = user, aid
                external[pair] = max(external.get(pair, 0), count)
        # B’s content data also uses the same artist order as A
        a_order = [r['artist_id'] for r in sorted(a_rows, key=lambda r: int(r['artist_index']))]
        write_csv(output / 'content_candidate_artists.csv',
                  ['artist_index', 'a_artist_id', 'recommendation_candidate'],
                  ((i, aid, int(aid in eligible_a)) for i, aid in enumerate(a_order)))
        candidate_info = {'status': 'preliminary name-based evaluation; review identities before final claims',
                          'a_catalog_artists': len(a_rows), 'mapped_a_artists': len(mapped_a),
                          'a_catalog_matching_rate': len(mapped_a)/len(a_rows),
                          'mapped_a_content_eligible_artists': len(eligible_a),
                          'sample_linked_rows': linked_count,
                          'sample_linked_row_rate': linked_count/len(plays),
                          'sample_unlinked_rows': len(plays)-linked_count,
                          'sample_content_usable_rows': len(external),
                          'sample_content_excluded_rows': len(plays)-len(external)}
        # Check that there are users with enough common-artist history
        external_histories = collections.Counter(u for u, _ in external)
        if any(n >= args.min_history for n in external_histories.values()):
            candidate_info['split'] = export_interactions(
                ((u, a, p) for (u, a), p in external.items()),
                output / 'content_external_candidates', artist_order=a_order, **split_kwargs(args))
        else:
            candidate_info['split'] = {'status': 'not_enough_common_history_for_any_user'}
        report['artist_linking'] = candidate_info
    write_json(output / 'report.json', report)
    print('B preprocessing finished:', output.resolve(), flush=True)
    print(json.dumps({'sample_users': len(selected), 'sample_clean_rows': len(plays),
                      'cf_eligible_users': report['cf']['eligible_users']}, indent=2))

if __name__ == '__main__':
    main()
