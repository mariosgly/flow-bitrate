"""Inventory, sharded decode audits, and one global manifest merge. HPC only."""

import argparse
import csv
import fcntl
import hashlib
import json
import re
import sqlite3
import tarfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf

import make_postfilter_manifest as old
from postfilter_v2_common import ShardJob, atomic_json, audio_exclusion, digest, inventory, read_config

WAVCAPS = {
    'FreeSound': ('fsd_final.json', 'wavcaps_freesound'),
    'BBC_Sound_Effects': ('bbc_final.json', 'wavcaps_bbc'),
    'SoundBible': ('sb_final.json', 'wavcaps_soundbible'),
    'AudioSet_SL': ('as_final.json', 'wavcaps_audioset'),
}


def subset(path):
    return next((part.replace('_', '-') for part in Path(path).parts
                 if part.replace('_', '-').startswith(('train-clean-', 'train-other-', 'dev-', 'test-'))), '')


def audioset_labels(config, clips):
    directory = Path(config['audioset_metadata'])
    nodes = json.loads((directory / 'ontology.json').read_text())
    tree = {node['id']: node for node in nodes}
    names = {node['id']: node['name'] for node in nodes}
    with (directory / 'class_labels_indices.csv').open(newline='') as handle:
        names.update({row['mid']: row['display_name'] for row in csv.DictReader(handle)})
    with (directory / 'mid_to_display_name.tsv').open() as handle:
        names.update(dict(line.rstrip('\n').split('\t', 1) for line in handle if '\t' in line))

    def descendants(root):
        found, pending = set(), [root]
        while pending:
            node = pending.pop()
            if node not in found:
                found.add(node)
                pending.extend(tree.get(node, {}).get('child_ids', []))
        return found

    speech = descendants('/m/09x0r')
    music = descendants('/m/04rlf')
    for node in nodes:
        if node['name'] == 'Singing':
            music |= descendants(node['id'])

    def classify(mids):
        labels = set()
        for mid in mids:
            if mid in speech:
                labels.add('speech')
            elif mid in music:
                labels.add('music')
            elif mid in names:
                # Channel/background descriptors alone don't establish another source.
                if names[mid] not in {'Music', 'Human voice', 'Human sounds', 'Silence',
                                     'Noise', 'Inside, small room', 'Inside, large room or hall',
                                     'Inside, public space', 'Outside, urban or manmade',
                                     'Outside, rural or natural'}:
                    labels.add('other_sound')
        return sorted(labels)

    weak = defaultdict(list)
    for filename in ('balanced_train_segments.csv', 'unbalanced_train_segments.csv', 'eval_segments.csv'):
        path = Path(config['audioset_segments']) / filename
        with path.open(newline='') as handle:
            for row in csv.reader((line for line in handle if not line.startswith('#')), skipinitialspace=True):
                if row and row[0] in clips:
                    weak[row[0]].append(dict(start=float(row[1]), end=float(row[2]),
                                             mids=row[3].split(','), official_split='test' if filename == 'eval_segments.csv' else 'train'))
    strong, strong_starts = defaultdict(list), defaultdict(set)
    for filename in ('audioset_train_strong.tsv', 'audioset_eval_strong.tsv'):
        with (directory / filename).open(newline='') as handle:
            for row in csv.reader(handle, delimiter='\t'):
                if len(row) != 4:
                    continue
                video, _, start = row[0].rpartition('_')
                if video in clips and start.isdigit():
                    strong_starts[video].add(int(start))
                    strong[video, int(start)].append(dict(start=float(row[1]), end=float(row[2]), mid=row[3],
                                                         official_split='test' if 'eval' in filename else 'train'))
    result = {}
    for video in clips:
        entries = weak.get(video, [])
        starts = {round(entry['start'] * 1000) for entry in entries}
        starts.update(strong_starts[video])
        official_test = (any(entry['official_split'] == 'test' for entry in entries)
                         or any(event['official_split'] == 'test' for start in strong_starts[video]
                                for event in strong[video, start]))
        # WavCaps filenames identify a video, not an arbitrary segment. Don't guess if ambiguous.
        if len(starts) != 1:
            result[video] = dict(rule_labels=[], label_source='caption_model', segment_match='missing_or_ambiguous',
                                 official_split='test' if official_test else 'unknown')
            continue
        start = next(iter(starts))
        events = strong.get((video, start), [])
        mids = sorted({event['mid'] for event in events} if events else {mid for entry in entries for mid in entry['mids']})
        result[video] = dict(rule_labels=classify(mids),
                             label_source='audioset_strong' if events else 'audioset_weak',
                             event_labels=[dict(mid=mid, name=names.get(mid, mid)) for mid in mids],
                             event_intervals=events, source_start_seconds=start / 1000,
                             official_split='test' if official_test else 'train')
    return result


def build_inventory(config, base, output):
    lock = (output / 'inventory.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'inventory.json').exists():
        info = json.loads((output / 'inventory.json').read_text())
        if info['config_hash'] != digest(config):
            raise ValueError('Inventory settings changed; use a new output folder')
        print('Inventory already published. Reusing it.', flush=True)
        return
    extra = Path(config['libritts_extra'])
    for name in ('dev_clean', 'dev_other', 'test_clean', 'test_other',
                 'libritts_r_failed_speech_restoration_examples'):
        if not (extra / f'.{name}.complete.json').is_file():
            raise RuntimeError(f'Extraction is incomplete: {name}. Finish slurm/libritts_extract.slurm first.')
    old.SCAN_DIRECTORY = output / 'scans'
    old.SCAN_DIRECTORY.mkdir(exist_ok=True)
    old.PROGRESS_PATH = output / 'inventory.progress.json'
    previous = Path(config['previous_manifest'])
    # Copy small scan indexes, never audio. Old caches are opened read-only.
    for source in previous.with_suffix('.scans').glob('*.sqlite'):
        target = old.SCAN_DIRECTORY / source.name
        if not target.exists():
            temporary = target.with_suffix('.part')
            with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as inp, sqlite3.connect(temporary) as out:
                inp.backup(out)
            temporary.replace(target)
    records, exclusions = old.collect(base)
    extra = Path(config['libritts_extra'])
    if not extra.is_dir():
        raise FileNotFoundError(f'Run scripts/extract_libritts_r.py first: {extra}')
    extra_subsets = set()
    for path in old.audio_files(extra):
        role = subset(path)
        if role in {'dev-clean', 'dev-other', 'test-clean', 'test-other'}:
            extra_subsets.add(role)
            speaker = path.stem.split('_', 1)[0]
            records.append(old.record([path], 'libritts_r', 'speech', f'speech:{speaker}', path.stem,
                                      'val' if role.startswith('dev-') else 'test'))
    if extra_subsets != {'dev-clean', 'dev-other', 'test-clean', 'test-other'}:
        raise ValueError(f'Need all four extracted LibriTTS-R dev/test subsets; found {extra_subsets}')

    failed = set()
    archive = extra.parent / 'libritts_r_failed_speech_restoration_examples.tar.gz'
    with tarfile.open(archive) as source:
        for member in source:
            if member.isfile():
                text = source.extractfile(member).read().decode('utf-8', errors='replace')
                failed.update(re.findall(r'\b\d+_\d+_\d+_\d+\b', text))
    root = Path(base['wavcaps'])
    metadata, metadata_hashes = {}, {}
    for source, (filename, dataset) in WAVCAPS.items():
        path = root / 'json_files' / source / filename
        value = json.loads(path.read_text())
        metadata[dataset] = {Path(str(item['id'])).stem: item for item in value['data']}
        metadata_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    videos = {}
    for clip in metadata['wavcaps_audioset']:
        candidate = clip[1:] if len(clip) == 12 and clip.startswith('Y') else clip
        if re.fullmatch(r'[A-Za-z0-9_-]{11}', candidate):
            videos[clip] = candidate
    annotations = audioset_labels(config, set(videos.values()))
    for source, (_, dataset) in WAVCAPS.items():
        if source == 'FreeSound':
            continue
        normalized = re.sub('[^a-z0-9]', '', source.lower())
        folders = [path for path in (root / 'data').iterdir() if path.is_dir()
                   and re.sub('[^a-z0-9]', '', path.name.lower()) in {normalized, normalized + 'flac', normalized + 'wav'}]
        if not folders:
            raise FileNotFoundError(f'No {source} audio folder under {root / "data"}; expected source name with optional _flac/_wav')
        copies = defaultdict(list)
        for folder in folders:
            for path in old.audio_files(folder):
                copies[path.stem].append(str(path))
        matched = 0
        for clip, item in metadata[dataset].items():
            paths = sorted(copies.get(clip, []), key=lambda path: (Path(path).suffix.lower() != '.flac', path))
            if not paths:
                exclusions.append(dict(path=f'{dataset}:{clip}', reason='missing_audio'))
                continue
            split = 'pool'
            if dataset == 'wavcaps_audioset':
                if clip not in videos:
                    exclusions.append(dict(path=paths[0], reason='invalid_youtube_id'))
                    continue
                video = videos[clip]
                group = f'youtube:{video}'
                if annotations[video]['official_split'] == 'test':
                    split = 'test'
            elif dataset == 'wavcaps_soundbible' and item.get('author'):
                group = f'soundbible:{str(item["author"]).strip().casefold()}'
            else:
                # BBC metadata shown has no reliable recording-session identity.
                group = f'{dataset}:recording:{clip}'
            records.append(old.record(paths, dataset, 'sound', group, clip, split))
            matched += 1
        print(f'{dataset}: metadata={len(metadata[dataset])}, matched audio={matched}', flush=True)
        if not matched:
            raise ValueError(f'No ID-matched audio for {dataset}')

    reserved = set()
    if Path(base['diagnostic_manifest']).is_file():
        with Path(base['diagnostic_manifest']).open(newline='') as handle:
            reserved = {row['group'].casefold() for row in csv.DictReader(handle)}
    old.assign_splits(records, config['seed'], reserved)
    # Preserve previous non-Libri held-out groups when extending the collection.
    with previous.open(newline='') as handle:
        prior_groups = {row['group']: row['split'] for row in csv.DictReader(handle) if row['dataset'] != 'libritts_r'}
    buckets = [[] for _ in range(config['num_shards'])]
    seen = set()
    for row in sorted(records, key=lambda item: (item['dataset'], item['source_id'])):
        if row['dataset'] == 'libritts_r' and row['source_id'] in failed:
            exclusions.append(dict(path=str(row['paths'][0]), reason='known_libritts_restoration_failure'))
            continue
        if row['group'] in prior_groups:
            row['split'] = prior_groups[row['group']]
        row['key'] = row['dataset'] + ':' + row['source_id']
        if row['key'] in seen:
            raise ValueError(f'Duplicate inventory identity: {row["key"]}')
        seen.add(row['key'])
        row['paths'] = [str(Path(path).resolve()) for path in row['paths']]
        row['sampling_domain'] = 'sound' if row['dataset'].startswith('wavcaps_') else row['domain']
        row['sampling_dataset'] = 'wavcaps' if row['dataset'].startswith('wavcaps_') else row['dataset']
        if row['dataset'].startswith('wavcaps_'):
            item = metadata[row['dataset']][row['source_id']]
            row['text'] = {key: item[key] for key in ('caption', 'description', 'title', 'file_name', 'tags', 'category') if item.get(key)}
            row['caption'] = item.get('caption', '')
            if row['dataset'] == 'wavcaps_audioset':
                row.update(annotations[videos[row['source_id']]])
        else:
            row.update(rule_labels=[row['domain']], label_source='dataset_identity')
        buckets[int(hashlib.sha256(row['key'].encode()).hexdigest(), 16) % len(buckets)].append(row)
    shards = []
    for index, rows in enumerate(buckets):
        path = output / f'inventory_{index:02d}.jsonl'
        temporary = path.with_suffix('.part')
        with temporary.open('w') as handle:
            for row in rows:
                handle.write(json.dumps(row) + '\n')
        temporary.replace(path)
        shards.append(dict(recordings=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    atomic_json(output / 'inventory.excluded.json', exclusions)
    atomic_json(output / 'inventory.json', dict(config_hash=digest(config), config=config,
                base_config=base, metadata_sha256=metadata_hashes, shards=shards,
                recordings=len(seen), previous_manifest_sha256=hashlib.sha256(previous.read_bytes()).hexdigest()))
    print(f'Inventory ready: {len(seen)} recordings in {len(shards)} shards', flush=True)


def audit(config, output, shard, workers):
    info, rows = inventory(config, output, shard)
    job = ShardJob(output, 'audit', shard, digest(dict(inventory=info, stage='decode-v2')), config['max_seconds'])
    # A changed file invalidates that shard's saved result, not every completed recording.
    stale = []
    for key, value in job.database.execute('SELECT key, value FROM results'):
        value = json.loads(value)
        if 'error' not in value:
            try:
                stat = Path(value['path']).stat()
                if (stat.st_size, stat.st_mtime_ns) == (value['file_size'], value['mtime_ns']):
                    continue
            except OSError:
                pass
            stale.append(key)
    if stale:
        with job.database:
            job.database.executemany('DELETE FROM results WHERE key=?', [(key,) for key in stale])
        job.done.difference_update(stale)
    job.progress(len(rows))
    previous = {}
    with Path(config['previous_manifest']).open(newline='') as handle:
        previous = {row['path']: row for row in csv.DictReader(handle)}
    local = threading.local()

    def inspect(row):
        if not hasattr(local, 'database'):
            path = Path(config['previous_audit'])
            local.database = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) if path.exists() else None
        errors = []
        for path in row['paths']:
            try:
                stat = Path(path).stat()
            except OSError as error:
                errors.append(dict(path=path, reason=str(error)))
                continue
            value, reused = None, False
            cached = previous.get(path)
            if cached and (int(cached['file_size']), int(cached['mtime_ns'])) == (stat.st_size, stat.st_mtime_ns):
                value = {key: int(cached[key]) for key in ('frames', 'sample_rate', 'channels')}
                value.update(duration_seconds=float(cached['duration_seconds']), pcm_sha256=cached['pcm_sha256'])
            elif local.database is not None:
                saved = local.database.execute('SELECT size, mtime, result FROM files WHERE path=?', (path,)).fetchone()
                if saved and saved[:2] == (stat.st_size, stat.st_mtime_ns):
                    value = json.loads(saved[2])
            reused = value is not None
            if value is None:
                value = old.inspect_audio(path)
            if 'error' in value:
                errors.append(dict(path=path, reason=value['error']))
                continue
            return dict(path=path, **value, file_size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                        reused=reused, alternatives_excluded=errors)
        return dict(error='no_usable_audio_copy', alternatives_excluded=errors)

    pending = [row for row in rows if row['key'] not in job.done]
    print(f'Audit shard {shard}: {len(job.done)} cached, {len(pending)} outstanding', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(pending), workers * 2):
            if not job.running():
                break
            futures = {pool.submit(inspect, row): row for row in pending[offset:offset + workers * 2]}
            for future in as_completed(futures):
                row = futures[future]
                job.save([(row['key'], future.result())])
            job.progress(len(rows))
    state = job.progress(len(rows))
    return 0 if state['complete'] else 75


def merge(config, output):
    lock = (output / 'merge.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    records, exclusions, label_models = [], json.loads((output / 'inventory.excluded.json').read_text()), set()
    for shard in range(config['num_shards']):
        info, rows = inventory(config, output, shard)
        results = {}
        for stage in ('audit', 'labels'):
            prefix = output / f'{stage}_{shard:02d}'
            state = json.loads(prefix.with_suffix('.progress.json').read_text())
            if not state['complete'] or state['total'] != len(rows):
                raise ValueError(f'Incomplete {stage} shard {shard}; rerun that array task')
            with sqlite3.connect(prefix.with_suffix('.sqlite').as_uri() + '?mode=ro', uri=True) as source:
                from classify_wavcaps import label_fingerprint
                expected = digest(dict(inventory=info, stage='decode-v2')) if stage == 'audit' else label_fingerprint(config, info)
                if source.execute("SELECT value FROM state WHERE key='fingerprint'").fetchone() != (expected,):
                    raise ValueError(f'Stale {stage} shard {shard}')
                results[stage] = {key: json.loads(value) for key, value in source.execute('SELECT key, value FROM results')}
            if results[stage].keys() != {row['key'] for row in rows}:
                raise ValueError(f'Record coverage mismatch for {stage} shard {shard}')
        for row in rows:
            audio, label = results['audit'][row['key']], results['labels'][row['key']]
            exclusions.extend(audio.pop('alternatives_excluded', []))
            reason = audio_exclusion(row, audio, config['evaluation_seconds'])
            if reason:
                exclusions.append(dict(path=audio.get('path', row['paths'][0]), reason=reason))
                continue
            labels = label['content_labels']
            row.update(audio, domain=labels[0] if len(labels) == 1 else 'mixed' if labels else 'unknown',
                       content_labels=';'.join(labels), needs_review=label['needs_review'],
                       label_source=label['label_source'], metadata_text=row.get('text', {}))
            label_models.add(label.get('model_revision', 'none'))
            records.append(row)
    if len(label_models - {'none'}) > 1:
        raise ValueError('Mixed labeling model revisions; use consistent shard settings')
    # Global deduplication happens after all shards; held-out copies win.
    kept, seen_pcm, seen_ids, groups, seen_paths = [], set(), set(), {}, set()
    for row in sorted(records, key=lambda row: ({'test': 0, 'val': 1, 'train': 2}[row['split']], row['path'])):
        identity = row['dataset'], row['source_id']
        if row['pcm_sha256'] in seen_pcm or identity in seen_ids or row['path'] in seen_paths:
            exclusions.append(dict(path=row['path'], reason='duplicate_id_or_decoded_pcm'))
            continue
        if groups.setdefault(row['group'], row['split']) != row['split']:
            raise ValueError(f'Group crosses splits: {row["group"]}')
        stat = Path(row['path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (row['file_size'], row['mtime_ns']):
            raise ValueError(f'Audio changed after shard audit: {row["path"]}')
        seen_pcm.add(row['pcm_sha256']); seen_ids.add(identity); seen_paths.add(row['path'])
        kept.append(row)
    required = {'ears', 'libritts_r', 'jamendo', 'medleydb'} | {item[1] for item in WAVCAPS.values()}
    if not required <= {row['dataset'] for row in kept}:
        raise ValueError('A required dataset has no usable recordings after filtering')
    for split in ('train', 'val', 'test'):
        if {row['sampling_domain'] for row in kept if row['split'] == split} != {'speech', 'music', 'sound'}:
            raise ValueError(f'Missing sampling domain in {split}')
    fields = ['path', 'split', 'group', 'domain', 'dataset', 'source_id', 'start_seconds', 'sample_rate',
              'channels', 'frames', 'duration_seconds', 'file_size', 'mtime_ns', 'pcm_sha256',
              'sampling_domain', 'sampling_dataset', 'content_labels', 'label_source', 'needs_review',
              'caption', 'metadata_text', 'source_start_seconds', 'official_split', 'event_labels', 'event_intervals']
    path = output / 'postfilter.csv'
    # Readiness is published only after every loader profile passes.
    (output / 'postfilter.summary.json').unlink(missing_ok=True)
    temporary = path.with_suffix('.csv.part')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in kept:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    temporary.replace(path)
    with (output / 'postfilter.excluded.csv.part').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'reason'])
        writer.writeheader(); writer.writerows(exclusions)
    (output / 'postfilter.excluded.csv.part').replace(output / 'postfilter.excluded.csv')
    sample, sample_counts, uncertain = [], Counter(), 0
    for row in sorted(kept, key=lambda row: digest(row['key'])):
        if not row['dataset'].startswith('wavcaps_'):
            continue
        key = row['dataset'], row['domain']
        if sample_counts[key] < 5 or (row['needs_review'] and uncertain < 100):
            sample.append(row)
            sample_counts[key] += 1
            uncertain += int(row['needs_review'])
    with (output / 'postfilter.review.csv.part').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['key', 'path', 'dataset', 'domain', 'content_labels',
                                                   'label_source', 'needs_review', 'caption'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(sample)
    (output / 'postfilter.review.csv.part').replace(output / 'postfilter.review.csv')
    statistics = defaultdict(lambda: [0, 0.0, set()])
    for row in kept:
        value = statistics[row['split'], row['domain'], row['dataset']]
        value[0] += 1; value[1] += row['duration_seconds'] / 3600; value[2].add(row['group'])
    from flow_bitrate.check_postfilter_data import check_loaders
    profiles = check_loaders(path, config['evaluation_seconds'], 4)
    points = [dict(split=key[0], domain=key[1], dataset=key[2], recordings=value[0], hours=value[1], groups=len(value[2]))
              for key, value in sorted(statistics.items())]
    atomic_json(output / 'postfilter.summary.json', dict(audit='full_decode_and_seek_v1',
                manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), config=config,
                recordings=len(kept), statistics=points, excluded=dict(Counter(row['reason'] for row in exclusions)),
                loader_check=profiles, inventory=info, label_model_revisions=sorted(label_models),
                reused_decode_checks=sum(row['reused'] for row in kept),
                notes=['Content labels are metadata annotations/inferences, not verified absence labels.',
                       'BBC split unit is recording ID; recording-session overlap is not established.',
                       'All WavCaps sources share one sampling bucket, sampled uniformly over recordings.',
                       'Exact PCM duplicates removed; near duplicates and frozen-codec pretraining overlap are not ruled out.']))
    for point in points:
        print(point, flush=True)
    print(f'Ready: {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['inventory', 'audit', 'merge', 'status'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    config, base, output = read_config(args.config)
    if not 0 <= args.shard < config['num_shards'] or args.workers < 1:
        raise ValueError('Invalid shard or worker count')
    if args.stage == 'inventory':
        build_inventory(config, base, output)
    elif args.stage == 'audit':
        return audit(config, output, args.shard, args.workers)
    elif args.stage == 'merge':
        merge(config, output)
    else:
        for shard in range(config['num_shards']):
            parts = []
            for stage in ('labels', 'audit'):
                path = output / f'{stage}_{shard:02d}.progress.json'
                value = json.loads(path.read_text()) if path.exists() else None
                parts.append(f'{stage}: {value["completed"]}/{value["total"]}' if value else f'{stage}: not started')
            print(f'{shard:02d}  ' + '  '.join(parts))
        print('Manifest ready:', (output / 'postfilter.summary.json').exists())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
