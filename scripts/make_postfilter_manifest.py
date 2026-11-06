"""Build and fully decode-check the postfilter dataset. Run this on HPC, not locally."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path('/scratch/mg8253/flow-bitrate')
AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3', '.ogg'}
PROGRESS_PATH = None
SCAN_DIRECTORY = None
REFRESH_INDEX = False


def report(stage, **details):
    status = dict(time=datetime.now(timezone.utc).isoformat(timespec='seconds'), stage=stage, **details)
    print(f"[{status['time']}] {stage}: " + ' '.join(f'{key}={value}' for key, value in details.items()), flush=True)
    if PROGRESS_PATH is not None:
        temporary = PROGRESS_PATH.with_suffix('.json.part')
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(PROGRESS_PATH)


def audio_files(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset directory is not accessible: {root}')
    cache_path = SCAN_DIRECTORY / (hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:20] + '.sqlite')
    database = sqlite3.connect(cache_path)
    database.execute('CREATE TABLE IF NOT EXISTS paths (original TEXT PRIMARY KEY, resolved TEXT)')
    database.execute('CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value INTEGER)')
    if REFRESH_INDEX:
        database.execute('DELETE FROM paths')
        database.execute('DELETE FROM state')
        database.commit()
    known = dict(database.execute('SELECT original, resolved FROM paths'))
    if database.execute("SELECT value FROM state WHERE key='complete'").fetchone():
        database.close()
        report('Reusing completed directory scan', root=str(root), files=len(known))
        return [Path(path) for path in sorted(set(known.values()))]
    report('Scanning audio directories', root=str(root), cached_files=len(known))
    visited, new_files, last_report = set(), 0, time.monotonic()
    def walk_error(error):
        raise error
    for directory, folders, names in os.walk(root, followlinks=True, onerror=walk_error):
        stat = Path(directory).stat()
        identity = stat.st_dev, stat.st_ino
        if identity in visited:
            folders[:] = []
            continue
        visited.add(identity)
        folders.sort()
        for name in sorted(names):
            if Path(name).suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            original = str(Path(directory) / name)
            if original not in known:
                resolved = str(Path(original).resolve())
                known[original] = resolved
                database.execute('INSERT OR REPLACE INTO paths VALUES (?, ?)', (original, resolved))
                new_files += 1
                if new_files % 256 == 0:
                    database.commit()
            if time.monotonic() - last_report >= 15:
                database.commit()
                report('Scanning audio directories', root=str(root), files=len(known), directory=directory)
                last_report = time.monotonic()
    database.execute("INSERT OR REPLACE INTO state VALUES ('complete', 1)")
    database.commit()
    database.close()
    report('Directory scan complete', root=str(root), files=len(known))
    return [Path(path) for path in sorted(set(known.values()))]


def record(paths, dataset, domain, group, source_id, split='pool'):
    return dict(paths=[str(path) for path in paths], dataset=dataset, domain=domain,
                group=group, source_id=source_id, split=split, start_seconds='')


def collect(config):
    records, exclusions = [], []
    report('Collecting EARS recordings')
    for path in audio_files(config['ears']):
        speaker = next((part for part in path.parts if re.fullmatch(r'p\d{3}', part)), None)
        if speaker is None or not 1 <= int(speaker[1:]) <= 107:
            exclusions.append(dict(path=str(path), reason='unknown_ears_speaker'))
            continue
        number = int(speaker[1:])
        split = 'train' if number <= 99 else 'val' if number <= 101 else 'test'
        records.append(record([path], 'ears', 'speech', f'speech:ears:{speaker}', path.stem, split))
    report('Collecting LibriTTS-R recordings', collected=len(records))
    for path in audio_files(config['libritts']):
        subset = next((part for part in path.parts if part.startswith(('train-clean-', 'train-other-', 'dev-', 'test-'))), None)
        speaker = path.stem.split('_', 1)[0]
        if subset is None or not speaker.isdigit():
            exclusions.append(dict(path=str(path), reason='unknown_libritts_subset_or_speaker'))
            continue
        split = 'val' if subset.startswith('dev-') else 'test' if subset.startswith('test-') else 'pool'
        records.append(record([path], 'libritts_r', 'speech', f'speech:{speaker}', path.stem, split))
    # Jamendo's numeric audio folders do not identify artists; use its metadata.
    report('Reading Jamendo artist metadata', collected=len(records))
    metadata = Path(config['jamendo_metadata'])
    if not metadata.is_file():
        raise FileNotFoundError(f'Need Jamendo artist metadata at {metadata}; see README installation commands')
    tracks = {}
    with metadata.open(newline='') as handle:
        reader = csv.reader(handle, delimiter='\t')
        header = next(reader, [])
        if header[:4] != ['TRACK_ID', 'ARTIST_ID', 'ALBUM_ID', 'PATH']:
            raise ValueError('Jamendo metadata needs TRACK_ID, ARTIST_ID, ALBUM_ID, PATH columns')
        for row in reader:
            if len(row) >= 4:
                tracks[Path(row[3]).stem] = row[1]
    jamendo = defaultdict(list)
    for path in audio_files(config['jamendo']):
        jamendo[path.stem].append(path)
    for track_id, paths in sorted(jamendo.items()):
        artist = tracks.get(track_id)
        if not artist:
            exclusions.append(dict(path=str(paths[0]), reason='jamendo_track_missing_artist_metadata'))
            continue
        paths.sort(key=lambda path: ({'.flac': 0, '.wav': 1, '.mp3': 2}.get(path.suffix.lower(), 3), str(path)))
        records.append(record(paths, 'jamendo', 'music', f'music:jamendo:{artist}', track_id))
    # MedleyDB is an additional held-out corpus. Never read stems/raw microphones.
    report('Collecting MedleyDB mixes', collected=len(records))
    for path in audio_files(config['medleydb']):
        if path.stem.lower().endswith('_mix'):
            artist = path.stem[:-4].split('_', 1)[0]
            records.append(record([path], 'medleydb', 'music', f'music:{artist}', path.stem, 'test'))
    wavcaps = Path(config['wavcaps'])
    metadata_path = wavcaps / 'json_files/FreeSound/fsd_final.json'
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    report('Reading FreeSound uploader metadata', path=str(metadata_path))
    payload = json.loads(metadata_path.read_text())
    items = payload['data'] if isinstance(payload, dict) else payload
    freesound = defaultdict(list)
    for folder in ('FreeSound_flac', 'FreeSound_wav', 'FreeSound'):
        directory = wavcaps / 'data' / folder
        if directory.is_dir():
            for path in audio_files(directory):
                freesound[path.stem].append(path)
    if not freesound:
        raise ValueError(f'No extracted FreeSound audio under {wavcaps / "data"}')
    used_ids, last_report = set(), time.monotonic()
    for index, item in enumerate(items):
        if time.monotonic() - last_report >= 15:
            report('Matching FreeSound metadata to audio', checked=index, total=len(items), collected=len(records))
            last_report = time.monotonic()
        clip = Path(str(item['id'])).stem
        if clip in used_ids:
            continue
        used_ids.add(clip)
        paths = freesound.get(clip, [])
        author = item.get('author') or item.get('username') or item.get('uploader')
        if isinstance(author, dict):
            author = author.get('username') or author.get('name') or author.get('id')
        author = str(author or '').strip().strip('/')
        if '/people/' in author:
            author = author.split('/people/', 1)[1].split('/')[0]
        if not paths or not author:
            exclusions.append(dict(path=str(paths[0]) if paths else f'FreeSound:{clip}',
                                   reason='missing_audio' if not paths else 'missing_freesound_uploader'))
            continue
        paths.sort(key=lambda path: (path.suffix.lower() != '.flac', str(path)))
        records.append(record(paths, 'wavcaps_freesound', 'sound', f'sound:{author.casefold()}', clip))
    for clip in freesound.keys() - used_ids:
        exclusions.append(dict(path=str(freesound[clip][0]), reason='freesound_audio_missing_metadata'))
    # Optional MusicCaps audio must have YouTube IDs in its filenames; no downloads here.
    if config.get('musiccaps'):
        for path in audio_files(config['musiccaps']):
            match = re.match(r'^(?:Y)?([A-Za-z0-9_-]{11})(?:$|[_ .])', path.stem)
            if match:
                video = match.group(1)
                records.append(record([path], 'musiccaps', 'music', f'music:youtube:{video}', video, 'test'))
            else:
                exclusions.append(dict(path=str(path), reason='musiccaps_filename_missing_youtube_id'))
    required = {'ears', 'libritts_r', 'jamendo', 'medleydb', 'wavcaps_freesound'}
    if config.get('musiccaps'):
        required.add('musiccaps')
    missing = required - {row['dataset'] for row in records}
    if missing:
        raise ValueError(f'No metadata-matched audio for: {sorted(missing)}')
    report('Dataset inventory complete', recordings=len(records), exclusions=len(exclusions))
    return records, exclusions


def assign_splits(records, seed, reserved):
    # Honour available official LibriTTS dev/test speakers, including accidental copies.
    official = {row['group']: row['split'] for row in records
                if row['dataset'] == 'libritts_r' and row['split'] in {'val', 'test'}}
    for row in records:
        if row['group'] in official:
            row['split'] = official[row['group']]
        if row['group'].casefold() in reserved:
            row['split'] = 'test'
    pools = defaultdict(list)
    for row in records:
        if row['split'] == 'pool':
            pools[row['dataset']].append(row)
    for dataset, values in pools.items():
        groups = sorted({row['group'] for row in values},
                        key=lambda group: hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest())
        if len(groups) < 3:
            raise ValueError(f'Need at least three source groups for {dataset}')
        count = max(1, len(groups) // 10)
        # With official LibriTTS dev/test files, leave its training subsets intact.
        official_roles = {row['split'] for row in records if row['dataset'] == dataset and row['group'] in official}
        val = set() if 'val' in official_roles else set(groups[:count])
        offset = len(val)
        test = set() if 'test' in official_roles else set(groups[offset:offset + count])
        for row in values:
            row['split'] = 'val' if row['group'] in val else 'test' if row['group'] in test else 'train'


def inspect_audio(path):
    try:
        with sf.SoundFile(path) as audio:
            rate, channels = audio.samplerate, audio.channels
            digest = hashlib.sha256(f'{rate}:{channels}:'.encode())
            frames, peak = 0, 0.0
            for block in audio.blocks(blocksize=65536, dtype='float32', always_2d=True):
                if not np.isfinite(block).all():
                    raise ValueError('non-finite samples')
                digest.update(block.astype('<f4', copy=False).tobytes())
                peak = max(peak, float(np.abs(block.mean(axis=1)).max(initial=0)))
                frames += len(block)
            if not frames or peak <= 1e-5:
                raise ValueError('empty or silent after mono conversion')
            # Check the same partial seek/read operation that the loader uses.
            audio.seek(max(0, frames // 2 - rate))
            excerpt = audio.read(min(2 * rate, frames), dtype='float32', always_2d=True)
            if len(excerpt) != min(2 * rate, frames) or not np.isfinite(excerpt).all():
                raise ValueError('failed partial seek/read')
            return dict(frames=frames, sample_rate=rate, channels=channels,
                        duration_seconds=frames / rate, pcm_sha256=digest.hexdigest())
    except (OSError, RuntimeError, ValueError) as error:
        return dict(error=str(error))


def main():
    global PROGRESS_PATH, SCAN_DIRECTORY, REFRESH_INDEX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/postfilter_data.json')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--refresh-index', action='store_true', help='rescan roots after adding/removing dataset files')
    parser.add_argument('--input-manifest', type=Path, help='audit an existing CSV instead of discovering datasets')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config['manifest']).resolve()
    if not output.is_relative_to(SCRATCH.resolve()):
        raise ValueError(f'Manifest must stay under {SCRATCH}')
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    PROGRESS_PATH = output.with_suffix('.progress.json')
    SCAN_DIRECTORY = output.with_suffix('.scans')
    SCAN_DIRECTORY.mkdir(exist_ok=True)
    REFRESH_INDEX = args.refresh_index
    report('Starting data preparation', manifest=str(output), workers=args.workers)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f'{output} exists; use --overwrite to rebuild from the cached audit')
    if args.input_manifest:
        with args.input_manifest.open(newline='') as handle:
            records = list(csv.DictReader(handle))
        for row in records:
            row['paths'] = [str((args.input_manifest.parent / row['path']).resolve())]
            row.setdefault('dataset', 'custom')
            row.setdefault('source_id', Path(row['path']).stem)
        exclusions = []
    else:
        records, exclusions = collect(config)
        reserved = set()
        diagnostic = Path(config['diagnostic_manifest'])
        if diagnostic.is_file():
            with diagnostic.open(newline='') as handle:
                reserved = {row['group'].casefold() for row in csv.DictReader(handle)}
        report('Assigning source-group splits', recordings=len(records))
        assign_splits(records, config['seed'], reserved)
    paths = sorted({path for row in records for path in row['paths']})
    if any(Path(path).suffix.lower() == '.mp3' for path in paths) and 'MP3' not in sf.available_formats():
        raise RuntimeError('This soundfile/libsndfile build cannot decode MP3. Repair the environment before training.')
    database = sqlite3.connect(output.with_suffix('.audit.sqlite'))
    database.execute('CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, mtime INTEGER, result TEXT)')
    inspected, pending, last_report = {}, [], time.monotonic()
    report('Checking saved decode audits', files=len(paths))
    for index, path in enumerate(paths):
        if time.monotonic() - last_report >= 15:
            report('Checking saved decode audits', checked=index, total=len(paths),
                   cached=len(inspected), need_decode=len(pending), path=path)
            last_report = time.monotonic()
        try:
            stat = Path(path).stat()
        except OSError as error:
            inspected[path] = dict(error=str(error))
            continue
        saved = database.execute('SELECT size, mtime, result FROM files WHERE path=?', (path,)).fetchone()
        if saved and saved[:2] == (stat.st_size, stat.st_mtime_ns) and 'error' not in json.loads(saved[2]):
            inspected[path] = json.loads(saved[2])
        else:
            pending.append((path, stat.st_size, stat.st_mtime_ns))
    report('Full decode audit', audio_copies=len(paths), cached=len(inspected), to_decode=len(pending))
    completed, uncommitted, last_commit = 0, 0, time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as workers, tqdm(total=len(pending), desc='decode audit') as progress:
        for start in range(0, len(pending), 64):
            batch = pending[start:start + 64]
            futures = {workers.submit(inspect_audio, item[0]): item for item in batch}
            while futures:
                done, _ = wait(futures, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    path, size, mtime = futures.pop(future)
                    result = future.result()
                    inspected[path] = result
                    database.execute('INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?)',
                                     (path, size, mtime, json.dumps(result)))
                    completed += 1
                    uncommitted += 1
                    progress.update()
                    if uncommitted >= 16:
                        database.commit()
                        uncommitted = 0
                if time.monotonic() - last_commit >= 15:
                    database.commit()
                    uncommitted = 0
                    report('Full decode audit', completed=completed, total=len(pending),
                           cached=len(paths) - len(pending),
                           outstanding_examples=[item[0] for item in list(futures.values())[:2]])
                    last_commit = time.monotonic()
            database.commit()
            uncommitted = 0
    database.close()
    report('Decode audit complete; selecting valid recordings', checked=len(paths))
    selected = []
    for item in records:
        row = {key: value for key, value in item.items() if key not in {'paths', 'path'}}
        chosen = next((path for path in item['paths'] if 'error' not in inspected[path]), None)
        if chosen is None:
            for path in item['paths']:
                exclusions.append(dict(path=path, reason=inspected[path]['error']))
            continue
        for path in item['paths']:
            if 'error' in inspected[path]:
                exclusions.append(dict(path=path, reason=inspected[path]['error']))
        row.update(inspected[chosen])
        row['path'] = chosen
        stat = Path(chosen).stat()
        row['file_size'], row['mtime_ns'] = stat.st_size, stat.st_mtime_ns
        minimum = config['evaluation_seconds'] if row['split'] == 'test' else 2
        if row['duration_seconds'] < minimum + .02:
            exclusions.append(dict(path=chosen, reason=f'shorter_than_{minimum}_seconds'))
            continue
        if row['dataset'] == 'ears' and row['sample_rate'] != 48000:
            exclusions.append(dict(path=chosen, reason='ears_must_be_original_48khz'))
            continue
        selected.append(row)
    # Prefer held-out copies when identical decoded PCM exists across corpora.
    report('Removing duplicate recordings', candidates=len(selected))
    kept, seen_audio, seen_ids, groups = [], set(), set(), {}
    for row in sorted(selected, key=lambda item: ({'test': 0, 'val': 1, 'train': 2}[item['split']], item['path'])):
        identity = row['dataset'], row['source_id']
        if row['pcm_sha256'] in seen_audio or identity in seen_ids:
            exclusions.append(dict(path=row['path'], reason='duplicate_id_or_decoded_pcm'))
            continue
        if groups.setdefault(row['group'], row['split']) != row['split']:
            raise ValueError(f'Source group crosses splits: {row["group"]}')
        seen_audio.add(row['pcm_sha256'])
        seen_ids.add(identity)
        kept.append(row)
    for split in ('train', 'val', 'test'):
        for domain in ('speech', 'music', 'sound'):
            if not any(row['split'] == split and row['domain'] == domain for row in kept):
                raise ValueError(f'No readable {split}/{domain} audio; inspect dataset paths and metadata')
    if not args.input_manifest:
        for dataset in ('ears', 'libritts_r', 'jamendo', 'medleydb', 'wavcaps_freesound'):
            if not any(row['dataset'] == dataset for row in kept):
                raise ValueError(f'No usable audio after decode checks for {dataset}')
    fields = ['path', 'split', 'group', 'domain', 'dataset', 'source_id', 'start_seconds', 'sample_rate',
              'channels', 'frames', 'duration_seconds', 'file_size', 'mtime_ns', 'pcm_sha256']
    report('Writing audited manifest', recordings=len(kept), exclusions=len(exclusions))
    temporary = output.with_suffix('.csv.part')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, '') for key in fields} for row in kept)
    temporary.replace(output)
    from flow_bitrate.check_postfilter_data import check_loaders
    report('Checking actual CPU dataloaders')
    loader_profiles = check_loaders(output, config['evaluation_seconds'], min(args.workers, 4))
    rejected = output.with_suffix('.excluded.csv')
    with rejected.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'reason'])
        writer.writeheader()
        writer.writerows(exclusions)
    counts = Counter((row['split'], row['domain'], row['dataset']) for row in kept)
    statistics = []
    for (split, domain, dataset), count in sorted(counts.items()):
        values = [row for row in kept if (row['split'], row['domain'], row['dataset']) == (split, domain, dataset)]
        point = dict(split=split, domain=domain, dataset=dataset, recordings=count,
                     groups=len({row['group'] for row in values}),
                     hours=sum(row['duration_seconds'] for row in values) / 3600)
        statistics.append(point)
        print(point, flush=True)
    summary = dict(manifest_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), audit='full_decode_and_seek_v1',
                   config=config, soundfile_version=sf.__version__, libsndfile_version=sf.__libsndfile_version__,
                   recordings=len(kept), statistics=statistics, excluded=dict(Counter(row['reason'] for row in exclusions)),
                   loader_check=loader_profiles,
                   metadata_sha256={name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                    for name, path in [('jamendo', config['jamendo_metadata']),
                                                       ('freesound', str(Path(config['wavcaps']) / 'json_files/FreeSound/fsd_final.json'))]
                                    if Path(path).is_file()},
                   notes=['WavCaps user-created splits are ignored. Only FreeSound is used; AudioSet_SL, BBC and SoundBible are excluded.',
                          'WAV/FLAC copies share an ID; exact decoded-PCM duplicates are removed. Resampled/near duplicates are not ruled out.',
                          'EARS speaker split follows its benchmark; Jamendo/FreeSound splits are our source-group split.',
                          'Cross-corpus artist identities and released-checkpoint pretraining overlap are not fully known.'])
    summary_path = output.with_suffix('.summary.json')
    temporary = summary_path.with_suffix('.json.part')
    temporary.write_text(json.dumps(summary, indent=2) + '\n')
    temporary.replace(summary_path)
    report('Ready', manifest=str(output), recordings=len(kept), summary=str(summary_path))
    print(f'Ready: {output}\nAudit summary: {summary_path}\nExcluded files: {rejected}', flush=True)


if __name__ == '__main__':
    main()
