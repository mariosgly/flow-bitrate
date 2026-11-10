"""Small bits shared by the v2 manifest and labeling jobs."""

import fcntl
import hashlib
import json
import math
import os
import signal
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audio_exclusion(row, audio, evaluation_seconds):
    if 'error' in audio:
        return audio['error']
    seconds = evaluation_seconds if row['split'] == 'test' else 2
    if audio['frames'] < math.ceil(seconds * audio['sample_rate']):
        return f'shorter_than_{seconds}_seconds'
    if row['dataset'] == 'ears' and audio['sample_rate'] != 48000:
        return 'ears_must_be_original_48khz'
    return None


def read_config(path=None):
    config = json.loads(Path(path or ROOT / 'configs/postfilter_data_v2.json').read_text())
    base = json.loads((ROOT / config['base_config']).read_text())
    output = Path(config['output'])
    if not output.is_relative_to(Path('/scratch/mg8253/flow-bitrate')):
        raise ValueError('v2 outputs must stay in the project scratch folder')
    if output == Path(config['previous_manifest']).parent:
        raise ValueError('Use a different folder from the original manifest')
    output.mkdir(parents=True, exist_ok=True)
    return config, base, output


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.part')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class ShardJob:
    def __init__(self, folder, stage, shard, fingerprint, seconds):
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        self.prefix = folder / f'{stage}_{shard:02d}'
        self.lock = self.prefix.with_suffix('.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.stop = False
        self.deadline = min(time.time() + seconds,
                            float(os.environ.get('SLURM_JOB_END_TIME', 'inf')) - 120)
        for name in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(name, self.interrupted)
        self.database = sqlite3.connect(self.prefix.with_suffix('.sqlite'))
        self.database.execute('PRAGMA synchronous=FULL')
        self.database.execute('CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)')
        self.database.execute('CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, value TEXT)')
        saved = self.database.execute("SELECT value FROM state WHERE key='fingerprint'").fetchone()
        if saved and saved[0] != fingerprint:
            raise ValueError('Inputs/settings changed. Use a new output folder, not old shard results.')
        self.database.execute("INSERT OR IGNORE INTO state VALUES ('fingerprint', ?)", (fingerprint,))
        self.database.commit()
        self.done = {key for key, in self.database.execute('SELECT key FROM results')}

    def interrupted(self, *_):
        self.stop = True

    def running(self):
        return not self.stop and time.time() < self.deadline

    def save(self, values):
        with self.database:
            self.database.executemany('INSERT OR REPLACE INTO results VALUES (?, ?)',
                                      [(key, json.dumps(value)) for key, value in values])
        self.done.update(key for key, _ in values)

    def progress(self, total):
        value = dict(completed=len(self.done), total=total, complete=len(self.done) == total,
                     updated=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        atomic_json(self.prefix.with_suffix('.progress.json'), value)
        print(f'{self.prefix.name}: {len(self.done)}/{total}', flush=True)
        return value


def inventory(config, output, shard):
    info = json.loads((output / 'inventory.json').read_text())
    if info['config_hash'] != digest(config):
        raise ValueError('Config differs from the published inventory; use a new folder')
    path = output / f'inventory_{shard:02d}.jsonl'
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != info['shards'][shard]['sha256']:
        raise ValueError(f'Incomplete/changed inventory: {path}')
    return info, [json.loads(line) for line in content.splitlines()]
