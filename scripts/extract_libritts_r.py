"""Check and extract the four downloaded LibriTTS-R archives on HPC."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/scratch/mg8253/datasets/libritss'))
    parser.add_argument('--shard', type=int, choices=range(5), help='extract one archive, for the Slurm array')
    args = parser.parse_args()
    checksums = {}
    for line in (args.root / 'md5sum.txt').read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            checksums[Path(parts[1].lstrip('*')).name] = parts[0]
    output = args.root / 'extracted'
    output.mkdir(exist_ok=True)
    names = ['dev_clean', 'dev_other', 'test_clean', 'test_other',
             'libritts_r_failed_speech_restoration_examples']
    if args.shard is not None:
        names = [names[args.shard]]
    for name in names:
        archive = args.root / f'{name}.tar.gz'
        if not archive.is_file():
            raise FileNotFoundError(archive)
        marker = output / f'.{name}.complete.json'
        signature = dict(size=archive.stat().st_size, mtime=archive.stat().st_mtime_ns)
        if marker.exists() and json.loads(marker.read_text()).get('archive') == signature:
            print(f'Already extracted: {archive.name}', flush=True)
            continue
        print(f'Checking MD5: {archive.name}', flush=True)
        digest = hashlib.md5()
        with archive.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        expected = checksums.get(archive.name)
        if expected is None and name != 'libritts_r_failed_speech_restoration_examples':
            raise ValueError(f'No official checksum for {archive.name}')
        if expected and digest.hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {archive}')
        print(f'Extracting: {archive.name}', flush=True)
        with tarfile.open(archive) as source:
            for member in source:
                target = (output / member.name).resolve()
                if not target.is_relative_to(output.resolve()) or member.issym() or member.islnk():
                    raise ValueError(f'Unexpected archive member: {member.name}')
                source.extract(member, output)
        temporary = marker.with_suffix('.part')
        temporary.write_text(json.dumps(dict(archive=signature, md5=digest.hexdigest())))
        temporary.replace(marker)
    print(f'Ready: {output}', flush=True)


if __name__ == '__main__':
    main()
