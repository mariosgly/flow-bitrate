"""Download once with network access; labeling jobs only read local weights."""

import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / 'configs/postfilter_data_v2.json').read_text())
target = Path(config['model_dir'])
target.mkdir(parents=True, exist_ok=True)
marker = target / 'download.json'
request = target / 'download.request.json'
if request.exists():
    revision = json.loads(request.read_text())['revision']
elif marker.exists():
    revision = json.loads(marker.read_text())['revision']
else:
    revision = HfApi().model_info(config['model_id']).sha
    temporary = request.with_suffix('.part')
    temporary.write_text(json.dumps(dict(model=config['model_id'], revision=revision)))
    temporary.replace(request)
snapshot_download(config['model_id'], revision=revision, local_dir=target,
                  allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
temporary = marker.with_suffix('.part')
temporary.write_text(json.dumps(dict(model=config['model_id'], revision=revision), indent=2))
temporary.replace(marker)
print(f'Ready: {target} (revision {revision})')
