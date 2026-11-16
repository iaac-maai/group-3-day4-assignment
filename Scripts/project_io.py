"""Get the exact course data and saved model files when they are not local."""
from pathlib import Path
import hashlib
import json
import re
from urllib.request import urlopen

DATA_SHA256 = '354d28e07432137cdc97a2197a556269137e74135af01186420e2e2917a19ee9'
DATA_URL = ('https://raw.githubusercontent.com/iaac-maai/group-3-day2-assignment/'
            '6b2a6d17af8c4d8950f12e60dc97f8b9a8d4dbc1/data/mumbai_cleaned.csv')
MODEL_FILES = ('model.pt', 'scaler.joblib', 'config.json')

def download(url, destination, expected_sha256=None):
    destination = Path(destination)
    try:
        with urlopen(url, timeout=30) as response:
            content = response.read()
    except Exception as exc:
        raise RuntimeError(f'Could not download {url}. Check access or copy the file to {destination}.') from exc
    if expected_sha256 and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('Downloaded file has an unexpected checksum; the local file was not replaced.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination

def ensure_data(root):
    path = Path(root) / 'data/mumbai_cleaned.csv'
    if not path.exists():
        download(DATA_URL, path, DATA_SHA256)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError('Use the unchanged Day 2 CSV: its contents and row order determine the test split.')
    return path

def ensure_model(root):
    root = Path(root)
    missing = [name for name in MODEL_FILES if not (root / 'models' / name).exists()]
    if not missing:
        return root / 'models'
    publication = json.loads((root / 'publication.json').read_text(encoding='utf-8'))
    repo, revision = publication.get('model_repo_id'), publication.get('model_revision')
    if not repo:
        raise RuntimeError('Model publication is still pending. Use the local ZIP, set TRAIN=True, or fill in '
                           'publication.json after the model has actually been published.')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
        raise ValueError('model_repo_id must be an owner/repository identifier.')
    if not revision or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Set model_revision to the published 40-character commit SHA for reproducibility.')
    for name in missing:
        download(f'https://huggingface.co/{repo}/resolve/{revision}/{name}', root / 'models' / name)
    return root / 'models'
