"""Check saved predictions, dimensions, documentation links and published downloads."""
import hashlib, json, re, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from Scripts.predict_local import predict
from Scripts.project_io import ensure_model, ensure_data

root=Path(__file__).resolve().parent
config=json.loads((root/'models/config.json').read_text())
frame=pd.read_csv(root/'reports/scenario_predictions.csv',index_col=0)
assert np.allclose(predict(frame),frame.MLP_INR_month,rtol=1e-6,atol=.1)
try:
    predict(frame.drop(columns=config['features'][0]))
except ValueError:
    pass
else:
    raise AssertionError('Missing input columns must be rejected')
nb=json.loads((root/'notebook/neural_network.ipynb').read_text(encoding='utf-8'))
assert all(c.get('execution_count') and not any(o['output_type']=='error' for o in c.get('outputs',[])) for c in nb['cells'] if c['cell_type']=='code')
for doc in [root/'README.md',root/'models/README.md']:
    text=doc.read_text(encoding='utf-8')
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
        if not target.startswith(('https://','http://','#')):
            assert (doc.parent/target.split('#')[0]).exists(), (doc,target)
assert np.isclose(pd.read_csv(root/'reports/test_metrics.csv',index_col=0).loc['Day 4 MLP','MAE_INR'],38008.44703378854,atol=.1)
print('Local reload, missing-column guard, notebook outputs, metrics and Markdown links: PASS')
if '--remote' in __import__('sys').argv:
    with tempfile.TemporaryDirectory() as temp:
        fresh=Path(temp)
        (fresh/'publication.json').write_bytes((root/'publication.json').read_bytes())
        folder=ensure_model(fresh)
        ensure_data(fresh)
        remote_config=json.loads((folder/'config.json').read_text())
        model=nn.Sequential(nn.Linear(remote_config['input_dim'],64),nn.ReLU(),nn.Dropout(remote_config['dropout']),nn.Linear(64,32),nn.ReLU(),nn.Dropout(remote_config['dropout']),nn.Linear(32,1))
        model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True)); model.eval()
        pre=joblib.load(folder/'scaler.joblib')
        with torch.no_grad():
            actual=np.maximum(model(torch.tensor(pre.transform(frame[remote_config['features']]),dtype=torch.float32)).numpy().ravel()*remote_config['target_scale'],0)
        assert np.allclose(actual,frame.MLP_INR_month,rtol=1e-6,atol=.1)
        print('Empty-directory pinned Hugging Face download, CSV checksum and four predictions: PASS')
