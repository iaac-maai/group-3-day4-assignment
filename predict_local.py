"""Offline prediction from this project's trusted local artifacts."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent

def load_model():
    from project_io import ensure_model
    ensure_model(ROOT)
    config = json.loads((ROOT/'models/config.json').read_text())
    model = nn.Sequential(
        nn.Linear(config['input_dim'],64),nn.ReLU(),nn.Dropout(config['dropout']),
        nn.Linear(64,32),nn.ReLU(),nn.Dropout(config['dropout']),nn.Linear(32,1))
    model.load_state_dict(torch.load(ROOT/'models/model.pt',map_location='cpu',weights_only=True))
    model.eval()
    preprocessor = joblib.load(ROOT/'models/scaler.joblib')
    return model, preprocessor, config

def predict(frame):
    model, preprocessor, config = load_model()
    missing = set(config['features']) - set(frame.columns)
    if missing: raise ValueError(f'Missing input columns: {sorted(missing)}')
    values = torch.tensor(preprocessor.transform(frame[config['features']]),dtype=torch.float32)
    with torch.no_grad():
        return np.maximum(model(values).numpy().ravel()*config['target_scale'],0)

if __name__ == '__main__':
    examples = pd.read_csv(ROOT/'reports/scenario_predictions.csv',index_col=0)
    estimates = predict(examples)
    assert np.allclose(estimates,examples['MLP_INR_month'],rtol=1e-6,atol=.1)
    print('Fresh-process reload verified against all saved scenarios.')
    for name, estimate in zip(examples.index,estimates):
        print(f'{name}: INR {estimate:,.2f}/month')
