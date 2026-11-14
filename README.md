# Predicting Mumbai rent with a neural network

**AI Methods I · Day 4 · Track A1 · IAAC MaAI 2026/27**
**Group 3:** Ahmed Belahcen, Nandini Rathod, Sanjana Kiran and Akanksha Dhumal.

**Model:** [published Hugging Face model](https://huggingface.co/iaac-maai/day4-group3-ai_methods_MumbaiRentals) · [model card](models/README.md).
**Notebook:** [neural_network.ipynb](notebook/neural_network.ipynb), with executed outputs.
**Results:** [evaluation report](reports/evaluation.json).

## What is the project about?

The Day 2 Random Forest predicted monthly advertised rent from eight property features. This project tests a small neural network on the same task and the same test rows. A housing researcher could use the estimates to compare early budget scenarios. An underestimate could lead to an unrealistic budget; an overestimate could rule out an affordable option. The model is a coursework experiment, not a current market valuation tool.

## Data

The cleaned Mumbai file has **18,846 rows and 26 columns**. Each row is an advertisement. The target is `Price_num`, in INR per month. The source is Harshal Honde's [Property Rental Listings Dataset](https://www.kaggle.com/datasets/harshalhonde/property-rental-listings-dataset); the group's earlier documentation declares CC BY 4.0. See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) for the pinned source and checksum.

The notebook downloads the exact cleaned Day 2 CSV if it is missing, then verifies its checksum. Data and model weights are excluded from GitHub by `.gitignore`.

## How the experiment works

1. Keep the Day 2 test split: 15,076 training rows and 3,770 test rows, seed 42.
2. Use 12,060 fitting rows and 3,016 validation rows for model selection.
3. Impute and standardize numeric inputs; impute and one-hot encode categories. Fit preprocessing on the fitting set only during tuning.
4. Compare dropout 0 and 0.2 with the same architecture, optimizer, learning rate and batch size.
5. Choose the lowest validation MAE, then train a fresh model on all outer-training rows for the selected number of epochs.
6. Compare the frozen model with the original Random Forest and a median baseline on the same test set.

The final MLP has **23 inputs → 64 → 32 → 1**, ReLU activations and **3,649 parameters**. Adam uses learning rate 0.001 and batch size 256. The loss is MSE on rent divided by 100,000. Validation selected no dropout and 28 epochs. The tuning encoder had 22 inputs; refitting on the larger training set added one category, so the saved model uses 23.

Training runs on CPU with two PyTorch threads. Measured timings and library versions for the checked run are recorded in `reports/evaluation.json`; timings vary by machine.

## Results

| Model | MAE (INR/month) | RMSE (INR/month) | R² |
|---|---:|---:|---:|
| Median baseline | 67,123.75 | 140,522.95 | -0.0955 |
| Day 2 Random Forest | 38,044.93 | 82,899.45 | 0.6187 |
| Day 4 MLP | 38,008.45 | 82,594.83 | 0.6215 |

The MLP reduces MAE by about 43.4% compared with the median baseline. It improves on the forest by only about 36 INR/month, or 0.1%. The fair conclusion is that the two models perform similarly on this test set.

![01 training curves](figures/01_training_curves.png)

Most of the improvement happens in the first few epochs. Validation MAE selects epoch 28 without dropout; continuing training does not give a steady improvement.

![03 test comparison](figures/03_test_comparison.png)

Both learned models improve on the median baseline. The MLP and Random Forest are almost tied, so the small difference is not enough to claim a meaningful advantage.

![05 failure cases](figures/05_failure_cases.png)

These are the eight largest errors. Their rent and area combinations deserve source checking, but they remain in the evaluation rather than being removed to improve the score.

![06 housing scenarios](figures/06_housing_scenarios.png)

The examples show how estimates change between four handmade property descriptions. They have no observed rent, so this is an application example rather than another accuracy test.

## Run the project

Use Python 3.12. From the project directory on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_notebook.py
```

You can also open the notebook in Jupyter or VS Code, select this environment and use **Restart Kernel → Run All**. The notebook accepts either the project root or `notebook/` as its working directory.

`TRAIN=True` runs the full experiment. `TRAIN=False` loads the final neural network, uses the recorded histories and refits the inexpensive Day 2 comparators. It does not need intermediate neural-network checkpoints. `publication.json` pins the existing Hugging Face model to revision `1b628c61c956fdd97543c1f21e258d023a041109`. Missing model files download automatically from that revision. Retraining only writes local files; it never uploads or overwrites the published model.

For a prediction/reload check, run `python predict_local.py`. `python execute_local.py --train` executes the notebook cells sequentially without Jupyter's kernel transport, capturing their real outputs. Use `python run_notebook.py --train` for a fresh Jupyter-kernel training run, or `python run_notebook.py --no-train` to load the published model. `python verify_project.py --remote` checks fresh downloads and predictions.

## Files

| Path | Purpose |
|---|---|
| `data/` | Exact cleaned CSV, downloaded if absent; excluded from GitHub |
| `notebook/` | Documented notebook and execution outputs |
| `figures/` | Seven exported plots |
| `models/` | Weights, preprocessing, configuration, model card and its figures |
| `reports/` | Histories, metrics, scenarios and inspection outputs |
| `source_reference/` | Day 2 evidence used to reproduce the comparator |
| `project_io.py` | Data checksum and missing-file downloads |
| `predict_local.py` | Load the saved model and predict |
| `publication.json` | Published repository and pinned revision |
| `requirements.txt` | Pinned packages |
| `LICENSE` | MIT license for project code and model |

## Limitations

These are historical advertisements, not verified rental transactions. The collection date is unknown, `Location` does not include neighborhoods, some area units may be inconsistent, and many balcony values are missing. There are 752 test rows with an exact feature-and-target match in training. We report this because the random split can make generalization look better than it is. Excluding those exact matches gives an MLP MAE of about 41,619 INR/month, but that check cannot identify every repeated dwelling. The model is not validated for current valuations, other cities, eligibility decisions or financial commitments.

## AI usage

ChatGPT/Codex assisted with code, figures and documentation. Verification included the dataset checksum, the Day 2 split and Random Forest settings, full notebook execution and saved-model predictions.

## License and hand-in

Code and model: MIT. Data: separately attributed under its declared CC BY 4.0 license. 
