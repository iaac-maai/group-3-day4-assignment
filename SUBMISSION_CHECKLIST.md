# Day 4 requirements

| Required item | Evidence and status |
|---|---|
| Same Day 2 data, target and test split | CSV checksum, Price_num, seed 42; preserved |
| Data inspection, train/validation/test separation | Notebook sections 1–2 |
| MLP training loop and baseline | PyTorch MLP, Day 2 Random Forest, median baseline |
| Model inspection | summary, parameter_table, draw_network, layer_outputs; local helper implementations |
| Controlled regularisation experiment | Dropout 0 vs 0.2, selected on validation |
| Curves, predictions and failures | Seven exported figures with readings |
| Application | Four handmade housing scenarios |
| Save, reload and predict | Local weights, preprocessing and architecture; separate-process check |
| Data retrieval | Automatic pinned CSV download with checksum when missing; fresh download verified |
| No-training mode | Uses final model files and recorded histories; intermediate neural checkpoints no longer required |
| Documented execution | Sequential Python-cell execution with actual outputs; Sequential full execution with saved outputs; Jupyter kernel startup remains environment-dependent |
| torchinfo | Required in requirements.txt; not available locally, so the explicitly labeled summary fallback was used |
| README and model card | English, figures, method, results, limitations, usage and AI disclosure |
| Repository conventions | Required folders, .gitignore, requirements and MIT license; source data separately attributed |
| GitHub/Hugging Face publication | Existing published model pinned in publication.json |
| Remote download in a fresh session | Pinned model and data download plus prediction verified |

