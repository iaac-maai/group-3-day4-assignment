# Technical validation — 14 November 2026

Source reviewed: GitHub `945ac1e` (including the team's completed sections 6–9), compared with `Day_4/submissions/README.md` in `iaac-maai/ai-methods-1-2026`.

- Complete sequential notebook execution with `TRAIN=False`: passed all 11 code cells, including final model inspection, test evaluation and reload.
- Model architecture: 23 final encoded inputs and 3,649 parameters; validation training uses 22 inputs. Inspection now uses the preprocessing belonging to the network it displays.
- Expected test MLP MAE: 38,008.447 INR/month; Random Forest MAE: 38,044.928; 3,770 test rows, including 752 exact matches with training. These limitations remain disclosed.
- Full sequential retraining: passed all 11 code cells. Validation selected dropout 0 and epoch 28; final test metrics match the recorded values.
- Fresh Jupyter kernel execution: attempted but did not start producing notebook outputs and was interrupted. Sequential execution captures real cell outputs, but is not proof of VS Code/Jupyter kernel transport. This remains a host-specific limitation.
- Local environment: Python 3.12.14, torch 2.14.1+cpu, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, joblib 1.6.0. `torchinfo` is unavailable here; the notebook explicitly reports its local inspection fallback.
- Hugging Face repository exists publicly at revision `1b628c61c956fdd97543c1f21e258d023a041109`. Downloaded weights have SHA-256 `c9e2b04fbf66153f4e197a6f2d8e51522cb8abda55309889341015dd5f8696c4`, identical to the pre-existing local model. No remote files were changed.

The published card still lacks some of the detail required by the assignment (for example hardware/time and a complete per-feature preprocessing table); the expanded corrected card is supplied in `models/README.md`. It has not been uploaded over the published card. The two source ZIPs referenced in the earlier ChatGPT chat were not available in this Windows session; the current GitHub tree and original VS Code folder were used instead, with a local backup made before modifications.

Technical validation does not confirm team review, data licence provenance beyond the group's declared attribution, or acceptance of changes after the course submission snapshot.

Remote verification completed: all three pinned model files and the exact CSV were downloaded into an empty temporary directory. The dataset checksum and predictions for all four saved scenarios passed. Local Markdown links, missing-input rejection and saved notebook outputs also passed.
