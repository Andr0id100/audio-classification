| filename               | Description                                                                          |
|------------------------|--------------------------------------------------------------------------------------|
| eda.ipynb              | Basic exploration of the dataset, label distribution, audio samples, etc.            |
| baseline.ipynb         | A simple classifier built using MFCC's as input to establish a baseline for the task |
| clap-inference.ipynb   | Inference on the zero shot model                                                     |
| results-analysis.ipynb | Detailed analysis of the performance of the different models with commments          |
| scripts/mfcc_train.py  | Baseline training on the provided dataset                                            |
| scripts/mfcc_eval.py   | Inference on the trained MFCC model                                                  |
| scripts/model.py       | Model architecuture for the baseline                                                 |
| scripts/hf_train.py    | Model fine-tuning on the provided dataset                                            |
| scripts/hf_train.py    | Inference on the fine tuned models                                                   |
| scripts/utils.py       | General utilities for data loading and evaluation                                    |
| preds/*.csv            | Model predictions on the test (and validation) split                                 |


The model weights can be downloaded from this [link](https://drive.google.com/file/d/1CEB_IXgtmZeBZuUETAcDDaFg1vF5XlAn/view?usp=sharing). They should be placed in the 'models' directory in the root directory.

Note that for ease of access, the eda and analysis notebook have also been added in the form of PDF files.
