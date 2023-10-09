## Description 
---

In this repository one can find experiments for T5 and Whisper compression
from the paper:

`Sparse Finetuning for Inference Acceleration
of Large Language Models`.

### Repository structure

- `fsml` — the directory with implementation of pruning algorithms
- `translation` - translation example
- `automatic_speech_recognition` - automatic_speech_recognition example

### Requirements

Experiments were run within a following environment:

```
# conda packages

python                    3.10.0               h12debd9_5s
pytorch                   2.0.1           py3.10_cuda11.7_cudnn8.5.0_0    pytorch
pytorch-cuda              11.7                 h778d358_5    pytorch

# pip packages
torch                    2.0.1
torchaudio               2.0.2
torchvision              0.15.2
accelerate               0.22.0
datasets                 2.13.1
transformers             4.33.0
sacrebleu                1.5.0
rouge-score              0.1.2
librosa                  0.10.0.post2
```

To setup the enviroment run:

`conda create -y -n myenv python=3.10.0 pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia`

And then:

`pip install -r requirements.txt`

Optionally one can install wandb:

`pip install wandb`

We use `accelerate` for distributed training.
