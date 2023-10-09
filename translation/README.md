## Description

The training and script is adopted from HuggingFace [translation example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation). 

### Directory structure

- `configs` — configs specifying properties of pruning algorithm
- `scripts` - shell scripts launching training script `run_translation.py`
- `run_translation.py` - main training script

- `scripts/train.sh` - dense training script
- `scripts/run_gradual_pruning.sh` - pruning with CE only
- `scripts/run_gradual_pruning_standard_kd.sh` - standard KD
- `scripts/run_gradual_pruning_squarehead_kd.sh` - squarehead pruning