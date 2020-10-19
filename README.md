# Synthesize, Execute and Debug: Learning to Repair for Neural Program Synthesis

This repo provides the code for experiments in the paper

> Kavi Gupta, Peter Ebert Christensen*, Xinyun Chen*, Dawn Song, <cite> Synthesize, Execute and Debug: Learning to Repair for Neural Program Synthesis, in NeurIPS 2020. (* Equal contribution) </cite>

Paper [[arXiv](https://arxiv.org/abs/2007.08095)]  

## Setup
This codebase uses Python 3. It was originally based on [the nearai program synthesis repo](https://github.com/nearai/program_synthesis)
  so it contains a lot of code not directly related to the SED paper.

1. (optionally) Create a virtualenv.
2. Install packages from `requirements.txt`: `pip install -r requirements.txt`
3. Install packages from `requirements2.txt`: `pip install -r requirements2.txt`
4. Install program_synthesis as package for development: `pip install -e .`

## Training models

Download the preprocessed dataset:
```
cd data
wget https://s3.us-east-2.amazonaws.com/karel-dataset/karel.tar.gz
tar xf karel.tar.gz
```

### Synthesis Model

Train a synthesis model (LRGL):
```
python3 train.py --dataset karel --model_type karel-lgrl \
  --num_epochs 100 --lr 1 --model_dir logdirs/synthesis
```

### Debugger Model

Generate training data for finetuning (LGRL). To create the LGRL-GD baseline, replace
  `--max_beam_trees 32` with `max_beam_trees 1`.

```
mkdir -p baseline

python3 program_synthesis/eval.py --model_type karel-lgrl \
  --dataset karel --max_beam_trees 32 \
  --model_dir logdirs/synthesis \
  --run-predict --predict-path baseline/lgrl_train.json \
  --evaluate-on-all --eval-train

python3 program_synthesis/eval.py --model_type karel-lgrl \
  --dataset karel --max_beam_trees 32 \
  --model_dir logdirs/synthesis \
  --run-predict --predict-path baseline/lgrl_val.json \
  --evaluate-on-all

```

Train a debugger model
```
# with TraceEmbed
python3 program_synthesis/train.py --dataset karel --model_type karel-lgrl-ref \
  --karel-mutate-ref --karel-mutate-n-dist 1,2,3 \
  --karel-trace-enc aggregate:conv_all_grids=True \
  --num_epochs 50  --lr 1 \
  --model_dir logdirs/debuggerTE

# without TraceEmbed
python3 program_synthesis/train.py --dataset karel --model_type karel-lgrl-ref \
  --karel-mutate-ref --karel-mutate-n-dist 1,2,3 \
  --karel-trace-enc none \
  --num_epochs 50 --lr 1 \
  --model_dir logdirs/debuggerWOTE
```

Finetune debugger models on the finetuning dataset (note that you might need to change `--pretrained-step 872000`
  to whatever the latest step of your debugger model is, you can see this by looking for the largest checkpoint
  in the `logdirs/debuggerTE` or `logdirs/debuggerWOTE` folder)

```
# with TraceEmbed
python3 program_synthesis/train.py --dataset karel --model_type karel-lgrl-ref \
  --karel-file-ref-val baseline/lgrl_val.json --karel-file-ref-train baseline/lgrl_train.json \
  --pretrained entire-model::logdirs/debuggerWOTE --pretrained-step 872000 \
  --karel-trace-enc aggregate:conv_all_grids=True \
  --num_epochs 50 --lr 0.0001 \
  --model_dir logdirs/debuggerWOTE-finetuned

# without TraceEmbed
python3 program_synthesis/train.py --dataset karel --model_type karel-lgrl-ref \
  --karel-file-ref-val baseline/lgrl_val.json --karel-file-ref-train baseline/lgrl_train.json \
  --pretrained entire-model::logdirs/debuggerWOTE --pretrained-step 872000 \
  --karel-trace-enc none \
  --num_epochs 50 --lr 0.0001 \
  --model_dir logdirs/debuggerWOTE-finetuned
```

## Evaluating Debugger Model

Run the following command for the MODEL_DIR for the model you want to execute.

```
python3 program_synthesis/eval.py --model_type karel-lgrl-ref \
  --dataset karel --max_beam_trees 64 \
  --iterative-search best_first --iterative-search-start-with-beams --evaluate-on-all
  --model_dir $LOGDIR \
  --karel-file-ref-val baseline/lgrl_val.json \
  --iterative-search-step-limit 25
```

Note that the results will not be out of 2500, the size of the evaluation dataset.
This is because we only consider in this evaluation the examples that the synthesizer
does not get correct. To get the end-to-end accuracy, take the number of correct
programs in this sample, and add it to the number of correct programs as reported by

```
python scripts/synthesizer_stats.py baseline/lgrl_val.json
```

Then divide by 2500.

## Citation

If you use the code in this repo, please cite the following paper:

```
@inproceedings{gupta2020synthesize,
  title={Synthesize, Execute and Debug: Learning to Repair for Neural Program Synthesis},
  author={Gupta, Kavi and Christensen, Peter Ebert and Chen, Xinyun and Song, Dawn},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

## Contact

If you have any difficulties with running these commands, please contact `$email@$domain.edu` where `email=kavig` and `domain=mit`.