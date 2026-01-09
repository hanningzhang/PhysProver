# PhysProver

For the Reinforcement Learning (RL) experiments, please go to `/verl` folder. For the Generation and Evaluation experiments, please go to `/EvaluationKit` folder.

## Evaluation

Go to `/EvaluationKit`

Please first install the PhysLean environment from `https://github.com/HEPLean/PhysLean/tree/v4.20.0`.

Specifically, please follow the [Installation Guide](https://physlean.com/GettingStarted) in the current directory.

For other evaluation environments, please use the `verl` environment as we mainly need `vllm` package.

The `/data` folder contains the prompts for DeepSeek Prover, Kimina Prover, and Goedel Prover.

For generation, please run `infer.sh`, and below is an example:
```
mkdir -p gen_data

CUDA_VISIBLE_DEVICES=0 python vllm_infer.py \
  --n 16 \
  --temperature 1.0 \
  --dataset data/physlean_test_deepseek.json \
  --model deepseek-ai/DeepSeek-Prover-V2-7B \
  --output_file gen_data/physlean_test_deepseek_prover_7b_n16.json 
```

For evaluation, please run `run.sh`, and below is an example:
```
python LeanChecker.py \
    --verify_type proof \
    --dataset_path gen_data/physlean_test_deepseek_prover_7b_n16.json \
    --Lean4_server_url /home/hanning5/PhysProver/EvaluationKit/PhysLean \    ##Please change into your own path
    --save_path ./physlean_test_deepseek_prover_7b_n16_result.json \
    --ckpt_path ./physlean_test_deepseek_prover_7b_n16 \
    --batch_size 16 \
    --timeout 120 \
    --tmp_folder .temp_verify_folder_conjecture       ##For temporary storage during verification
```

After this step, run `eval_acc.py` to get the accuracy:
```
python eval_acc.py --save_folder ./physlean_test_deepseek_prover_7b_n16
```

## Reinforcement Learning using Verl

Go to `/verl`

We use `verl v0.6.0` version and please install from source.

Process data:
```
python examples/data_preprocess/physicslean.py --local_dataset_path physicslean_ds_curri 
```

We have made some changes.

We update `verl/utils/dataset/rl_dataset.py` to remove the chat template because we already have it in our dataset.

We update `verl/workers/reward_manager/naive.py` to allow multi-process Lean4 evaluation.

We update files in `verl/utils/reward_score/` to support Lean4 verification.

Please install `PhysLean` in `verl/utils/reward_score/` directory following [Installation Guide](https://physlean.com/GettingStarted).

Please also change `line 152` in `verl/utils/reward_score/physicslean.py` to your own path.

Then you can run
```
bash examples/grpo_trainer/run_ds_prover.sh
```

