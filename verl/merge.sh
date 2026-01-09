# python scripts/legacy_model_merger.py merge \
#     --backend fsdp \
#     --local_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_1e-6_bs256_test/global_step_5/actor \
#     --target_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_1e-6_bs256_test/global_step_5/huggingface

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_1e-6_bs256_test/global_step_10/actor \
#     --target_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_1e-6_bs256_test/global_step_10/actor/huggingface

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_grpo_physlean/ds_prover_grpo_1e-6_bs256_test/global_step_10/actor \
#     --target_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_grpo_physlean/ds_prover_grpo_1e-6_bs256_test/global_step_10/actor/huggingface

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_ppo_1e-6_bs256_newsplit_noapply_curri/global_step_55/actor \
    --target_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_ppo_1e-6_bs256_newsplit_noapply_curri/global_step_55/actor/huggingface

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_ppo_1e-6_bs256_newsplit_noapply_curri/global_step_60/actor \
    --target_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_ppo_1e-6_bs256_newsplit_noapply_curri/global_step_60/actor/huggingface

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_ppo_1e-6_bs256_newsplit_noapply_curri/global_step_65/actor \
    --target_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_ppo_1e-6_bs256_newsplit_noapply_curri/global_step_65/actor/huggingface

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_ppo_1e-6_bs256_newsplit_noapply_curri/global_step_70/actor \
    --target_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_ppo_1e-6_bs256_newsplit_noapply_curri/global_step_70/actor/huggingface


# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_1e-6_bs256_test/global_step_15/actor \
#     --target_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_1e-6_bs256_test/global_step_15/actor/huggingface
# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_1e-6_bs256_test/global_step_20/actor \
#     --target_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_1e-6_bs256_test/global_step_20/actor/huggingface


# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_1e-6_bs256_test/global_step_25/actor \
#     --target_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_1e-6_bs256_test/global_step_25/actor/huggingface
# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_1e-6_bs256_test/global_step_30/actor \
#     --target_dir /taiga/illinois/eng/cs/tozhang/hanning5/verl/checkpoints/verl_ppo_physlean/ds_prover_1e-6_bs256_test/global_step_30/actor/huggingface
