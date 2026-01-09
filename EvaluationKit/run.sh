python LeanChecker.py \
    --verify_type proof \
    --dataset_path gen_data/physlean_test_deepseek_prover_7b_n16.json \
    --Lean4_server_url /home/hanning5/PhysProver/EvaluationKit/PhysLean \
    --save_path ./physlean_test_deepseek_prover_7b_n16_result.json \
    --ckpt_path ./physlean_test_deepseek_prover_7b_n16 \
    --batch_size 16 \
    --timeout 120 \
    --tmp_folder .temp_verify_folder_conjecture ##For temporary storage during verification