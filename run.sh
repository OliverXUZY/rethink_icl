python llm_test.py \
    --dataset "glue-sst2" \
    --base_model_path "meta-llama/Llama-2-7b-chat-hf" \
    --method direct \
    --out_dir 'out/test/llama-2-7b-chat' \
    --do_zeroshot \
    --test_batch_size 8 \
    --num_gpus 2 \
    --log_file out/llama-2-7b-chat/log_sst2_dem.txt \
    --use_demonstrations --k 16 --seed 13,21,42,87,100 2>&1 | tee out/test/output.txt

    # --log_file out/llama-2-7b-chat/log_sst2_dem.txt
