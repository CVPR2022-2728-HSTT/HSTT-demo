# inside the container
horovodrun -np 4 python src/tasks/run_video_qa.py \
  --config src/configs/agqa.json \
  --do_inference 1 --output_dir storage/finetune/agqa_expm_balanced_dfs \
  --inference_model_step 119446 \
  --inference_batch_size 64 \
  --inference_n_clips 1 \
  --inference_vid_db storage/video_db/tokens \
  --inference_txt_db storage/txt_db/test_balanced_sampled.txt \
  --inference_metric 1 \
  --sampled 0