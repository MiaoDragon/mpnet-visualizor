python3 pointnet_plan_one_path.py --model_path ../PointMPnet_res/r2d/pointnet/simple_complex/ \
--learning_rate 0.01 \
--memory_strength 0.5 --n_memories 10000 \
--n_tasks 1 --device 1 --data_path /media/arclabdl1/HD1/Ahmed/rigid-body/dataset/ \
--start_epoch 2 --memory_type res --env_type r2d --world_size 20 \
--total_input_size 2806 --AE_input_size 2800 --mlp_input_size 34 --output_size 3 \
--env_idx 0 --path_idx 4012  --model_type simple --AE_type complex
