python -m beingvla.inference.vla_internvl_inference \
    --model_path /path/to/your/model \
    --motion_code_path "/path/to/wrist+/path/to/finger" \
    --input_image ./playground/unplug_airpods.jpg \
    --task_description "unplug the charging cable from the AirPods" \
    --hand_mode both \
    --num_samples 3 \
    --num_seconds 4 \
    --enable_render true \
    --output_dir ./work_dirs/