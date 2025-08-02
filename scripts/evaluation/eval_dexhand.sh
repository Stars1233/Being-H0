python -m beingvla.models.motion.m2m.aligner.eval_policy \
    --model-path /path/to/Being-H0-XXX-Align \
    --zarr-path /path/to/real-robot/data \
    --task_description "Put the little white duck into the cup." \
    --action-chunk-length 16