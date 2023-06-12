#!/bin/bash

for seed in 1 2 3
do
    for beta in 0.1 0.2 0.3
    do
        MUJOCO_GL="osmesa"\
        LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl/:$LD_LIBRARY_PATH PYTHONPATH=.\
        python3 -u main.py setup=hipbmdp env=dmcontrol-walker-skill-distribution-v37\
        agent=distral setup.seed=$seed agent.distral_alpha=1.0 agent.distral_beta=$beta replay_buffer.batch_size=256
    done
done
