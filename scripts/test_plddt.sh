#!/usr/bin/env bash
cd ../
python main.py --type plddt \
               --run test \
               --version 1 \
               --af_data /path/to/swissprot_alphafold_v2 \
               --precomputed_data /path/to/plddt_precomputed.pkl \
               --batch_size 10 \
               --num_data_workers 0 \
               --checkpoint_model_id -1 \
               --root_dir output \
               --gpus 1 \
               --bar 10 \


