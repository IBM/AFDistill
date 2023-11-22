#!/usr/bin/env bash
cd ../
python main.py --type tm \
               --run train \
               --version 0 \
               --af_data /path/to/swiss/data \
               --pdb_uniprot_map /path/to/pdb_chain_uniprot.tsv \
               --tm_exec /path/to/TMalign_cpp \
               --cached_pdbs /path/to/cached_pdbs \
               --precomputed_data /path/to/tm_precomputed.pkl \
               --pdb_dataset /path/to/pdbs \
               --learning_rate 1e-6 \
               --batch_size 10 \
               --balanced_training 0 \
               --num_data_workers 4 \
               --checkpoint_model_id -1 \
               --focal_loss_gamma 3 \
               --root_dir output \
               --gpus 1 \
               --enable_checkpointing True \
               --num_sanity_val_steps 0 \
               --fast_dev_run 0 \
               --overfit_batches 0 \
               --limit_train_batches 1.0 \
               --limit_val_batches 1.0 \
               --log_every_n_steps 100 \
               --checkpoint_step_frequency 100 \
               --accumulate_grad_batches 1 \
               --val_check_interval 100 \
               --max_time 00:23:00:00 \
               --bar 10 \


