#!/usr/bin/env bash
cd ../
python main.py --type tm \
               --run test \
               --version 0 \
               --af_data /path/to/swissprot_alphafold_v2 \
               --pdb_uniprot_map /path/to/pdb_chain_uniprot.lst \
               --tm_exec /path/to/TMalign_cpp \
               --cached_pdbs /path/to/cached_pdbs \
               --precomputed_data /path/to/tm_precomputed.pkl \
               --pdb_dataset /path/to/pdb \
               --batch_size 10 \
               --num_data_workers 4 \
               --checkpoint_model_id -1 \
               --root_dir output \
               --gpus 1 \
               --bar 10 \


