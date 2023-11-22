import pytorch_lightning as pl
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import glob
import pickle
from Bio.PDB import PDBParser
import pandas as pd
import subprocess
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
import logging
import re
import argparse
import multiprocessing


class DataProcessor:
    def __init__(self,
                 af_data,
                 cached_pdbs,
                 pdb_uniprot_map,
                 tm_exec,
                 num_vocab_bins,
                 num_jobs=1,
                 pdb_dataset=''):
        self.af_data = af_data
        self.pdb_uniprot_map = pdb_uniprot_map
        self.cached_pdbs = cached_pdbs
        self.pdb_dataset = pdb_dataset
        self.num_vocab_bins = num_vocab_bins
        self.tm_exec = tm_exec
        self.num_jobs = num_jobs

    def _get_af_pdbs(self):
        afs = glob.glob(os.path.join(self.af_data, 'AF*.pdb'))
        return afs

    def _get_pdb_uniprot_map(self):
        pdb_uniprot_map = pd.read_csv(self.pdb_uniprot_map, delimiter="\t", skiprows=1)
        pdb_uniprot_map.set_index("SP_PRIMARY", inplace=True)
        return pdb_uniprot_map

    def _get_sequence(self, structure):
        seq = ''
        for residue in list(structure.get_chains())[0]:
            for atom in residue:
                if atom.name == 'CA' and residue.get_resname() != 'HOH':
                    # try:
                        seq += residue.get_resname().title()[0]
                    # except:
                    #     break
        return seq

    # adapted from https://github.com/microsoft/foldingdiff
    def _run_tmalign(self, query, reference, tm_exec):
        assert os.path.isfile(query)
        assert os.path.isfile(reference)

        # The command
        cmd = f"{tm_exec} {query} {reference}"
        try:
            devnull = open(os.devnull, 'w')
            output = subprocess.check_output(cmd, shell=True, stderr=devnull)
            devnull.close()
        except subprocess.CalledProcessError:
            logging.debug(f"TM scoring failed on query={query} and reference={reference}")
            return None

        # Parse output
        score_lines = []
        for line in output.decode().split("\n"):
            if line.startswith("TM-score"):
                score_lines.append(line)

        # Fetch the chain number
        key_getter = lambda s: re.findall(r"Chain_[12]{1}", s)[0]
        score_getter = lambda s: float(re.findall(r"=\s+([0-9.]+)", s)[0])
        results_dict = {key_getter(s): score_getter(s) for s in score_lines}

        try:
            result = results_dict["Chain_2"]
        except Exception as e:
            logging.debug(f"Parsing TM results failed on query={os.path.basename(query)} and reference={os.path.basename(reference)}")
            return None

        return result

    def process(self, jobid=0):

        bin_width = 1 / self.num_vocab_bins
        bins = np.arange(start=0, stop=1.0, step=bin_width)

        parser = PDBParser(QUIET=True)

        pdb_uniprot_map = self._get_pdb_uniprot_map()

        af_pdbs = self._get_af_pdbs()

        if self.num_jobs > 1:
            splits = np.array_split(af_pdbs, self.num_jobs)
            af_pdbs = splits[jobid]

        logging.info(f"data processor {jobid} working on {len(af_pdbs)} files")

        results = []

        for i, af_pdb in enumerate(af_pdbs):

            if i % 100 == 0:
                logging.info(f"data processor {jobid}: processing {i}/{len(af_pdbs)}, collected {len(results)}")

            #extract AF PDB id
            basefile = os.path.basename(af_pdb)
            af_id = basefile[3:basefile[3:].find('-') + 3]

            if af_id not in pdb_uniprot_map.index:
                continue

            # find matching PDBs
            found_pdbs = pdb_uniprot_map .loc[[af_id]]
            pdb_ids = found_pdbs['PDB'].unique()
            found_pdbs.set_index('PDB', inplace=True)

            # score each matching PDB with AF_PDB, and select highest TM score
            tms = []
            for pdb_id in pdb_ids:

                # check if this pdb is in PDB dataset
                if self.pdb_dataset:
                    matched_pdb = os.path.join(self.pdb_dataset, f'pdb{pdb_id}.ent')

                    if not os.path.exists(matched_pdb):
                        # check if this pdb is in earlier cached directory, if not - download it
                        matched_pdb = os.path.join(self.cached_pdbs, f'{pdb_id}.pdb')

                        if not os.path.exists(matched_pdb) or os.path.getsize(matched_pdb) == 0:
                            os.system(f'wget -O {matched_pdb} https://files.rcsb.org/download/{pdb_id}.pdb > /dev/null 2>&1')

                            if not os.path.exists(matched_pdb) or os.path.getsize(matched_pdb) == 0:
                                logging.debug(f'data processor {jobid}: PDB {pdb_id} could not be downloaded')
                                continue

                tmscore = self._run_tmalign(query=matched_pdb, reference=af_pdb, tm_exec=self.tm_exec)
                if tmscore:
                    tms.append(tmscore)

            if len(tms) == 0:
                continue

            # select best TM score and find corresponding bin
            tm_value = max(tms)
            tm_class = np.digitize(tm_value, bins)

            seq = self._get_sequence(parser.get_structure('', af_pdb))

            results.append({'id': af_id, 'seq': seq, 'value': tm_value, 'class': tm_class})

        logging.info(f"data processor {jobid}: done")

        return results


class DistillDataModule(pl.LightningDataModule):

    def __init__(self,  tokenizer,
                        af_data,
                        cached_pdbs,
                        precomputed_data,
                        pdb_uniprot_map,
                        tm_exec,
                        balanced_training,
                        batch_size,
                        num_data_workers,
                        num_vocab_bins,
                        train_val_test_splits,
                        max_seq_len):
        super().__init__()

        self.prepare_data_per_node = False

        self.af_data = af_data
        self.cached_pdbs = cached_pdbs
        self.precomputed_data = precomputed_data
        self.pdb_uniprot_map = pdb_uniprot_map
        self.tm_exec = tm_exec
        self.tokenizer = tokenizer
        self.balanced_training = balanced_training
        self.batch_size = batch_size
        self.num_data_workers = num_data_workers
        self.num_vocab_bins = num_vocab_bins
        self.train_val_test_splits = train_val_test_splits
        self.max_seq_len = max_seq_len

    def prepare_data(self):

        if os.path.exists(self.precomputed_data):
            logging.info(f'Loading precomputed dataset from {self.precomputed_data}')
            self.data = pickle.load(open(self.precomputed_data, 'rb'))
        else:
            logging.info(f'Building dataset from scratch')
            processor = DataProcessor(af_data=self.af_data,
                                      cached_pdbs=self.cached_pdbs,
                                      pdb_uniprot_map=self.pdb_uniprot_map,
                                      tm_exec=self.tm_exec,
                                      num_vocab_bins=self.num_vocab_bins)
            self.data = processor.process()
            pickle.dump(self.data, open(self.precomputed_data, 'wb'))

        # split into training, validation and test splits
        # normalize, just in case
        train_val_test_splits = np.array(self.train_val_test_splits)/sum(self.train_val_test_splits)
        train_ratio, val_ratio, test_ratio = train_val_test_splits

        self.data_train, valtest = train_test_split(self.data, train_size=train_ratio, random_state=42)

        val_ratio = val_ratio / (val_ratio + test_ratio)
        self.data_val, self.data_test = train_test_split(valtest, train_size=val_ratio, random_state=42)

    def setup(self, stage=None):

        if stage == 'fit':
            self.dataset_train = DistillDataset(num_vocab_bins=self.num_vocab_bins,
                                                data=self.data_train,
                                                max_seq_len=self.max_seq_len,
                                                tokenizer=self.tokenizer)
        elif stage == 'validate':
            self.dataset_dev = DistillDataset(num_vocab_bins=self.num_vocab_bins,
                                              data=self.data_val,
                                              max_seq_len=self.max_seq_len,
                                              tokenizer=self.tokenizer)
        elif stage == 'test':
            self.dataset_test = DistillDataset(num_vocab_bins=self.num_vocab_bins,
                                               data=self.data_test,
                                               max_seq_len=self.max_seq_len,
                                               tokenizer=self.tokenizer)
        else:
            return

    def train_dataloader(self):

        def class_imbalance_sampler(labels):
            class_count = torch.bincount(labels)
            class_weighting = 1. / class_count.float()
            sample_weights = class_weighting[labels]
            sampler = WeightedRandomSampler(sample_weights, len(labels))
            return sampler

        if self.balanced_training:

            all_values = [d['value'] for d in self.data_train]
            _, bins = np.histogram(all_values, bins='fd', density=False)
            tm_labels = np.digitize(all_values, bins)

            sampler = class_imbalance_sampler(torch.tensor(tm_labels, dtype=torch.long))

            dl = DataLoader(self.dataset_train,
                       batch_size=self.batch_size,
                       collate_fn=self.dataset_train._collate_fn,
                       num_workers=self.num_data_workers,
                       shuffle=False, pin_memory=True, sampler=sampler)
        else:
            dl = DataLoader(self.dataset_train,
                            batch_size=self.batch_size,
                            collate_fn=self.dataset_train._collate_fn,
                            num_workers=self.num_data_workers,
                            shuffle=True, pin_memory=True)

        return dl

    def val_dataloader(self):
        return DataLoader(self.dataset_dev,
                          batch_size=self.batch_size,
                          collate_fn=self.dataset_train._collate_fn,
                          num_workers=self.num_data_workers,
                          shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                          batch_size=self.batch_size,
                          collate_fn=self.dataset_test._collate_fn,
                          num_workers=self.num_data_workers,
                          shuffle=False, pin_memory=True)


class DistillDataset(Dataset):

    def __init__(self, data, num_vocab_bins, max_seq_len, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_vocab_bins = num_vocab_bins

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item['seq'], item['value'], item['class'], item['id']

    def _collate_fn(self, data):
        seqs, tm_values, tm_classes, ids = zip(*data)
        seq_list = [' '.join(seq[:self.max_seq_len]) for seq in seqs]

        batch_size = len(data)

        # Tokenize sequences
        sequence_list_tok = self.tokenizer(seq_list, return_tensors='pt', padding=True, add_special_tokens=True)

        # Sequences, attention masks, and logits mask
        input_ids = sequence_list_tok['input_ids']
        attention_mask = sequence_list_tok['attention_mask']
        logits_mask = torch.ones(batch_size, self.num_vocab_bins).long()

        # Convert to tensors
        tm_classes = torch.tensor(tm_classes, dtype=torch.long)
        tm_classes_mask = torch.ones(batch_size, dtype=torch.long)
        tm_values = torch.as_tensor(tm_values)

        return input_ids, attention_mask, logits_mask, tm_classes, tm_classes_mask, ids, tm_values


# main function to run process method of DataProcessor class as a standalone code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--num_jobs", type=int, default=5)
    parser.add_argument("--num_vocab_bins", type=int, default=50)
    parser.add_argument("--af_data", type=str, default='', required=True)
    parser.add_argument("--cached_pdbs", type=str, default='', required=True)
    parser.add_argument("--pdb_dataset", type=str, default='')
    parser.add_argument("--pdb_uniprot_map", type=str, default='', required=True)
    parser.add_argument("--tm_exec", type=str, default='', required=True)
    parser.add_argument("--dump_results", type=str, default='', required=True)
    parser.add_argument("--log", choices=["debug", "info"], default="info", help="logging level")
    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper(), format='%(asctime)s - %(levelname)s - %(message)s')

    args_dict = vars(args).copy()
    args_dict.pop('dump_results')
    args_dict.pop('log')

    processor = DataProcessor(**args_dict)

    if args.num_jobs > 1:
        # Process chunks in parallel using a process pool
        with multiprocessing.Pool(processes=args.num_jobs) as pool:
            chunk_results = pool.map(processor.process, list(range(args.num_jobs)))

        # Flatten the list of results
        results = [result for chunk_result in chunk_results for result in chunk_result]
    else:
        results = processor.process()

    pickle.dump(results, open(args.dump_results, 'wb'))
