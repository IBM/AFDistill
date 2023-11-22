import pytorch_lightning as pl
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import glob
import pickle
from Bio.PDB import PDBParser
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
import logging
import argparse
from torch.nn.utils.rnn import pad_sequence


class DataProcessor:
    def __init__(self, af_data, num_vocab_bins):
        self.af_data = af_data
        self.num_vocab_bins = num_vocab_bins

    def _get_af_pdbs(self):
        afs = glob.glob(os.path.join(self.af_data, 'AF*.pdb'))
        return afs

    def _get_sequence(self, structure):
        seq = ''
        for residue in list(structure.get_chains())[0]:
            for atom in residue:
                if atom.name == 'CA' and residue.get_resname() != 'HOH':
                    seq += residue.get_resname().title()[0]
        return seq

    def _get_plddt(self, structure):
        plddt = []
        for residue in list(structure.get_chains())[0]:
            for atom in residue:
                if atom.name == 'CA' and residue.get_resname() != 'HOH':
                    plddt.append(atom.get_bfactor()/100)
        return plddt

    def process(self):

        bin_width = 1 / self.num_vocab_bins
        bins = np.arange(start=0, stop=1.0, step=bin_width)

        parser = PDBParser(QUIET=True)

        af_pdbs = self._get_af_pdbs()

        results = []

        for i, af_pdb in enumerate(af_pdbs):

            if i % 1000 == 0:
                logging.info(f"data processor: processing {i}/{len(af_pdbs)}, collected {len(results)}")

            #extract AF PDB id
            basefile = os.path.basename(af_pdb)
            af_id = basefile[3:basefile[3:].find('-') + 3]

            structure = parser.get_structure('', af_pdb)

            # extract pLDDT values from PDB (they are located in place of bfactor)
            seq = self._get_sequence(structure)
            plddt_values = self._get_plddt(structure)

            assert len(seq) == len(plddt_values), f'Mismatch in {af_id} between seq length {len(seq)} and plddt len {len(plddt_values)}'

            plddt_classes = np.digitize(plddt_values, bins)

            results.append({'id': af_id, 'seq': seq, 'value': plddt_values, 'class': plddt_classes})

        logging.info(f"data processor: done")

        return results


class DistillDataModule(pl.LightningDataModule):

    def __init__(self,  tokenizer,
                        af_data,
                        precomputed_data,
                        balanced_training,
                        batch_size,
                        num_data_workers,
                        num_vocab_bins,
                        train_val_test_splits,
                        max_seq_len):
        super().__init__()

        self.prepare_data_per_node = False

        self.af_data = af_data
        self.precomputed_data = precomputed_data
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
            processor = DataProcessor(af_data=self.af_data, num_vocab_bins=self.num_vocab_bins)
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
            all_values = [np.mean(d['value']) for d in self.data_train]
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
        seqs, plddt_values, plddt_classes, ids = zip(*data)

        # Tokenize sequences
        seq_list = [' '.join(seq[:self.max_seq_len]) for seq in seqs]
        sequence_list_tok = self.tokenizer(seq_list, return_tensors='pt', padding=True, add_special_tokens=True)

        # tokens and attention mask
        input_ids = sequence_list_tok['input_ids']
        attention_mask = sequence_list_tok['attention_mask']

        logits_mask = []
        plddt_classes_mask = []
        plddt_classes_pad = []
        plddt_values_pad = []
        for cl, val in zip(plddt_classes, plddt_values):
            ones = torch.ones(len(cl[:self.max_seq_len]) + 2, dtype=torch.bool)
            ones[0] = ones[-1] = 0
            logits_mask.append(ones)
            plddt_classes_mask.append(torch.ones(len(cl[:self.max_seq_len]), dtype=torch.bool))
            plddt_classes_pad.append(torch.as_tensor(cl[:self.max_seq_len], dtype=torch.long))
            plddt_values_pad.append(torch.as_tensor(val[:self.max_seq_len]))

        plddt_classes_pad = pad_sequence(plddt_classes_pad).permute((1, 0))
        plddt_values_pad = pad_sequence(plddt_values_pad).permute((1, 0))
        logits_mask = pad_sequence(logits_mask).permute((1, 0))
        plddt_classes_mask = pad_sequence(plddt_classes_mask).permute((1, 0))

        return input_ids, attention_mask, logits_mask, plddt_classes_pad, plddt_classes_mask, ids, plddt_values_pad


# main function to run process method of DataProcessor class as a standalone code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--num_vocab_bins", type=int, default=50)
    parser.add_argument("--af_data", type=str, default='', required=True)
    parser.add_argument("--dump_results", type=str, default='', required=True)
    parser.add_argument("--log", choices=["debug", "info"], default="info", help="logging level")
    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper(), format='%(asctime)s - %(levelname)s - %(message)s')

    args_dict = vars(args).copy()
    args_dict.pop('dump_results')
    args_dict.pop('log')

    processor = DataProcessor(**args_dict)

    results = processor.process()

    pickle.dump(results, open(args.dump_results, 'wb'))
