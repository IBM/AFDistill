from pytorch_lightning import loggers as pl_loggers
import argparse
import pytorch_lightning as pl
import os
from pytorch_lightning.utilities import rank_zero_only
from model.litdistiller import LitDistiller
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.timer import Timer
import numpy as np
from data.dataset_tm import DistillDataModule as DistillDataModuleTM
from data.dataset_plddt import DistillDataModule as DistillDataModulePLDDT
import logging
from transformers import BertTokenizer


class CheckpointOnStep(pl.Callback):
    def __init__(self, checkpoint_dir, checkpoint_step_frequency, duration, save_step=False):
        self.checkpoint_step_frequency = checkpoint_step_frequency
        self.save_step = save_step
        self.duration = duration
        self.checkpoint_dir = checkpoint_dir

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step

        if step % self.checkpoint_step_frequency == 0 and self.checkpoint_step_frequency > 0:
            if self.save_step:
                ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}.ckpt")
                trainer.save_checkpoint(ckpt_path)

            # save it also as the last model
            ckpt_path = os.path.join(self.checkpoint_dir, "checkpoint_last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    # hack to exit training and validation once training timer is done
    @rank_zero_only
    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.should_stop:
            print('on_validation_start: Training done')
            ckpt_path = os.path.join(self.checkpoint_dir, "checkpoint_last.ckpt")
            trainer.save_checkpoint(ckpt_path)
            print('Saved checkpoint')
            raise KeyboardInterrupt

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        try:
            timer = [c for c in trainer.callbacks if isinstance(c, Timer)][0]
            # import pdb; pdb.set_trace()
            if timer.time_elapsed() >= timer._duration:
                timer._duration = timer.time_elapsed() + self.duration
        except:
            return


def main(args):

    logging.basicConfig(level=args.loging_level.upper(), format='%(asctime)s - %(levelname)s - %(message)s')

    # pl.seed_everything(42)

    args.eval_dir = os.path.join(args.root_dir, f'{args.type}_v{args.version}')
    args.checkpoint_dir = os.path.join(args.eval_dir, 'checkpoints')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    TB = pl_loggers.TensorBoardLogger(args.root_dir, name='', version=f'{args.type}_v{args.version}', default_hp_metric=False)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model, cache=args.bert_cache, do_lower_case=False)

    if args.checkpoint_model_id < 0:
        checkpoint_model_path = os.path.join(args.checkpoint_dir, 'checkpoint_last.ckpt')
    else:
        checkpoint_model_path = os.path.join(args.checkpoint_dir, f"model-step={args.checkpoint_model_id}.ckpt")
    if not os.path.exists(checkpoint_model_path):
        checkpoint_model_path = None

    if args.run == 'inference':
        assert os.path.exists(checkpoint_model_path), 'Provided checkpoint does not exists, cannot run the inference'

        model = LitDistiller.load_from_checkpoint(checkpoint_path=checkpoint_model_path).cuda()

        val_pred = model.single_inference(args.single_input, tokenizer)

        if args.type == 'tm':
            logging.info(f'Predicted TM score: {val_pred}')
        else:
            logging.info(f'Predicted pLDDT scores per residue: {val_pred}')
            logging.info(f'Sequence pLDDT score: {np.mean(val_pred)}')

    else:

        if args.type == 'tm':
            dm = DistillDataModuleTM(af_data=args.af_data,
                                     cached_pdbs=args.cached_pdbs,
                                     precomputed_data=args.precomputed_data,
                                     pdb_uniprot_map=args.pdb_uniprot_map,
                                     tm_exec=args.tm_exec,
                                     tokenizer=tokenizer,
                                     balanced_training=args.balanced_training,
                                     batch_size=args.batch_size,
                                     num_data_workers=args.num_data_workers,
                                     num_vocab_bins=args.num_vocab_bins,
                                     train_val_test_splits=args.train_val_test_splits,
                                     max_seq_len=args.max_seq_len)
        elif args.type == 'plddt':
            dm = DistillDataModulePLDDT(af_data=args.af_data,
                                        precomputed_data=args.precomputed_data,
                                        tokenizer=tokenizer,
                                        balanced_training=args.balanced_training,
                                        batch_size=args.batch_size,
                                        num_data_workers=args.num_data_workers,
                                        num_vocab_bins=args.num_vocab_bins,
                                        train_val_test_splits=args.train_val_test_splits,
                                        max_seq_len=args.max_seq_len)
        else:
            raise ValueError("Invalid dataset type. Allowed types are 'tm' and 'plddt'.")

        if args.run == 'train':

            run_duration = -1
            if hasattr(args, 'max_time') and args.max_time:
                arg = args.max_time.split(':')
                days, hrs, mnts, secs = int(arg[0]), int(arg[1]), int(arg[2]), int(arg[3])
                run_duration = secs + 60 * mnts + 3600 * hrs + 86400 * days

            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                dirpath=args.checkpoint_dir,
                filename='{epoch}-{step}-{val_loss:.2f}',
                save_top_k=2,
                mode='min',
                auto_insert_metric_name=False
            )

            dm.prepare_data()
            dm.setup(stage='fit')
            dm.setup(stage='validate')

            trainer = pl.Trainer.from_argparse_args(args,
                                                    default_root_dir=args.root_dir,
                                                    logger=TB,
                                                    callbacks=[
                                                        CheckpointOnStep(args.checkpoint_dir,
                                                                         args.checkpoint_step_frequency,
                                                                         run_duration),
                                                        checkpoint_callback,
                                                        RichProgressBar(args.bar)])

            model = LitDistiller(type=args.type,
                                 start_from_scratch=args.start_from_scratch,
                                 pretrained_model=args.pretrained_model,
                                 cache_dir=args.bert_cache,
                                 num_vocab_bins=args.num_vocab_bins,
                                 focal_loss_gamma=args.focal_loss_gamma,
                                 learning_rate=args.learning_rate,
                                 alphabet_downstream=args.alphabet_downstream)

            trainer.fit(model=model, datamodule=dm, ckpt_path=checkpoint_model_path)

        elif args.run == 'test':
            assert os.path.exists(checkpoint_model_path), 'Provided checkpoint does not exists, cannot run the test'

            model = LitDistiller.load_from_checkpoint(checkpoint_path=checkpoint_model_path)

            dm.prepare_data()
            dm.setup(stage='test')

            trainer = pl.Trainer.from_argparse_args(args, logger=TB)

            trainer.test(model, datamodule=dm)

        else:
            raise ValueError("Invalid run mode. Allowed modes are 'train', 'test', and 'single_inference'.")


if __name__ == "__main__":
    
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Arguments')

    parser.add_argument("--run", type=str, default='train')
    parser.add_argument("--loging_level", choices=["debug", "info"], default="info", help="logging level")
    parser.add_argument("--single_input", type=str, default='MVQIKAAALAVLFASNVLANP')
    parser.add_argument('--root_dir', type=str, default='output')
    parser.add_argument('--pretrained_model', type=str, default='Rostlab/prot_bert')
    parser.add_argument('--version', type=str, default='0')
    parser.add_argument('--precomputed_data', type=str, default='')
    parser.add_argument('--af_data', type=str, default='')
    parser.add_argument('--pdb_dataset', type=str, default='')
    parser.add_argument('--pdb_uniprot_map', type=str, default='')
    parser.add_argument('--tm_exec', type=str, default='')
    parser.add_argument('--cached_pdbs', type=str, default='')
    parser.add_argument('--bert_cache', type=str, default='cache')
    parser.add_argument('--num_data_workers', type=int, default=4)
    parser.add_argument('--train_val_test_splits', type=int, nargs = '+', default=[0.8, 0.1, 0.1])
    parser.add_argument('--checkpoint_step_frequency', type=int, default=-1)
    parser.add_argument('--checkpoint_model_id', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--max_seq_len', default=598, type=int)
    parser.add_argument('--start_from_scratch', default=0, type=int)
    parser.add_argument('--num_vocab_bins', default=50, type=int)
    parser.add_argument('--type', default='plddt', type=str)
    parser.add_argument("--balanced_training", type=int, default=1)
    parser.add_argument("--focal_loss_gamma", type=float, default=1)
    parser.add_argument('--bar', default=0, type=int)
    parser.add_argument('--alphabet_downstream', default=None, type=str)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
