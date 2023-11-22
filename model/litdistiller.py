import torch
import pytorch_lightning as pl
from transformers import BertForMaskedLM
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
import copy
import transformers
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class FocalLoss(torch.nn.modules.loss._WeightedLoss):

    def __init__(self, gamma, weight=None):
        super(FocalLoss, self).__init__(weight)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        # input: batch_size x num_vocab_bins
        # target: batch_size
        ce_loss = torch.nn.functional.cross_entropy(input, target, weight=self.weight, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = ((1 - p) ** self.gamma * ce_loss)

        return focal_loss


class LitDistiller(pl.LightningModule):
    def __init__(self,
                 type,
                 start_from_scratch,
                 pretrained_model,
                 cache_dir,
                 num_vocab_bins,
                 focal_loss_gamma,
                 learning_rate,
                 alphabet_downstream):
        super().__init__()

        self.save_hyperparameters()

        self.type = type
        self.start_from_scratch = start_from_scratch
        self.pretrained_model = pretrained_model
        self.cache_dir = cache_dir
        self.num_vocab_bins = num_vocab_bins
        self.focal_loss_gamma = focal_loss_gamma
        self.learning_rate = learning_rate
        self.alphabet_downstream = alphabet_downstream

        bin_width = 1 / self.num_vocab_bins
        # used as buffer to place on proper device
        self.register_buffer('bin_centers', torch.arange(start=0.5 * bin_width, end=1, step=bin_width), persistent=False)

        if self.start_from_scratch:
            config = transformers.AutoConfig.from_pretrained(self.pretrained_model, cache_dir=self.cache_dir)
            self.model = BertForMaskedLM(config)
        else:
            self.model = BertForMaskedLM.from_pretrained(self.pretrained_model, cache_dir=self.cache_dir)

        LMhead_config = copy.deepcopy(self.model.config)
        LMhead_config.vocab_size = self.num_vocab_bins
        self.model.cls = BertOnlyMLMHead(LMhead_config)

        self.focal = FocalLoss(gamma=self.focal_loss_gamma)
        self.ce = torch.nn.CrossEntropyLoss()

        if self.alphabet_downstream:
            self.register_buffer('EMB', self.model.bert.get_input_embeddings().weight[5:25].detach().clone(), persistent=False)
            self.register_buffer('CLS', self.model.bert.get_input_embeddings().weight[2:3].detach().clone(), persistent=False)
            self.register_buffer('SEP', self.model.bert.get_input_embeddings().weight[3:4].detach().clone(), persistent=False)
            self.register_buffer('PAD', self.model.bert.get_input_embeddings().weight[0:1].detach().clone(), persistent=False)

            alphabet_distill = 'LAGVESIKRDTPNQFYMHCW'

            permutation_indices = torch.tensor([alphabet_distill.index(c) for c in self.alphabet_downstream], dtype=torch.long)
            self.register_buffer('PERM', torch.eye(permutation_indices.size(0))[permutation_indices], persistent=False)

    def forward(self, seq=None, seq_mask=None, seq_embeds=None):
        out = self.model(input_ids=seq, attention_mask=seq_mask, inputs_embeds=seq_embeds)
        logits_pred = out.logits

        if self.type == 'tm':
            logits_pred = logits_pred[:, 0, :]
        # else: plddt

        return logits_pred

    def training_step(self, batch, batch_idx):

        # seq: batch_size X seq_len
        # seq_mask: batch_size X seq_len
        # logits_mask: batch_size X num_vocab_bins (50)
        # cl: batch_size
        # cl_mask: batch_size
        # seq_id: batch_size
        # val: batch_size

        seq, seq_mask, logits_mask, cl, cl_mask, seq_id, val = batch

        cl = torch.clamp(cl, 0, self.num_vocab_bins-1)

        logits_pred = self(seq, seq_mask)

        if self.type == 'plddt':
            logits_pred = logits_pred[logits_mask]
            cl = cl[cl_mask]

        if self.focal_loss_gamma > 0:
            loss_ce = self.focal(logits_pred, cl).mean()
        else:
            loss_ce = self.ce(logits_pred, cl)

        self.log_dict({'train_loss': loss_ce}, on_step=True, on_epoch=True, logger=True, batch_size=seq.size(0), prog_bar=False)

        return loss_ce

    def validation_step(self, batch, batch_idx):

        seq, seq_mask, logits_mask, cl, cl_mask, seq_id, val = batch

        cl = torch.clamp(cl, 0, self.num_vocab_bins-1)

        logits_pred = self(seq, seq_mask)

        if self.type == 'plddt':
            loss_ce = self.ce(logits_pred[logits_mask], cl[cl_mask])
        else:
            loss_ce = self.ce(logits_pred, cl)

        self.log_dict({'val_loss': loss_ce}, on_step=True, on_epoch=True, logger=True, batch_size=seq.size(0), prog_bar=False)

        if self.type == 'tm':
            pred_vals = torch.sum(torch.nn.functional.softmax(logits_pred, -1) * self.bin_centers[None, :], axis=-1).detach().cpu().numpy().tolist()
            true_vals = val.detach().cpu().numpy().tolist()
        else:
            pred_vals = []
            true_vals = []
            for l, m in zip(logits_pred, logits_mask):
                pred_vals.append(torch.sum(torch.nn.functional.softmax(l[m], -1) * self.bin_centers[None, :], axis=-1).detach().cpu().numpy().tolist())
            for v, m in zip(val, cl_mask):
                true_vals.append(v[m].cpu().numpy().tolist())

            if batch_idx == 0:
                for i, (p, t, s_id) in enumerate(zip(pred_vals, true_vals, seq_id)):

                    fig = plt.figure(figsize=(10, 5))
                    ax = fig.add_subplot()
                    ax.plot(t, 'r-', label='pLDDT_true')
                    ax.plot(p, 'g-', label='pLDDT_pred')
                    ax.set_ylim(0, 1)
                    ax.legend(loc=3, prop={'size': 15})
                    ax.tick_params(axis='both', which='major', labelsize=15)
                    ax.set_title(s_id, fontsize=15)
                    self.logger.experiment.add_figure(f'val_pLDDT_{i}', fig, self.global_step)

        return pred_vals, true_vals

    def validation_epoch_end(self, outputs):
        pred_all = []
        true_all = []
        for out in outputs:
            if self.type == 'tm':
                pred_all += out[0]
                true_all += out[1]
                label = "TM"
            else:
                for p, t in zip(out[0], out[1]):
                    pred_all.append(sum(p)/len(p))
                    true_all.append(sum(t)/len(t))
                label = "pLDDT"

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(true_all, pred_all, 'k.')
        ax[0].set_ylabel('pred')
        ax[1].set_ylabel('true')
        sns.kdeplot(data=pd.DataFrame({'true': true_all, 'pred': pred_all}), x="true", y="pred", ax=ax[1], fill=True, levels=10, thresh=.1)
        ax[0].set_title(f'Scatter plot of {label}')
        ax[1].set_title(f'Density plot of {label}')
        self.logger.experiment.add_figure(f'val_{label}_all', fig, self.global_step)

    def test_step(self, batch, batch_idx):

        seq, seq_mask, logits_mask, cl, cl_mask, seq_id, val = batch

        cl = torch.clamp(cl, 0, self.num_vocab_bins-1)

        logits_pred = self(seq, seq_mask)

        if self.type == 'plddt':
            loss_ce = self.ce(logits_pred[logits_mask], cl[cl_mask])
        else:
            loss_ce = self.ce(logits_pred, cl)

        self.log_dict({'test_ce_loss': loss_ce}, on_step=True, on_epoch=True, logger=True, batch_size=seq.size(0), prog_bar=False)

    def single_inference(self, seq, tokenizer):
        sequence_tok = tokenizer(' '.join(seq), return_tensors='pt', padding=True, add_special_tokens=True)
        seq_id = sequence_tok['input_ids'].cuda()
        seq_mask = sequence_tok['attention_mask'].cuda()

        logits_pred = self(seq_id, seq_mask)

        vals_pred = torch.sum(torch.nn.functional.softmax(logits_pred, -1) * self.bin_centers[None, :], axis=-1).detach().cpu().numpy()[0]

        return vals_pred

    def sc_loss(self, logits):

        max_len = max([len(l) for l in logits])

        emb = []
        mask = []
        mask_seq = []

        for l in logits:
            seq_len = len(l)
            l_perm = torch.matmul(l, self.PERM)

            l_perm_softm = gumbel_softmax(l_perm)

            l_perm_softm_emb = torch.matmul(l_perm_softm, self.EMB)

            emb.append(torch.cat([self.CLS, l_perm_softm_emb, self.SEP, self.PAD.repeat(max_len - seq_len, 1)]).unsqueeze(0))

            ones = torch.ones((1, seq_len + 2), device=self.device)
            zeros = torch.zeros((1, max_len - seq_len), device=self.device)

            mask.append(torch.cat([ones, zeros], 1))

            if self.type == 'plddt':
                ones[0][0] = ones[0][-1] = 0
                mask_seq.append(torch.cat([ones, zeros], 1))

        emb_padded = torch.cat(emb, 0)
        mask = torch.cat(mask, 0)

        L = self(seq_embeds=emb_padded, seq_mask=mask)

        if self.type == 'plddt':
            mask_seq = torch.cat(mask_seq, 0)
            L = L[mask_seq > 0]

        vals = torch.sum(torch.nn.functional.softmax(L, -1) * self.bin_centers[None, :], axis=-1)

        return vals

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.learning_rate)
        return optimizer


def gumbel_softmax(logits, dim=-1):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret