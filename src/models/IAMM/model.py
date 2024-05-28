### TAKEN FROM https://github.com/kolloldas/torchnlp
import os
import torch
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F

import numpy as np
import math
from src.models.common import (
    EncoderLayer,
    DecoderLayer,
    LayerNorm,
    _gen_bias_mask,
    _gen_timing_signal,
    share_embedding,
    NoamOpt,
    _get_attn_subsequent_mask,
    get_input_from_batch,
    get_output_from_batch,
    top_k_top_p_filtering,
    MultiHeadAttention,
)
from src.utils import config
from src.utils.constants import MAP_EMO
from src.utils.data.loader import load_idf
from src.utils.hypergraph_utils import *

from sklearn.metrics import accuracy_score



class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=3000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        use_mask=False,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.enc,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                )
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, : inputs.shape[1], :].type_as(
                        inputs.data
                    )
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=3000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(
                *[DecoderLayer(*params) for l in range(num_layers)]
            )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        src_mask, mask_trg = mask
        dec_mask = torch.gt(
            mask_trg + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)], 0
        )
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.dec,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    encoder_output,
                    decoding=True,
                )
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x, _, attn_dist, _ = self.dec(
                        (x, encoder_output, [], (src_mask, dec_mask))
                    )
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (src_mask, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        x,
        attn_dist=None,
        enc_batch_extend_vocab=None,
        extra_zeros=None,
        temp=1,
        beam_search=False,
        attn_dist_db=None,
    ):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist
            #beta = nn.Parameter(torch.FloatTensor(1)).cuda()

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat(
                [enc_batch_extend_vocab.unsqueeze(1)] * x.size(1), 1
            )  ## extend for all seq

            if beam_search:
                enc_batch_extend_vocab_ = torch.cat(
                    [enc_batch_extend_vocab_[0].unsqueeze(0)] * x.size(0), 0
                )  ## extend for all seq
            logit = vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_)
            #logit = torch.log(vocab_dist_)
            return torch.log(logit)
        else:
            return F.log_softmax(logit, dim=-1)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        input_num = 4 if config.woEMO else 5
        input_dim = input_num * config.hidden_dim
        hid_num = 2 if config.woEMO else 3
        hid_dim = hid_num * config.hidden_dim
        out_dim = config.hidden_dim

        self.lin_1 = nn.Linear(input_dim, hid_dim, bias=False)
        self.lin_2 = nn.Linear(hid_dim, out_dim, bias=False)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)

        return x


class IAMM(nn.Module):
    def __init__(
        self,
        vocab,
        decoder_number,
        model_file_path=None,
        is_eval=False,
        load_optim=False,
    ):
        super(IAMM, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.word_freq = np.zeros(self.vocab_size)
        self.special_indexes = [config.UNK_idx, config.PAD_idx, config.EOS_idx, config.SOS_idx, config.USR_idx, config.SYS_idx, config.CLS_idx]

        self.is_eval = is_eval
        self.rels = ["c_intent", "c_need", "c_want", "c_effect", "c_react"]
        self.con_topk = len(self.rels)
        num_emotions = 32
        self.compress_dim = 10

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)

        self.encoder = self.make_encoder(config.emb_dim)
        self.total_encoder = self.make_encoder(config.emb_dim)
        self.cog_encoder = self.make_encoder(config.emb_dim)

        self.sit_encoder = self.make_encoder(config.emb_dim)
        self.rel_encoder = self.make_encoder(config.emb_dim)
        self.total_con_encoder = self.make_encoder(config.emb_dim)
        self.sc_MHA = self.make_MHA(num_heads=2, depth=config.depth) #depth=40
        self.cc_MHA = self.sc_MHA
        self.rc_MHA = self.make_MHA(num_heads=2, depth=config.depth) #depth=40

        self.con_sc_MHA = self.make_MHA(num_heads=2, depth=config.depth) #depth=40
        self.con_cc_MHA = self.con_sc_MHA #self.make_MHA(num_heads=2, depth=config.depth) #depth=40
        self.con_rc_MHA = self.make_MHA(num_heads=2, depth=config.depth) #depth=40

        self.decoder = Decoder(
            config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
        )

        self.rel_decoder = Decoder(
            config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
        )

        self.select_w = nn.Linear(config.hidden_dim, 1, bias=False)

        self.logit_linear = nn.Linear(3 * config.hidden_dim, 1) 
        self.rs_logit_linear = nn.Linear(2 * config.hidden_dim, 1) 

        self.attention_layer = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attention_v = nn.Linear(config.hidden_dim, 1, bias=False)
        self.hidden_layer = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.output_layer = nn.Linear(config.hidden_dim, num_emotions)

        self.attention_layer2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attention_v2 = nn.Linear(config.hidden_dim, 1, bias=False)
        self.hidden_layer2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.output_layer2 = nn.Linear(config.hidden_dim, num_emotions)

        relation_dim = config.hidden_dim
        self.attention_layer4 = nn.Linear(relation_dim, relation_dim)
        self.attention_v4 = nn.Linear(relation_dim, 1, bias=False)
        self.hidden_layer4 = nn.Linear(relation_dim, relation_dim)
        self.output_layer4 = nn.Linear(relation_dim, num_emotions)

        self.attention_layer5 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attention_v5 = nn.Linear(config.hidden_dim, 1, bias=False)
        self.hidden_layer5 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.output_layer5 = nn.Linear(config.hidden_dim, num_emotions)

        self.ctx_linear = nn.Linear(config.ctx_topk, config.ctx_topk)
        self.con_linear = nn.Linear(config.con_topk, config.con_topk)
        self.sent_linear = nn.Linear(config.cs_topk, config.cs_topk)
        sent_hidden = config.ctx_topk * ( config.depth // config.heads)
        self.relation_linear = nn.Linear(sent_hidden, config.hidden_dim)
        con_hidden = config.con_topk * ( config.depth // config.heads)
        self.con_relation_linear = nn.Linear(con_hidden, config.hidden_dim)

        if not config.woCOG:
            self.cog_lin = MLP()

        self.generator = Generator(config.hidden_dim, self.vocab_size)
        self.activation = nn.Softmax(dim=1)

        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx, reduction="sum")
        if not config.woDiv:
            self.criterion.weight = torch.ones(self.vocab_size)

        self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(
                config.hidden_dim,
                1,
                8000,
                torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
            )

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=config.device)
            self.load_state_dict(state["model"])
            if load_optim:
                self.optimizer.load_state_dict(state["optimizer"])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def make_encoder(self, emb_dim):
        return Encoder(
            emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )

    def make_MHA(self, num_heads, depth):
        return MultiHeadAttention(
            input_depth=config.hidden_dim,
            total_key_depth=depth,
            total_value_depth=depth,
            output_depth=config.hidden_dim,
            num_heads=num_heads,
            bias_mask=None,
            dropout=0.0,
        )

    def save_model(self, running_avg_ppl, iter, acc_val):
        state = {
            "iter": iter,
            "optimizer": self.optimizer.state_dict(),
            "current_loss": running_avg_ppl,
            "model": self.state_dict(),
        }
        model_save_path = os.path.join(
            self.model_dir,
            "IAMM_{}_{:.4f}_{:.4f}".format(iter, running_avg_ppl, acc_val),
        )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def get_sit_logits(self, sit_outputs):
        projected = self.attention_layer(sit_outputs)
        projected = nn.Tanh()(projected)
        scores = nn.Softmax(dim=-1)(self.attention_v(projected).squeeze(2))
        scores = scores.unsqueeze(1)  # (batch_size, 1, seq_len)
        hidden_x = torch.bmm(scores, sit_outputs).squeeze(1)
        x = self.hidden_layer(hidden_x)
        x = nn.Tanh()(x)
        sit_emo_logits = self.output_layer(x)
        return sit_emo_logits

    def get_ctx_logits(self, enc_outputs):
        # Attention emotion
        projected = self.attention_layer2(enc_outputs)
        projected = nn.Tanh()(projected)
        scores = nn.Softmax(dim=-1)(self.attention_v2(projected).squeeze(2))
        scores = scores.unsqueeze(1)  # (batch_size, 1, seq_len)
        hidden_x = torch.bmm(scores, enc_outputs).squeeze(1)
        x = self.hidden_layer2(hidden_x)
        x = nn.Tanh()(x)
        emo_logits = self.output_layer2(x)
        return emo_logits 

    def get_relation_logits(self, enc_outputs):
        # Attention emotion
        projected = self.attention_layer4(enc_outputs)
        projected = nn.Tanh()(projected)
        scores = nn.Softmax(dim=-1)(self.attention_v4(projected).squeeze(2))
        scores = scores.unsqueeze(1)  # (batch_size, 1, seq_len)
        hidden_x = torch.bmm(scores, enc_outputs).squeeze(1)
        x = self.hidden_layer4(hidden_x)
        x = nn.Tanh()(x)
        emo_logits = self.output_layer4(x)
        return emo_logits 

    def get_con_logits(self, enc_outputs):
        # Attention emotion
        projected = self.attention_layer5(enc_outputs)
        projected = nn.Tanh()(projected)
        scores = nn.Softmax(dim=-1)(self.attention_v5(projected).squeeze(2))
        scores = scores.unsqueeze(1)  # (batch_size, 1, seq_len)
        hidden_x = torch.bmm(scores, enc_outputs).squeeze(1)
        x = self.hidden_layer5(hidden_x)
        x = nn.Tanh()(x)
        emo_logits = self.output_layer5(x)
        return emo_logits 

    def enc_comet(self, batch, i, data_type): #data_type: ctx, sit
        cs_indexes = []
        cs_embs = []
        cs_masks = []
        cs_outputs = []
        for r in self.rels:
            if data_type == "sit":
                r = r.replace("c_", "s_")
            batch_r = batch[r][:, i, :]
            emb = self.embedding(batch_r).to(config.device)
            mask = batch_r.data.eq(config.PAD_idx).unsqueeze(1)
            cs_indexes.append(batch_r)
            cs_embs.append(emb)
            if "react" not in r:
                enc_output = self.cog_encoder(emb, mask)
            else:
                enc_output = self.cog_encoder(emb, mask)
            enc_output = enc_output[:, 0, :].unsqueeze(1)
            cs_outputs.append(enc_output)
            mask = mask[:, :, 1].unsqueeze(2) # CLS[0] is masked
            cs_masks.append(mask)
        cs_outputs = torch.cat(cs_outputs, dim=1)
        cs_masks = torch.cat(cs_masks, dim=2)
        return cs_outputs, cs_masks

    def select_topk(self, key, value, mask=None, k=config.ctx_topk): 
        value_shape = value.shape
        scores, index = key.topk(k, dim=-1, largest=True, sorted=True)
        index = index.view(-1, k)
            
        bz_index = torch.tensor(range(index.shape[0])) * value_shape[-2]
        bz_index = bz_index.unsqueeze(-1).cuda()
        index = (index + bz_index).view(-1)
        value = value.reshape(-1, value.shape[-1])
        selected_values = torch.index_select(value, 0, index)
        if mask is None:
            return scores, selected_values
        else:
            selected_mask = torch.gather(mask.reshape(-1), dim=-1, index=index)
            return scores, selected_values, selected_mask

    def calculate_weight_value(self, scores, selected_values, value_shape, con_flag):
        if con_flag:
            k = config.con_topk
        else:
            k = config.ctx_topk
        selected_values = selected_values.view(value_shape[0], value_shape[1], value_shape[2], k, value_shape[4])
        if con_flag:
            scores = self.con_linear(scores)
        else:
            scores = self.ctx_linear(scores)
        scores = torch.sigmoid(scores).unsqueeze(-1)
        v_shape = selected_values.shape
        weighted_values = (scores * selected_values).view(v_shape[0], v_shape[1], v_shape[2], v_shape[3] * v_shape[4])
        return weighted_values 

    def calculate_sent_value(self, scores, selected_values, value_shape, k=config.cs_topk):
        selected_values = selected_values.view(value_shape[0], value_shape[1], k, value_shape[3])
        scores = self.sent_linear(scores)
        scores = torch.sigmoid(scores).unsqueeze(-1)
        v_shape = selected_values.shape
        weighted_values = (scores * selected_values).reshape(v_shape[0], v_shape[1] * v_shape[2], v_shape[3])
        return weighted_values 

    def reshape_mask(self, mask, m_shape):
        return mask.reshape(m_shape[0], m_shape[1] * config.cs_topk).unsqueeze(1)

    def build_relation(self, r1_logit, r1_value, r2_logit, r2_value, r1_mask, r2_mask, con_flag=False):
        if con_flag:
            ctx_k = config.con_topk
        else:
            ctx_k = config.ctx_topk
        r1_value = r1_value.unsqueeze(2).repeat(1, 1, r1_logit.shape[2], 1, 1) # Expand to the same dimension as r1_logit.
        scores, selected_values = self.select_topk(r1_logit, r1_value, mask=None, k=ctx_k)
        weighted_r1 = self.calculate_weight_value(scores, selected_values, r1_value.shape, con_flag=con_flag)

        r2_value = r2_value.unsqueeze(2).repeat(1, 1, r2_logit.shape[2], 1, 1)
        scores, selected_values  = self.select_topk(r2_logit, r2_value, mask=None, k=ctx_k)
        weighted_r2 = self.calculate_weight_value(scores, selected_values, r2_value.shape, con_flag=con_flag)

        # Select and merge the relations
        cross_r1_logit = torch.mean(r1_logit, dim=2)
        cross_r2_logit = torch.mean(r2_logit, dim=2)
        r1_mask = r1_mask.repeat(1, config.heads, 1)
        r2_mask = r2_mask.repeat(1, config.heads, 1)

        cross_r1_scores, cross_r1, cross_r1_mask = self.select_topk(cross_r2_logit, weighted_r1, r1_mask, config.cs_topk)
        cross_r2_scores, cross_r2, cross_r2_mask = self.select_topk(cross_r1_logit, weighted_r2, r2_mask, config.cs_topk)
        weighted_cross_r1 = self.calculate_sent_value(cross_r1_scores, cross_r1, weighted_r1.shape, k=config.cs_topk)
        weighted_cross_r2 = self.calculate_sent_value(cross_r2_scores, cross_r2, weighted_r2.shape, k=config.cs_topk)
        relations = torch.cat((weighted_cross_r1, weighted_cross_r2), dim=1)
        cross_r1_mask = self.reshape_mask(cross_r1_mask, weighted_r1.shape)
        cross_r2_mask = self.reshape_mask(cross_r2_mask, weighted_r2.shape)
        relations_mask = torch.cat((cross_r1_mask, cross_r2_mask), dim=2)

        if con_flag:
            relations = self.con_relation_linear(relations)
        else:
            relations = self.relation_linear(relations)
        return relations, relations_mask 
        
    def forward(self, batch):
        # Input for context and situation.

        # Context input
        enc_batch = batch["input_batch"]
        src_mask = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        mask_emb = self.embedding(batch["mask_input"])
        src_emb = self.embedding(enc_batch) + mask_emb

        # Situation input
        situation_batch = batch["situation_batch"]
        sit_emb = self.embedding(situation_batch)
        sit_mask = situation_batch.data.eq(config.PAD_idx).unsqueeze(1)

        ## Encode sit comet
        sit_con_outputs, sit_con_masks = self.enc_comet(batch, 0, data_type="sit")

        # Sentence encoder outputs.
        sent_ctx_outputs = []
        previous_ctx = None
        previous_ctx_mask = None
        previous_rel = None
        previous_rel_mask = None

        # Sentence comet encoder outputs
        sent_ctx_con_outputs = []
        sent_ctx_con_outputs_mask = []
        previous_con_rel = None
        previous_con_rel_mask = None

        for i in range(enc_batch.shape[1]): # the number of sentence
            ## Encode ctx
            current_ctx = src_emb[:, i, :, :]
            current_src_mask = src_mask[:, :, i, :]
            enc_outputs = self.encoder(current_ctx, current_src_mask)  # batch_size * seq_len * 300
            sent_ctx_outputs.append(enc_outputs)
        
            # Build relation: ctx <-> sit
            c2s_logit, c2s_value = self.sc_MHA(current_ctx, sit_emb, sit_emb, sit_mask, topk=True)
            s2c_logit, s2c_value = self.sc_MHA(sit_emb, current_ctx, current_ctx, current_src_mask, topk=True)
            sc_rel, sc_rel_mask = self.build_relation(c2s_logit, c2s_value, s2c_logit, s2c_value, current_src_mask, sit_mask)
    
            ## Encode ctx comet
            ctx_con_outputs, ctx_con_masks = self.enc_comet(batch, i, data_type="ctx")
            sent_ctx_con_outputs.append(ctx_con_outputs)
            sent_ctx_con_outputs_mask.append(ctx_con_masks)

            # Build con relation: ctx <-> sit
            con_c2s_logit, con_c2s_value = self.con_sc_MHA(ctx_con_outputs, sit_con_outputs, sit_con_outputs, sit_con_masks, topk=True)
            con_s2c_logit, con_s2c_value = self.con_sc_MHA(sit_con_outputs, ctx_con_outputs, ctx_con_outputs, ctx_con_masks, topk=True)
            con_sc_rel, con_sc_rel_mask = self.build_relation(con_c2s_logit, con_c2s_value, con_s2c_logit, \
                                                        con_s2c_value, ctx_con_masks, sit_con_masks, \
                                                        con_flag=True)
            
            if previous_ctx is None: # Fist ctx
                ## Ctx relations
                previous_ctx, previous_ctx_mask = current_ctx, current_src_mask 
                previous_rel, previous_rel_mask = sc_rel, sc_rel_mask
                # Comet
                previous_con_rel, previous_con_rel_mask = con_sc_rel, con_sc_rel_mask
            else: # The remaining ctx
                # Build ctx relation: ctx <-> pre_rel
                c2r_logit, c2r_value = self.rc_MHA(current_ctx, previous_rel, previous_rel, previous_rel_mask, topk=True)
                r2c_logit, r2c_value = self.rc_MHA(previous_rel, current_ctx, current_ctx, current_src_mask, topk=True)
                rc_rel, rc_rel_mask = self.build_relation(c2r_logit, c2r_value, r2c_logit, r2c_value, current_src_mask, previous_rel_mask)

                # Build ctx relation: ctx <-> pre_ctx
                cc2pc_logit, cc2pc_value = self.cc_MHA(current_ctx, previous_ctx, previous_ctx, previous_ctx_mask, topk=True)
                pc2cc_logit, pc2cc_value = self.cc_MHA(previous_ctx, current_ctx, current_ctx, current_src_mask, topk=True)
                pc_rel, pc_rel_mask = self.build_relation(cc2pc_logit, cc2pc_value, pc2cc_logit, pc2cc_value, current_src_mask, previous_ctx_mask)
                previous_ctx = torch.cat((previous_ctx, current_ctx), dim=1)
                previous_ctx_mask = torch.cat((previous_ctx_mask, current_src_mask), dim=2)

                previous_rel = torch.cat((previous_rel, sc_rel, rc_rel, pc_rel), dim=1)
                previous_rel_mask = torch.cat((previous_rel_mask, sc_rel_mask, rc_rel_mask, pc_rel_mask), dim=2)

                ## Comet relations
                # Build con relation: ctx <-> pre_rel
                con_c2r_logit, con_c2r_value = self.con_rc_MHA(ctx_con_outputs, previous_con_rel, previous_con_rel, previous_con_rel_mask, topk=True)
                con_r2c_logit, con_r2c_value = self.con_rc_MHA(previous_con_rel, ctx_con_outputs, ctx_con_outputs, ctx_con_masks, topk=True)
                con_rc_rel, con_rc_rel_mask = self.build_relation(con_c2r_logit, con_c2r_value, con_r2c_logit, \
                                                            con_r2c_value, ctx_con_masks, previous_con_rel_mask, \
                                                            con_flag=True)

                # Build con relation: ctx <-> pre_ctx
                previous_ctx_con = torch.cat(sent_ctx_con_outputs[:-1], dim=1)
                previous_ctx_con_mask = torch.cat(sent_ctx_con_outputs_mask[:-1], dim=2)
                con_cc2pc_logit, con_cc2pc_value = self.con_cc_MHA(ctx_con_outputs, previous_ctx_con, previous_ctx_con, previous_ctx_con_mask, topk=True)
                con_pc2cc_logit, con_pc2cc_value = self.con_cc_MHA(previous_ctx_con, ctx_con_outputs, ctx_con_outputs, ctx_con_masks, topk=True)
                con_pc_rel, con_pc_rel_mask = self.build_relation(con_cc2pc_logit, con_cc2pc_value,\
                                                            con_pc2cc_logit, con_pc2cc_value, \
                                                            ctx_con_masks, previous_ctx_con_mask, \
                                                            con_flag=True)
                previous_con_rel = torch.cat((previous_con_rel, con_sc_rel, con_rc_rel, con_pc_rel), dim=1)
                previous_con_rel_mask = torch.cat((previous_con_rel_mask, con_sc_rel_mask, con_rc_rel_mask, con_pc_rel_mask), dim=2)

        # Encode total ctx.
        total_ctx = torch.cat(sent_ctx_outputs, dim=1)
        mask_shape = src_mask.shape
        total_ctx_mask = src_mask.view(mask_shape[0], mask_shape[1], mask_shape[2] * mask_shape[3]) 
        total_ctx_outputs = self.total_encoder(total_ctx, total_ctx_mask)
        ctx_emo_logits = self.get_ctx_logits(total_ctx_outputs)

        # Encode situation.
        sit_outputs = self.sit_encoder(sit_emb, sit_mask)
        sit_emo_logits = self.get_ctx_logits(sit_outputs)

        # Encode relation.
        previous_rel = torch.cat((previous_rel, previous_con_rel), dim=1)
        previous_rel_mask = torch.cat((previous_rel_mask, previous_con_rel_mask), dim=2)
        #print(f"previous_rel: {previous_rel.shape}")
        rel_outputs = self.rel_encoder(previous_rel, previous_rel_mask)  # batch_size * seq_len * 300
        relation_logits = self.get_relation_logits(rel_outputs)
        
        ## Comet
        # Comet enc
        total_con_outputs = torch.cat((sit_con_outputs, sent_ctx_con_outputs[-1]), dim=1) 
        total_con_mask = torch.cat((sit_con_masks, sent_ctx_con_outputs_mask[-1]), dim=2)
        total_con_outputs = self.total_con_encoder(total_con_outputs, total_con_mask)
        con_logits = self.get_con_logits(total_con_outputs)

        select_score = self.select_w(previous_rel).squeeze(-1)
        rel_score, rel_value, rel_mask = self.select_topk(select_score, previous_rel, mask=previous_rel_mask, k=15)
        rel_score = rel_score.unsqueeze(-1)
        rel_value = rel_value.reshape(rel_score.shape[0], rel_score.shape[1], -1)
        rel_mask = rel_mask.reshape(rel_score.shape[0], 1, rel_score.shape[1]) 
        rel_value =  torch.sigmoid(rel_score) * rel_value 

        # Emotion logit
        emo_logits = ctx_emo_logits + sit_emo_logits + relation_logits + con_logits #+ con_rel_logits

        return total_ctx_mask, sit_mask, total_ctx_outputs, sit_outputs, ctx_emo_logits, sit_emo_logits, emo_logits, relation_logits, con_logits, rel_value, rel_mask

    def train_one_batch(self, batch, iter, train=True):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            situation_ext_batch,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()
        src_mask, sit_mask, ctx_output, sit_output, ctx_emo_logits, sit_emo_logits, emo_logits, relation_emo_logits, con_logits, rel_value, rel_mask = self.forward(batch)

        # Decode
        sos_token = (
            torch.LongTensor([config.SOS_idx] * enc_batch.size(0))
            .unsqueeze(1)
            .to(config.device)
        )
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), dim=1)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)

        # batch_size * seq_len * 300 (GloVe)
        dec_emb = self.embedding(dec_batch_shift)
        pre_logit, attn_dist = self.decoder(dec_emb, ctx_output, (src_mask, mask_trg))
        rel_pre_logit, rel_attn_dist = self.rel_decoder(dec_emb, rel_value, (rel_mask, mask_trg))

        rs_gate_logit = torch.cat([pre_logit, rel_pre_logit], dim=-1)
        rs_gate_score = torch.sigmoid(self.rs_logit_linear(rs_gate_logit))
        pre_logit = rs_gate_score * pre_logit + (1 - rs_gate_score) * rel_pre_logit

        ## compute output dist
        logit = self.generator(
            pre_logit,
            attn_dist,
            enc_batch_extend_vocab if config.pointer_gen else None,
            extra_zeros,
            attn_dist_db=None,
        )

        emo_label = torch.LongTensor(batch["program_label"]).to(config.device)
        #situation_vec_batch = batch["situation_vec_batch"]

        # Emotion loss
        sit_emo_loss = nn.CrossEntropyLoss(reduction='mean')(sit_emo_logits, emo_label).to(config.device)
        enc_emo_loss = nn.CrossEntropyLoss(reduction='mean')(ctx_emo_logits, emo_label).to(config.device)
        relation_emo_loss = nn.CrossEntropyLoss(reduction='mean')(relation_emo_logits, emo_label).to(config.device)
        con_emo_loss = nn.CrossEntropyLoss(reduction='mean')(con_logits, emo_label).to(config.device)

        ctx_loss = self.criterion_ppl(
            logit.contiguous().view(-1, logit.size(-1)),
            dec_batch.contiguous().view(-1),
        )


        loss = ctx_loss + sit_emo_loss + enc_emo_loss + relation_emo_loss + con_emo_loss 

        emo_loss = sit_emo_loss + enc_emo_loss + relation_emo_loss + con_emo_loss 
        pred_program = np.argmax(emo_logits.detach().cpu().numpy(), axis=1)
        program_acc = accuracy_score(batch["program_label"], pred_program)

        # print results for testing
        top_preds = ""
        comet_res = {}

        ctx_loss_list = []
        batch_size = emo_logits.size(0)
        top_preds = [[] for _ in range(batch_size)]  
        comet_res = []  

        if self.is_eval:
            # comet outputs
            top_probs, top_indices = emo_logits.detach().cpu().topk(3, dim=-1)
            for i, indices in enumerate(top_indices):
                top_preds[i] = [MAP_EMO[index.item()] for index in indices]
            for i in range(batch_size):
                temp_dict = {}
                for r in self.rels:
                    temp_dict[r] = []#txt
                comet_res.append(temp_dict)

            # update test batch
            for i in range(logit.shape[0]):
                logit_i = logit[i:i + 1].contiguous().view(-1, logit.size(-1))
                dec_batch_i = dec_batch[i:i + 1].contiguous().view(-1)
                loss_i = self.criterion_ppl(logit_i, dec_batch_i)
                ctx_loss_list.append(loss_i.item())

        if train:
            loss.backward()
            self.optimizer.step()

        return (
            ctx_loss.item() if train else ctx_loss_list,  
            math.exp(min(ctx_loss.item(), 100)) if train else np.mean(ctx_loss_list),  # modify testBatch
            emo_loss.item(),
            program_acc,
            top_preds,
            comet_res,
        )

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss


    def decoder_greedy(self, batch, max_dec_step=30):
        (   
            _,  
            _,  
            _,  
            enc_batch_extend_vocab,
            situation_ext_batch,
            extra_zeros,
            _,  
            _,  
            _,  
        ) = get_input_from_batch(batch)
        src_mask, sit_mask, ctx_output, sit_output, ctx_emo_logits, sit_emo_logits, emo_logits, relation_emo_logits, con_logits = self.forward(batch)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1): 
            ys_embed = self.embedding(ys)
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(ys_embed),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )   
            else:
                out, attn_dist = self.decoder(
                    ys_embed, ctx_output, (src_mask, mask_trg)
                )   
                sit_out, sit_attn_dist = self.sit_decoder(  # out : torch.Size([batch_size, 1, 300])
                    ys_embed, sit_output, (sit_mask, mask_trg)
                )   

                gate_logit = torch.cat([out, sit_out], dim=-1)
                gate_score = torch.sigmoid(self.logit_linear(gate_logit))
                out = gate_score * out + (1 - gate_score) * sit_out

            prob = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )   
            prob = torch.log(prob)

            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(
                [   
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]   
            )   
            next_word = next_word.data[0]

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " " 
            sent.append(st)
        return sent

    def decoder_greedy_batch(self, batch, max_dec_step=30):
        (   
            _,  
            _,  
            _,  
            enc_batch_extend_vocab,
            situation_ext_batch,
            extra_zeros,
            _,  
            _,  
            _,  
        ) = get_input_from_batch(batch)
        src_mask, sit_mask, ctx_output, sit_output, ctx_emo_logits, sit_emo_logits, emo_logits, relation_emo_logits, con_logits, rel_value, rel_mask = self.forward(batch)

        batch_size = ctx_output.size(0)  # testBatch
        ys = torch.ones(batch_size, 1).fill_(config.SOS_idx).long().to(config.device)  # torch.Size([batch_size, 1])  # testBatch
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)  # torch.Size([batch_size, 1, 1])
        decoded_words = []

        for i in range(max_dec_step + 1):  
            ys_embed = self.embedding(ys)  # torch.Size([batch_size, 1, 300])
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(ys_embed),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )   
            else:  

                out, attn_dist = self.decoder(  # out : torch.Size([batch_size, 1, 300])
                    ys_embed, ctx_output, (src_mask, mask_trg)
                )   
    
                # , rel_value, rel_mask
                rel_out, rel_attn_dist = self.rel_decoder(  # out : torch.Size([batch_size, 1, 300])
                    ys_embed, rel_value, (rel_mask, mask_trg)
                )   

                rs_gate_logit = torch.cat([out, rel_out], dim=-1)
                rs_gate_score = torch.sigmoid(self.rs_logit_linear(rs_gate_logit))
                out = rs_gate_score * out + (1 - rs_gate_score) * rel_out

            prob = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )   

            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(
                [   
                    "<EOS>" if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]   
            )   
            next_word = next_word.data  # testBatch

            ys = torch.cat(
                [ys, next_word.unsqueeze(1).long().to(config.device)],
                dim=1,
            ).to(config.device)  # testBatch
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)  

        sents = []  #  testBatch
        for _, row in enumerate(np.transpose(decoded_words)):
            sent = []
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " " 
            sent.append(st)
            sents.append(sent)  #  testBatch

        return sents  

    def decoder_topk(self, batch, max_dec_step=30):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            situation_ext_batch,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        src_mask, ctx_output, _ = self.forward(batch)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    self.embedding(ys), ctx_output, (src_mask, mask_trg)
                )

            logit = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            filtered_logit = top_k_top_p_filtering(
                logit[0, -1] / 0.7, top_k=0, top_p=0.9, filter_value=-float("Inf")
            )
            # Sample from the filtered distribution
            probs = F.softmax(filtered_logit, dim=-1)

            next_word = torch.multinomial(probs, 1).squeeze()
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            # _, next_word = torch.max(logit[:, -1], dim=1)
            next_word = next_word.item()

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent
