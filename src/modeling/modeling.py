import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from .transformers import BertPreTrainedModel
from .transformers import (
    BertPreTrainingHeads, BertEmbeddings, BertEncoder, BertPooler)
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from src.utils.load_save import load_state_dict_with_mismatch
import pdb

BertLayerNorm = LayerNorm


class VisualInputEmbedding_dfs(nn.Module):
    def __init__(self, config):
        super(VisualInputEmbedding_dfs, self).__init__()
        self.config = config
        self.num_sample = config.num_sample
        self.num_obj_per_frame = config.num_obj_per_frame
        self.num_rel_per_frame = config.num_rel_per_frame
        self.hidden_size = config.hidden_size
        # sequence embedding
        self.lin_obj = nn.Linear(config.obj_dim, config.hidden_size)
        self.lin_rel = nn.Linear(config.rel_dim, config.hidden_size)
        self.lin_frame = nn.Linear(config.frame_dim, config.hidden_size)
        self.lin_action = nn.Linear(config.action_dim, config.hidden_size)

        self.order_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = BertLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visual_inputs, idx_info):
        """
        Args:
            visual_inputs<list>: (B, dict)
            order_info<list>: (B, list)
        Returns:

        """
        bsz = len(visual_inputs)
        d_comb = visual_inputs[0]
        for i in range(1, bsz):
            d_comb = {key: torch.cat((d_comb[key], visual_inputs[i][key])) for key in visual_inputs[0].keys()}
        f_obj = d_comb['object'].view(-1, self.config.obj_dim)
        f_rel = d_comb['relation'].view(-1, self.config.rel_dim)
        f_frame = d_comb['frame']
        f_action = d_comb['action']

        device = f_obj.device
        obj_token = self.lin_obj(f_obj)
        rel_token = self.lin_rel(f_rel)
        frame_token = self.lin_frame(f_frame)
        action_token = self.lin_action(f_action)

        num_objs, num_rels, num_frames, num_actions, num_tokens = [], [], [], [], []
        order_embedding = []
        for info in idx_info:
            # pdb.set_trace()
            num_obj, num_rel, num_frame, num_action = [len(x) for x in info]
            num_tokens.append(num_obj + num_rel + num_frame + num_action)
            num_objs.append(num_obj)
            num_rels.append(num_rel)
            num_frames.append(num_frame)
            num_actions.append(num_action)

            obj_idxs, rel_idxs, scene_idxs, action_idxs = info
            order_idx = [
                torch.tensor(obj_idxs, dtype=torch.long,
                             device=device).flatten(),
                torch.tensor(rel_idxs, dtype=torch.long, device=device),
                torch.tensor(scene_idxs, dtype=torch.long, device=device),
                torch.tensor(action_idxs, dtype=torch.long, device=device)
            ]
            order_embeddings_ = self.order_embeddings(torch.cat(order_idx))
            order_embedding.append(order_embeddings_)

        obj_token = obj_token.split(num_objs, dim=0)
        rel_token = rel_token.split(num_rels, dim=0)
        frame_token = frame_token.split(num_frames, dim=0)
        action_token = action_token.split(num_actions, dim=0)

        visual_tokens = [torch.cat((obj_token[i], rel_token[i], frame_token[i], action_token[i])) for i in range(0, bsz)]
        visual_tokens = nn.utils.rnn.pad_sequence(visual_tokens, batch_first=True)  # (B, Lv, d)
        order_embeddings = nn.utils.rnn.pad_sequence(order_embedding, batch_first=True)  # (B, Lv, d)
        # -- Prepare masks
        pad_len = max(num_tokens)  # Lv
        num_tokens_ = torch.tensor(num_tokens, device=device).unsqueeze(1).expand(-1, pad_len)  # (bsz, pad_len)
        non_pad_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).lt(num_tokens_).squeeze(-1)  # (bsz, pad_len)

        embeddings = visual_tokens + order_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, non_pad_mask  # (B, token_length, d)


class HSTTBaseModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.visual_embeddings = VisualInputEmbedding_dfs(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def forward(self, text_input_ids, visual_inputs, order_info, attention_mask):
        input_shape = text_input_ids.size()
        device = text_input_ids.device
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.

        text_embedding_output = self.embeddings(
            input_ids=text_input_ids)  # (B, Lt, D)
        visual_embedding_output, non_pad_mask = self.visual_embeddings(
            visual_inputs, order_info)  # (B, Lv, d)
        visual_attention_mask = attention_mask.new_ones(
            visual_embedding_output.shape[:2])  # (B, Lv)
        visual_attention_mask = visual_attention_mask * non_pad_mask
        attention_mask = torch.cat(
            [attention_mask, visual_attention_mask], dim=-1)  # (B, lt+Lv, d)
        embedding_output = torch.cat(
            [text_embedding_output, visual_embedding_output],
            dim=1)  # (B, Lt+Lv, d)

        extended_attention_mask: torch.Tensor =\
            self.get_extended_attention_mask(
                attention_mask, input_shape, device)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=self.get_head_mask(
                None, self.config.num_hidden_layers)  # required input
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class HSTTModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = HSTTBaseModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        text_input_ids,
        visual_inputs,
        text_input_mask,
        mlm_labels=None,
        itm_labels=None,
    ):

        outputs = self.bert(
            text_input_ids=text_input_ids,
            visual_inputs=visual_inputs,
            attention_mask=text_input_mask,  # (B, Lt) note this mask is text only!!!
        )

        sequence_output, pooled_output = outputs[:2]
        # Only use the text part (which is the first `Lt` tokens) to save computation,
        # this won't cause any issue as cls only has linear layers.
        txt_len = text_input_mask.shape[1]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output[:, :txt_len], pooled_output)

        loss_fct = CrossEntropyLoss(reduction="none")
        if mlm_labels is not None:
            mlm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                mlm_labels.view(-1))
        else:
            mlm_loss = 0
        if itm_labels is not None:
            itm_loss = loss_fct(
                seq_relationship_score.view(-1, 2), itm_labels.view(-1))
        else:
            itm_loss = 0

        return dict(
            mlm_scores=prediction_scores,  # (B, Lt, vocab_size),  only text part
            mlm_loss=mlm_loss,  # (B, )
            mlm_labels=mlm_labels,  # (B, Lt), with -100 indicates ignored positions
            itm_scores=seq_relationship_score,  # (B, 2)
            itm_loss=itm_loss,  # (B, )
            itm_labels=itm_labels  # (B, )
        )


def instance_bce_with_logits(logits, labels, reduction="mean"):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(
        logits, labels, reduction=reduction)
    if reduction == "mean":
        loss *= labels.size(1)
    return loss


class HSTTForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(HSTTForSequenceClassification, self).__init__(config)
        self.config = config

        self.bert = HSTTBaseModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size,
                      config.hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(config.hidden_size * 2, config.num_labels)
        )

        self.init_weights()

    def forward(self, text_input_ids, visual_inputs, frame_info,
                text_input_mask, labels=None):
        outputs = self.bert(
            text_input_ids=text_input_ids,
            visual_inputs=visual_inputs,
            frame_info=frame_info,
            attention_mask=text_input_mask,  # (B, Lt) note this mask is text only!!!
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits, loss = self.calc_loss(logits, labels)
        return dict(
            logits=logits,
            loss=loss
        )

    def calc_loss(self, logits, labels):
        if labels is not None:
            if self.config.num_labels == 1:  # regression
                loss_fct = MSELoss(reduction="none")
                # labels = labels.to(torch.float)
                loss = loss_fct(
                    logits.view(-1), labels.view(-1))
            else:
                if self.config.loss_type == 'bce':  # [VQA]
                    loss = instance_bce_with_logits(
                        logits, labels, reduction="none")
                elif self.config.loss_type == "ce":  # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss(reduction="none")
                    loss = loss_fct(
                        logits.view(-1, self.config.num_labels),
                        labels.view(-1))
                else:
                    raise ValueError("Invalid option for config.loss_type")
        else:
            loss = 0
        return logits, loss

    def load_separate_ckpt(self, bert_weights_path=None):
        if bert_weights_path:
            load_state_dict_with_mismatch(self, bert_weights_path)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, hidden_states):
        return self.classifier(hidden_states)
