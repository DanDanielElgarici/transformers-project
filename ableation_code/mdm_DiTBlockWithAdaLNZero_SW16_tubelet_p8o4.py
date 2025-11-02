import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from model.BERT.BERT_encoder import load_bert
from utils.misc import WeightedSum


class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)
        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.mask_frames = kargs.get('mask_frames', False)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim)

        self.emb_policy = kargs.get('emb_policy', 'add')

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout,
                                                       max_len=kargs.get('pos_embed_max_len', 5000))
        self.emb_trans_dec = emb_trans_dec

        self.pred_len = kargs.get('pred_len', 0)
        self.context_len = kargs.get('context_len', 0)
        self.total_len = self.pred_len + self.context_len
        self.is_prefix_comp = self.total_len > 0
        self.all_goal_joint_names = kargs.get('all_goal_joint_names', [])

        self.tokenization = kargs.get('tokenization', 'frame')
        self.patch_size = int(kargs.get('patch_size', 8))
        self.patch_overlap = int(kargs.get('patch_overlap', 4))

        self.multi_target_cond = kargs.get('multi_target_cond', False)
        self.multi_encoder_type = kargs.get('multi_encoder_type', 'multi')
        self.target_enc_layers = kargs.get('target_enc_layers', 1)
        if self.multi_target_cond:
            if self.multi_encoder_type == 'multi':
                self.embed_target_cond = EmbedTargetLocMulti(self.all_goal_joint_names, self.latent_dim)
            elif self.multi_encoder_type == 'single':
                self.embed_target_cond = EmbedTargetLocSingle(self.all_goal_joint_names, self.latent_dim,
                                                              self.target_enc_layers)
            elif self.multi_encoder_type == 'split':
                self.embed_target_cond = EmbedTargetLocSplit(self.all_goal_joint_names, self.latent_dim,
                                                             self.target_enc_layers)

        if self.tokenization == 'temporal_patch':
            self.tokenizer = TemporalPatchTokenizer(self.data_rep, self.njoints, self.nfeats, self.latent_dim,
                                                    patch_size=self.patch_size, overlap=self.patch_overlap)
            self.detokenizer = TemporalPatchDetokenizer(self.data_rep, self.njoints, self.nfeats, self.latent_dim,
                                                        patch_size=self.patch_size, overlap=self.patch_overlap)
        else:
            # default behavior (per-frame)
            self.input_process = InputProcess(self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim)
            self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                                self.nfeats)

        if self.arch == 'trans_enc':
            print("DiT TRANS_ENC init with adaLN-Zero")
            attn_window = kargs.get('attn_window', 16)
            self.dit_blocks = nn.ModuleList([
                DiTBlockWithAdaLNZero(
                    hidden_size=self.latent_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.ff_size / self.latent_dim,
                    dropout=self.dropout,
                    activation=self.activation,
                    window_size=attn_window
                ) for _ in range(self.num_layers)
            ])
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                # We support CLIP encoder and DistilBERT
                print('EMBED TEXT')

                self.text_encoder_type = kargs.get('text_encoder_type', 'clip')

                if self.text_encoder_type == "clip":
                    print('Loading CLIP...')
                    self.clip_version = clip_version
                    self.clip_model = self.load_and_freeze_clip(clip_version)
                    self.encode_text = self.clip_encode_text
                elif self.text_encoder_type == 'bert':
                    assert self.arch == 'trans_dec'
                    # assert self.emb_trans_dec == False # passing just the time embed so it's fine
                    print("Loading BERT...")
                    # bert_model_path = 'model/BERT/distilbert-base-uncased'
                    bert_model_path = 'distilbert/distilbert-base-uncased'
                    self.clip_model = load_bert(
                        bert_model_path)  # Sorry for that, the naming is for backward compatibility
                    self.encode_text = self.bert_encode_text
                    self.clip_dim = 768
                else:
                    raise ValueError('We only support [CLIP, BERT] text encoders')

                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)

            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training

        # clip.model.convert_weights(
        #    clip_model)  # Actually this line is unnecessary since clip by default already on float16

        for p in clip_model.parameters():
            p.data = p.data.float()  # force float32

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[-2]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs,
                                                                                                  1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def clip_encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2  # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(
                device)  # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length - context_length], dtype=texts.dtype,
                                   device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(
                device)  # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float().unsqueeze(0)

    def bert_encode_text(self, raw_text):
        # enc_text = self.clip_model(raw_text)
        # enc_text = enc_text.permute(1, 0, 2)
        # return enc_text
        enc_text, mask = self.clip_model(
            raw_text)  # self.clip_model.get_last_hidden_state(raw_text, return_mask=True)  # mask: False means no token there
        enc_text = enc_text.permute(1, 0, 2)
        mask = ~mask  # mask: True means no token there, we invert since the meaning of mask for transformer is inverted  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        return enc_text, mask

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        time_emb = self.embed_timestep(timesteps)  # [1, bs, d]

        if 'target_cond' in y.keys():
            # NOTE: We don't use CFG for joints - but we do wat to support uncond sampling for generation and eval!
            time_emb += self.mask_cond(
                self.embed_target_cond(y['target_cond'], y['target_joint_names'], y['is_heading'])[None],
                force_mask=y.get('target_uncond', False))  # For uncond support and CFG
            # time_emb += self.embed_target_cond(y['target_cond'], y['target_joint_names'], y['is_heading'])[None]

        # Build input for prefix completion
        if self.is_prefix_comp:
            x = torch.cat([y['prefix'], x], dim=-1)
            y['mask'] = torch.cat(
                [torch.ones([bs, 1, 1, self.context_len], dtype=y['mask'].dtype, device=y['mask'].device),
                 y['mask']], dim=-1)

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            if 'text_embed' in y.keys():  # caching option
                enc_text = y['text_embed']
            else:
                enc_text = self.encode_text(y['text'])
            if type(enc_text) == tuple:
                enc_text, text_mask = enc_text
                if text_mask.shape[0] == 1 and bs > 1:  # casting mask for the single-prompt-for-all case
                    text_mask = torch.repeat_interleave(text_mask, bs, dim=0)
            text_emb = self.embed_text(
                self.mask_cond(enc_text, force_mask=force_mask))  # casting mask for the single-prompt-for-all case
            if self.emb_policy == 'add':
                emb = text_emb + time_emb
            else:
                emb = torch.cat([time_emb, text_emb], dim=0)
                text_mask = torch.cat([torch.zeros_like(text_mask[:, 0:1]), text_mask], dim=1)
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb = time_emb + self.mask_cond(action_emb, force_mask=force_mask)
        if self.cond_mode == 'no_cond':
            # unconstrained
            emb = time_emb

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints * nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)  # [#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)  # [bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  # [bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru), axis=1)  # [bs, d+joints*feat, 1, #frames]

        if self.tokenization == 'temporal_patch':
            tokens, tok_aux = self.tokenizer(x)  # [Np, bs, d]
            x = tokens  # feed tokens
        else:
            x = self.input_process(x)

        # TODO - move to collate
        frames_mask = None
        is_valid_mask = y['mask'].shape[-1] > 1  # Don't use mask with the generate script
        if self.mask_frames and is_valid_mask:
            if self.tokenization == 'temporal_patch':
                # reduce frame mask into patch mask (True==pad/invalid)
                # y['mask']: [bs,1,1,T] where 1 means VALID in your code then negated later;
                # we need key_padding_mask where True==to be masked => invalid
                bs, _, _, T = y['mask'].shape
                P, S = self.patch_size, (self.patch_size - self.patch_overlap)
                n_patches = x.shape[0]  # seqlen after tokenization
                patch_mask = torch.zeros(bs, n_patches, dtype=torch.bool, device=x.device)
                for i in range(n_patches):
                    s = i * S
                    e = min(s + P, T)
                    # invalid if ALL frames in the patch are invalid (mask==0)
                    valid_any = y['mask'][:, :, :, s:e].reshape(bs, -1).any(dim=1)
                    patch_mask[:, i] = ~valid_any
                if self.emb_trans_dec or self.arch == 'trans_enc':
                    step_mask = torch.zeros((bs, 1), dtype=torch.bool, device=x.device)
                    frames_mask = torch.cat([step_mask, patch_mask], dim=1)
                else:
                    frames_mask = patch_mask
            else:
                frames_mask = torch.logical_not(y['mask'][..., :x.shape[0]].squeeze(1).squeeze(1)).to(device=x.device)
                if self.emb_trans_dec or self.arch == 'trans_enc':
                    step_mask = torch.zeros((bs, 1), dtype=torch.bool, device=x.device)
                    frames_mask = torch.cat([step_mask, frames_mask], dim=1)

        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]

            # Extract conditioning token and motion tokens
            cond_token = xseq[0:1]  # [1, bs, d] - conditioning token
            motion_tokens = xseq[1:]  # [seqlen, bs, d] - motion tokens

            # dit blocc
            for dit_block in self.dit_blocks:
                motion_tokens = dit_block(motion_tokens, cond_token.squeeze(0))

            output = motion_tokens  # [seqlen, bs, d]

        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((time_emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]

            if self.text_encoder_type == 'clip':
                output = self.seqTransDecoder(tgt=xseq, memory=emb, tgt_key_padding_mask=frames_mask)
            elif self.text_encoder_type == 'bert':
                output = self.seqTransDecoder(tgt=xseq, memory=emb, memory_key_padding_mask=text_mask,
                                              tgt_key_padding_mask=frames_mask)  # Rotem's bug fix
            else:
                raise ValueError()

            if self.emb_trans_dec:
                output = output[1:]  # [seqlen, bs, d]

        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        # Extract completed suffix
        if self.is_prefix_comp:
            if self.tokenization == 'temporal_patch':
                # keep patches whose (start+P) > context_len
                starts = tok_aux["starts"]  # tensor on same device
                P = tok_aux["P"]
                keep = (starts + P) > self.context_len
                # find first kept token index
                if keep.any():
                    first_keep = int(torch.nonzero(keep, as_tuple=False)[0].item())
                    output = output[first_keep:]  # trim tokens
                else:
                    # no token crosses boundary -> drop all (rare edge)
                    output = output[:0]
                # mask `y['mask']` is only used above; no need to adjust it now
            else:
                output = output[self.context_len:]
                y['mask'] = y['mask'][..., self.context_len:]

        if self.tokenization == 'temporal_patch':
            xrec = self.detokenizer(output, tok_aux)  # [bs, J, F, T]
        else:
            xrec = self.output_process(output)  # [bs, J, F, T]
        return xrec

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output


class EmbedTargetLocSingle(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim, num_layers=1):
        super().__init__()
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.target_cond_dim = len(self.extended_goal_joint_names) * 4  # 4 => (x,y,z,is_valid)
        self.latent_dim = latent_dim
        _layers = [nn.Linear(self.target_cond_dim, self.latent_dim)]
        for _ in range(num_layers):
            _layers += [nn.SiLU(), nn.Linear(self.latent_dim, self.latent_dim)]
        self.mlp = nn.Sequential(*_layers)

    def forward(self, input, target_joint_names, target_heading):
        # TODO - generate validity from outside the model
        validity = torch.zeros_like(input)[..., :1]
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[
                sample_idx] else sample_joint_names
            for j in sample_joint_names_w_heading:
                validity[sample_idx, self.extended_goal_joint_names.index(j)] = 1.

        mlp_input = torch.cat([input, validity], dim=-1).view(input.shape[0], -1)
        return self.mlp(mlp_input)


class EmbedTargetLocSplit(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim, num_layers=1):
        super().__init__()
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.target_cond_dim = 4
        self.latent_dim = latent_dim
        self.splited_dim = self.latent_dim // len(self.extended_goal_joint_names)
        assert self.latent_dim % len(self.extended_goal_joint_names) == 0
        self.mini_mlps = nn.ModuleList()
        for _ in self.extended_goal_joint_names:
            _layers = [nn.Linear(self.target_cond_dim, self.splited_dim)]
            for _ in range(num_layers):
                _layers += [nn.SiLU(), nn.Linear(self.splited_dim, self.splited_dim)]
            self.mini_mlps.append(nn.Sequential(*_layers))

    def forward(self, input, target_joint_names, target_heading):
        # TODO - generate validity from outside the model
        validity = torch.zeros_like(input)[..., :1]
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[
                sample_idx] else sample_joint_names
            for j in sample_joint_names_w_heading:
                validity[sample_idx, self.extended_goal_joint_names.index(j)] = 1.

        mlp_input = torch.cat([input, validity], dim=-1)
        mlp_splits = [self.mini_mlps[i](mlp_input[:, i]) for i in range(mlp_input.shape[1])]
        return torch.cat(mlp_splits, dim=-1)


class EmbedTargetLocMulti(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim):
        super().__init__()

        # todo: use a tensor of weight per joint, and another one for biases, then apply a selection in one go like we to for actions
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.extended_goal_joint_idx = {joint_name: idx for idx, joint_name in
                                        enumerate(self.extended_goal_joint_names)}
        self.n_extended_goal_joints = len(self.extended_goal_joint_names)
        self.target_loc_emb = nn.ParameterDict({joint_name:
            nn.Sequential(
                nn.Linear(3, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim))
            for joint_name in self.extended_goal_joint_names})  # todo: check if 3 works for heading and traj
        # nn.Linear(3, latent_dim) for joint_name in self.extended_goal_joint_names})  # todo: check if 3 works for heading and traj
        self.target_all_loc_emb = WeightedSum(
            self.n_extended_goal_joints)  # nn.Linear(self.n_extended_goal_joints, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, input, target_joint_names, target_heading):
        output = torch.zeros((input.shape[0], self.latent_dim), dtype=input.dtype, device=input.device)

        # Iterate over the batch and apply the appropriate filter for each joint
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[
                sample_idx] else sample_joint_names
            output_one_sample = torch.zeros((self.n_extended_goal_joints, self.latent_dim), dtype=input.dtype,
                                            device=input.device)
            for joint_name in sample_joint_names_w_heading:
                layer = self.target_loc_emb[joint_name]
                output_one_sample[self.extended_goal_joint_idx[joint_name]] = layer(
                    input[sample_idx, self.extended_goal_joint_idx[joint_name]])
            output[sample_idx] = self.target_all_loc_emb(output_one_sample)
            # print(torch.where(output_one_sample.sum(axis=1)!=0)[0].cpu().numpy())

        return output


class DiTBlockWithAdaLNZero(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1, activation="gelu", window_size=16):
        super().__init__()

        self.attn = SlidingWindowSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            window_size=window_size
        )

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        """
        x: [seqlen, bs, hidden_size] - input sequence
        c: [bs, hidden_size] - combined conditioning (timestep + text/action)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        residual = x
        x_norm = self.norm1(x)
        x_modulated = x_norm * (1 + scale_msa.unsqueeze(0)) + shift_msa.unsqueeze(0)
        attn_out = self.attn(x_modulated)
        x = residual + gate_msa.unsqueeze(0) * attn_out

        residual = x
        x_norm = self.norm2(x)
        x_modulated = x_norm * (1 + scale_mlp.unsqueeze(0)) + shift_mlp.unsqueeze(0)
        mlp_out = self.mlp(x_modulated)
        x = residual + gate_mlp.unsqueeze(0) * mlp_out

        return x


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization for DiT blocks.
    """

    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)

        self.adaLN_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model, 6 * d_model)  # 4 for adaLN + 2 for alpha
        )

    def forward(self, x, cond_emb):
        params = self.adaLN_mlp(cond_emb)  # [1, bs, 6*d_model]
        params = params.squeeze(0)  # [bs, 6*d_model]

        # Split parameters
        gamma, beta, alpha1, alpha2 = params.chunk(4, dim=-1)  # Each: [bs, d_model]

        x_norm = self.norm(x)
        x = gamma.unsqueeze(0) * x_norm + beta.unsqueeze(0)

        x = alpha1.unsqueeze(0) * x

        return x

class SlidingWindowSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0, window_size=16): #Change here the window size
        super().__init__()
        self.window_size = int(window_size)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )

    def _build_mask(self, seqlen: int, device: torch.device):
        idx = torch.arange(seqlen, device=device)
        dist = idx[:, None] - idx[None, :]
        mask_bool = dist.abs() > self.window_size
        mask = torch.zeros((seqlen, seqlen), device=device, dtype=torch.float32)
        mask = mask.masked_fill(mask_bool, float('-inf'))
        return mask

    def forward(self, x):
        S, B, D = x.shape
        attn_mask = self._build_mask(S, x.device)
        y, _ = self.mha(x, x, x, attn_mask=attn_mask)
        return y

class TemporalPatchTokenizer(nn.Module):
    """
    Tokenizes along time: each token = a temporal patch of P frames (non-overlapping by default).
    """
    def __init__(self, data_rep, njoints, nfeats, latent_dim, patch_size=4, overlap=0):
        super().__init__()
        assert 0 <= overlap < patch_size, "overlap must be in [0, patch_size-1]"
        self.data_rep = data_rep
        self.njoints = njoints
        self.nfeats = nfeats
        self.input_feats = njoints * nfeats
        self.latent_dim = latent_dim
        self.patch_size = int(patch_size)
        self.overlap = int(overlap)
        self.stride = self.patch_size - self.overlap

        # project flattened patch [P * (J*F)] -> latent
        self.proj = nn.Linear(self.patch_size * self.input_feats, self.latent_dim)

    def forward(self, x):
        """
        x: [bs, njoints, nfeats, nframes]
        returns tokens: [seqlen_p, bs, latent_dim], aux for detokenize
        """
        bs, J, F_dim, T = x.shape
        P, S = self.patch_size, self.stride

        # unfold time into patches
        # indices of patch starts: 0, S, 2S, ...
        n_patches = 1 + max(0, (T - P) // S)
        starts = torch.arange(n_patches, device=x.device) * S
        # shape: [n_patches, bs, J, F, P]
        patches = []
        for s in starts:
            patch = x[..., s:s+P]  # [bs, J, F, P]
            if patch.shape[-1] < P:  # pad last if needed
                pad = P - patch.shape[-1]
                patch = F.pad(patch, (0, pad))
            patches.append(patch)
        patches = torch.stack(patches, dim=0)  # [Np, bs, J, F, P]
        patches = patches.permute(0,1,4,2,3)   # [Np, bs, P, J, F]
        patches = patches.reshape(patches.shape[0], bs, P * J * F_dim)  # [Np, bs, P*J*F]

        tokens = self.proj(patches)  # [Np, bs, D]
        return tokens, {"T": T, "starts": starts, "P": P, "S": S}

class TemporalPatchDetokenizer(nn.Module):
    """
    Inverse of TemporalPatchTokenizer (overlap-add if overlap>0).
    """
    def __init__(self, data_rep, njoints, nfeats, latent_dim, patch_size=4, overlap=0):
        super().__init__()
        self.data_rep = data_rep
        self.njoints = njoints
        self.nfeats = nfeats
        self.input_feats = njoints * nfeats
        self.latent_dim = latent_dim
        self.patch_size = int(patch_size)
        self.overlap = int(overlap)
        self.stride = self.patch_size - self.overlap

        self.unproj = nn.Linear(self.latent_dim, self.patch_size * self.input_feats)

    def forward(self, y_tokens, aux):
        Np, bs, D = y_tokens.shape
        T, starts, P, S = aux["T"], aux["starts"], aux["P"], aux["S"]
        J, nfeats = self.njoints, self.nfeats

        patches = self.unproj(y_tokens)                              # [Np, bs, P*J*nfeats]
        patches = patches.view(Np, bs, P, J, nfeats).permute(1,3,4,0,2)  # [bs, J, nfeats, Np, P]

        out = torch.zeros(bs, J, nfeats, T, device=y_tokens.device)
        norm = torch.zeros(T, device=y_tokens.device)
        for i, s in enumerate(starts):
            length = min(P, T - s)
            out[..., s:s+length] += patches[..., i, :length]
            norm[s:s+length] += 1.0
        norm = torch.clamp(norm, min=1.0)
        out = out / norm.view(1,1,1,-1)
        return out
