"""
CTA-E-Branchformer: Text-Dependent / Text-Independent Speaker Verification
============================================================================
Architecture:
  - Module A: Channel-Temporal Attention (CTA) with optional learnable_pooling
  - Module B: 6-block E-Branchformer backbone (MHSA global + DWConv local, parallel)
  - Module C: Weighted Multi-scale Feature Aggregation (MFA)
  - Module D: Multi-head Attentive Statistics Pooling (MHAP)

Dataset: Kathbath
  Structure: Root > Language Folder > Speaker ID Folder > *.npy files
  Paths:
    Train : /home/user2/7th/parikshith_8th/Kathbath_train/
    Test  : /home/user2/7th/parikshith_8th/Kathbath_test/

Reference paper: "An End-to-End Transformer-Based Architecture with
Channel-Temporal Attention for Robust Text-Dependent Speaker Verification"
(Shin et al., Appl. Sci. 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import glob
import random
import logging
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ==================================================================
# CONFIGURATION
# ==================================================================

CONFIG = {
    # --- Paths ---
    'train_dir': '/home/user2/7th/parikshith_8th/Kathbath_train/',
    'test_dir':  '/home/user2/7th/parikshith_8th/Kathbath_test/',
    'log_dir':        '/home/user2/7th/parikshith_8th/logs_ebranchformer_scratch_a4',
    'checkpoint_dir': '/home/user2/7th/parikshith_8th/checkpoints_ebranchformer_scratch_a4',

    # --- Training ---
    'batch_size':   512,
    'num_epochs':   50,
    'learning_rate': 0.001,
    'lr_decay':      0.97,   # StepLR gamma, step_size=1 each epoch
    'num_workers':   8,
    'seed':          42,
    'grad_clip_norm': 5.0,
    'weight_decay': 2e-5,

    # --- Data ---
    'in_channels':        80,   # 80-dim log-mel filterbank
    'train_chunk_frames': 200,  # ~2 s at 10 ms hop

    # --- Model ---
    'd_model':        256,
    'num_heads':      4,
    'num_blocks':     6,
    'ffn_expansion':  4,
    'dw_kernel_size': 31,
    'embedding_dim':  192,
    'dropout':        0.1,
    'conv_subsample_channels': 64,

    # CTA flags
    'cta_learnable_pooling': False,
    'cta_middle_channels':   8,

    # MHAP heads
    'pooling_heads': 4,
    'pooling_bottleneck': 128,

    # --- AAMSoftmax ---
    'aam_margin': 0.2,
    'aam_scale':  30,

    # --- Eval ---
    'eval_pairs':   10000,
    'dcf_p_target': 0.01,
    'dcf_c_miss':   1,
    'dcf_c_fa':     1,
}


# ==================================================================
# UTILITIES
# ==================================================================

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================================================================
# MODULE A: Channel-Temporal Attention (CTA)
# ==================================================================

class CTAModule(nn.Module):
    """
    Channel-Temporal Attention (from CTA-Conformer paper).

    Input:  X  shape (B, C, T, F)  -- channels, time-frames, freq-bins
    Output: Y  shape (B, C, T, F)  -- selectively gated

    learnable_pooling=False  ->  frequency-wise std pooling (paper default)
    learnable_pooling=True   ->  learned weighted std across frequency axis
    """

    def __init__(self, in_channels, middle_channels=8, learnable_pooling=False):
        super().__init__()
        self.learnable_pooling = learnable_pooling

        if learnable_pooling:
            self.register_parameter('freq_weights', None)

        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, in_channels, kernel_size=3, padding=1, bias=False)

    def _initialize_freq_weights(self, f_dim, device):
        if self.freq_weights is None or self.freq_weights.shape[0] != f_dim:
            weights = torch.ones(f_dim, device=device) / f_dim
            self.freq_weights = nn.Parameter(weights)
        return self.freq_weights

    def forward(self, x):
        B, C, T, F_dim = x.shape

        if self.learnable_pooling:
            w_raw = self._initialize_freq_weights(F_dim, x.device)
            w = torch.softmax(w_raw, dim=0)
            w = w.view(1, 1, 1, F_dim)
            mu = (x * w).sum(dim=-1, keepdim=True)
            var = (w * (x - mu) ** 2).sum(dim=-1, keepdim=True)
            z = torch.sqrt(var.clamp(min=1e-9)).squeeze(-1)
        else:
            mu = x.mean(dim=-1, keepdim=True)
            z = torch.sqrt(((x - mu) ** 2).mean(dim=-1).clamp(min=1e-9))

        z = z.unsqueeze(-1)
        d = F.relu(self.bn1(self.conv1(z)))
        omega = torch.sigmoid(self.conv2(d))
        return x * omega


# ==================================================================
# MODULE B: E-Branchformer Block
# ==================================================================

class EBranchformerBlock(nn.Module):
    """
    E-Branchformer block with:
      - Global branch : Multi-Head Self-Attention (MHSA)
      - Local branch  : Depthwise Separable Convolution with GLU
      - Merge         : Concat + pointwise projection
      - Sandwiched by macaron-style half-step FFNs
      - LayerNorm + residuals throughout
    """

    def __init__(self, d_model, num_heads, dw_kernel_size=31, ffn_expansion=4, dropout=0.1, layer_scale=1e-5):
        super().__init__()
        self.d_model = d_model
        self.layer_scale = layer_scale

        self.gamma_ffn1 = nn.Parameter(torch.ones(1) * layer_scale)
        self.gamma_mhsa = nn.Parameter(torch.ones(1) * layer_scale)
        self.gamma_conv = nn.Parameter(torch.ones(1) * layer_scale)
        self.gamma_ffn2 = nn.Parameter(torch.ones(1) * layer_scale)

        self.norm_ffn1 = nn.LayerNorm(d_model)
        self.ffn1 = self._make_ffn(d_model, ffn_expansion, dropout)

        self.norm_global = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout_mhsa = nn.Dropout(dropout)

        self.norm_local = nn.LayerNorm(d_model)
        pad = (dw_kernel_size - 1) // 2
        self.pointwise_in = nn.Linear(d_model, d_model * 2)
        self.dw_conv = nn.Conv1d(d_model, d_model, dw_kernel_size, padding=pad, groups=d_model)
        self.dw_bn = nn.BatchNorm1d(d_model)
        self.pointwise_out = nn.Linear(d_model, d_model)
        self.dropout_conv = nn.Dropout(dropout)

        self.merge_proj = nn.Linear(d_model * 2, d_model)
        self.merge_dropout = nn.Dropout(dropout)

        self.norm_ffn2 = nn.LayerNorm(d_model)
        self.ffn2 = self._make_ffn(d_model, ffn_expansion, dropout)

        self.norm_out = nn.LayerNorm(d_model)

    @staticmethod
    def _make_ffn(d_model, expansion, dropout):
        return nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.gamma_ffn1 * self.ffn1(self.norm_ffn1(x))

        res = x

        x_norm = self.norm_global(x)
        global_out, _ = self.mhsa(x_norm, x_norm, x_norm)
        global_out = self.dropout_mhsa(global_out)

        x_norm2 = self.norm_local(x)
        gate = self.pointwise_in(x_norm2)
        g1, g2 = gate.chunk(2, dim=-1)
        g = g1 * torch.sigmoid(g2)
        g = g.transpose(1, 2)
        g = F.silu(self.dw_bn(self.dw_conv(g)))
        g = g.transpose(1, 2)
        local_out = self.pointwise_out(g)
        local_out = self.dropout_conv(local_out)

        global_out = self.gamma_mhsa * global_out
        local_out = self.gamma_conv * local_out

        merged = torch.cat([global_out, local_out], dim=-1)
        merged = self.merge_dropout(self.merge_proj(merged))
        x = res + merged

        x = x + self.gamma_ffn2 * self.ffn2(self.norm_ffn2(x))

        return self.norm_out(x)


# ==================================================================
# MODULE C: Weighted MFA
# ==================================================================

class WeightedMFA(nn.Module):
    """
    Multi-scale Feature Aggregation with per-layer learnable scalar weights.
    Takes a list of (B, T, d_model) tensors, one per block.
    Returns (B, T, d_model * num_layers) concatenated weighted features.
    """

    def __init__(self, num_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, layer_outputs):
        w = torch.softmax(self.weights, dim=0)
        weighted = [w[i] * layer_outputs[i] for i in range(len(layer_outputs))]
        return torch.cat(weighted, dim=-1)


# ==================================================================
# MODULE D: Multi-Head Attentive Statistics Pooling (MHAP)
# ==================================================================

class MultiHeadAttentiveStatsPool(nn.Module):
    """
    Multi-head version of Attentive Statistics Pooling (ASP).
    Output: (B, in_dim * 2)  -- concatenated weighted mean + std
    """

    def __init__(self, in_dim, bottleneck_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads

        self.attn = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1),
                nn.Tanh(),
                nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1),
            )
            for _ in range(num_heads)
        ])

        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x):
        means, stds = [], []

        for head in self.attn:
            scores = head(x) / self.temperature
            alpha = torch.softmax(scores, dim=2)
            mean = (alpha * x).sum(dim=2)
            var = (alpha * x ** 2).sum(dim=2) - mean ** 2
            std = torch.sqrt(var.clamp(min=1e-9))
            means.append(mean)
            stds.append(std)

        mean_agg = torch.stack(means, dim=0).mean(dim=0)
        std_agg = torch.stack(stds, dim=0).mean(dim=0)

        return torch.cat([mean_agg, std_agg], dim=1)


# ==================================================================
# A4 ABLATION POOLING: plain Temporal Statistics Pooling (MHAP REMOVED)
# ==================================================================
#  Drop-in replacement for MultiHeadAttentiveStatsPool used in the A4
#  ablation. The attention mechanism is removed entirely: no attention
#  heads, no bottleneck, no learnable temperature. Pooling is the plain
#  unweighted mean and standard deviation across the time axis.
#
#  Input : x  (B, C, T)
#  Output: (B, C * 2)  -- concatenated mean + std, matching MHAP's output
#  dimension so the downstream BN / Linear layers are unchanged.
# ==================================================================

class StatsPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        mean = x.mean(dim=2)
        var = x.var(dim=2, unbiased=False)
        std = torch.sqrt(var.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


# ==================================================================
# FULL MODEL: CTA-E-Branchformer
# ==================================================================

class CTAEBranchformer(nn.Module):
    """
    Full speaker verification model.

    Pipeline:
      1. Conv subsampling: (B, T, 80) -> (B, C, T', F')
      2. CTA module (Module A): channel-temporal gating
      3. Flatten freq into channels, project to d_model: (B, T', d_model)
      4. 6 x E-Branchformer blocks (Module B), collecting each block output
      5. Weighted MFA (Module C): (B, T', d_model * num_blocks)
      6. LayerNorm
      7. MHAP (Module D): (B, d_model * num_blocks * 2)
      8. BN + Linear -> embedding-dim embedding + BN
    """

    def __init__(self, cfg):
        super().__init__()
        in_ch = cfg['in_channels']
        d_model = cfg['d_model']
        n_blocks = cfg['num_blocks']
        n_heads = cfg['num_heads']
        dw_k = cfg['dw_kernel_size']
        ffn_exp = cfg['ffn_expansion']
        dropout = cfg['dropout']
        c_sub = cfg['conv_subsample_channels']
        cta_m = cfg['cta_middle_channels']
        cta_lp = cfg['cta_learnable_pooling']
        pool_h = cfg['pooling_heads']
        pool_bottleneck = cfg['pooling_bottleneck']
        embd_dim = cfg['embedding_dim']

        self.conv_sub = nn.Sequential(
            nn.Conv2d(1, c_sub, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c_sub, c_sub, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        f_prime = (in_ch + 3) // 4

        self.cta = CTAModule(c_sub, middle_channels=cta_m, learnable_pooling=cta_lp)

        self.flatten_proj = nn.Linear(c_sub * f_prime, d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            EBranchformerBlock(d_model, n_heads, dw_k, ffn_exp, dropout)
            for _ in range(n_blocks)
        ])

        self.mfa = WeightedMFA(n_blocks)
        mfa_out_dim = d_model * n_blocks

        self.pre_pool_norm = nn.LayerNorm(mfa_out_dim)

        # --- A4 ablation: MHAP REMOVED -> plain temporal statistics pooling ---
        self.pooling = StatsPool()
        pool_out_dim = mfa_out_dim * 2

        self.bn1 = nn.BatchNorm1d(pool_out_dim)
        self.linear = nn.Linear(pool_out_dim, embd_dim)
        self.bn2 = nn.BatchNorm1d(embd_dim)

    def forward(self, x, return_block_outputs=False):
        x = x.unsqueeze(1)
        x = self.conv_sub(x)
        x = self.cta(x)

        B, C, T, F_dim = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, T, C * F_dim)
        x = self.input_dropout(self.flatten_proj(x))

        block_outs = []
        for block in self.blocks:
            x = block(x)
            block_outs.append(x)

        mfa_out = self.mfa(block_outs)
        mfa_out = self.pre_pool_norm(mfa_out)

        mfa_out = mfa_out.transpose(1, 2)
        pooled = self.pooling(mfa_out)

        out = self.bn1(pooled)
        out = self.linear(out)
        out = self.bn2(out)

        if return_block_outputs:
            return out, block_outs
        return out


# ==================================================================
# LOSS: AAMSoftmax
# ==================================================================

class AAMSoftmax(nn.Module):
    def __init__(self, n_class, in_features, m, s):
        super().__init__()
        self.m = m
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(n_class, in_features))
        nn.init.xavier_normal_(self.weight)
        self.ce = nn.CrossEntropyLoss()

        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, x, label):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm)
        one_hot = torch.zeros_like(cosine).scatter_(1, label.view(-1, 1).long(), 1)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return self.ce(output, label)


# ==================================================================
# DATASET  <<<  UPDATED FOR KATHBATH  >>>
# ==================================================================

class KathbathDataset(Dataset):
    """
    Dataset loader for the Kathbath corpus.

    Expected directory layout
    -------------------------
    <root>/
        <language>/          e.g. hindi, kannada, ...
            <speaker_id>/    e.g. 44, 105, ...
                *.npy        pre-extracted 80-dim log-mel features

    Speaker identity policy
    -----------------------
    By default (unique_across_languages=True) the numeric speaker ID is used
    as-is for label assignment, so speaker "44" is the SAME identity regardless
    of which language folder it sits in.

    Set unique_across_languages=False to treat each (language, speaker_id)
    pair as a distinct identity (safe fallback if IDs are not globally unique).
    """

    def __init__(self, root_dir, unique_across_languages=True):
        """
        Parameters
        ----------
        root_dir : str
            Path to the dataset root that contains language sub-folders.
        unique_across_languages : bool
            True  -> speaker label key = speaker_id  (Kathbath IDs are globally unique)
            False -> speaker label key = language/speaker_id  (conservative)
        """
        self.root_dir = root_dir
        self.unique_across_languages = unique_across_languages

        self.speaker_files = []       # list of (npy_path, int_label)
        self.speaker_to_label = {}    # str_key -> int
        self.num_speakers = 0

        # ----------------------------------------------------------------
        # Walk:  root  ->  language  ->  speaker_id  ->  *.npy
        # ----------------------------------------------------------------
        language_dirs = sorted([
            d for d in glob.glob(os.path.join(root_dir, '*'))
            if os.path.isdir(d)
        ])

        if not language_dirs:
            logging.warning('No language-level subdirectories found in %s', root_dir)

        total_files = 0
        for lang_dir in language_dirs:
            lang_name = os.path.basename(lang_dir)

            speaker_dirs = sorted([
                d for d in glob.glob(os.path.join(lang_dir, '*'))
                if os.path.isdir(d)
            ])

            if not speaker_dirs:
                logging.warning('No speaker dirs under language folder: %s', lang_dir)
                continue

            for spk_dir in speaker_dirs:
                spk_id = os.path.basename(spk_dir)   # e.g. "44"

                # Build the key used for label assignment
                if unique_across_languages:
                    label_key = spk_id                 # "44"
                else:
                    label_key = '%s/%s' % (lang_name, spk_id)  # "hindi/44"

                if label_key not in self.speaker_to_label:
                    self.speaker_to_label[label_key] = self.num_speakers
                    self.num_speakers += 1

                label = self.speaker_to_label[label_key]

                # Collect every .npy file directly inside the speaker folder
                npy_files = glob.glob(os.path.join(spk_dir, '*.npy'))

                if not npy_files:
                    logging.debug('No .npy files in %s', spk_dir)
                    continue

                for npy_path in npy_files:
                    self.speaker_files.append((npy_path, label))
                    total_files += 1

        # ----------------------------------------------------------------
        # Summary
        # ----------------------------------------------------------------
        if not self.speaker_files:
            logging.error(
                'CRITICAL: No .npy files found under %s. '
                'Check the directory structure and file extension.', root_dir
            )
        else:
            logging.info(
                'KathbathDataset | root: %s | languages: %d | speakers: %d | files: %d',
                root_dir,
                len(language_dirs),
                self.num_speakers,
                total_files,
            )

    def __len__(self):
        return len(self.speaker_files)

    def __getitem__(self, index):
        return self.speaker_files[index]


# ==================================================================
# DATA CLEANING + COLLATE FUNCTIONS  (unchanged)
# ==================================================================

def _clean_feat(feat):
    """Replace non-finite values with median; return None if all non-finite."""
    if not np.isfinite(feat).all():
        finite_vals = feat[np.isfinite(feat)]
        if len(finite_vals) > 0:
            feat[~np.isfinite(feat)] = np.median(finite_vals)
        else:
            return None
    return feat


def train_collate_fn(batch, num_frames=CONFIG['train_chunk_frames']):
    features, labels = [], []
    for npy_path, label in batch:
        try:
            feat = np.load(npy_path)
            # Handle both (F, T) and (T, F) layouts gracefully
            if feat.ndim == 2 and feat.shape[0] == CONFIG['in_channels']:
                feat = feat.T   # (F, T) -> (T, F)
            feat = _clean_feat(feat)
            if feat is None or np.std(feat) < 1e-6:
                continue
        except Exception as e:
            logging.debug('Error loading %s: %s', npy_path, e)
            continue

        # Chunk / pad to fixed length
        T = feat.shape[0]
        if T < num_frames:
            pad = num_frames - T
            feat = np.pad(feat, ((0, pad), (0, 0)), mode='wrap')
        elif T > num_frames:
            start = random.randint(0, T - num_frames)
            feat = feat[start:start + num_frames]

        features.append(feat)
        labels.append(label)

    if not features:
        return None, None

    return torch.FloatTensor(np.array(features)), torch.LongTensor(labels)


def eval_collate_fn(batch):
    loaded, labels, paths, max_len = [], [], [], 0

    for npy_path, label in batch:
        try:
            feat = np.load(npy_path)
            if feat.ndim == 2 and feat.shape[0] == CONFIG['in_channels']:
                feat = feat.T
            feat = _clean_feat(feat)
            if feat is None:
                feat = np.zeros((1, CONFIG['in_channels']))
        except Exception:
            feat = np.zeros((1, CONFIG['in_channels']))

        loaded.append(feat)
        labels.append(label)
        paths.append(npy_path)
        max_len = max(max_len, feat.shape[0])

    if not loaded:
        return None, None, None

    padded = []
    for feat in loaded:
        if feat.shape[0] < max_len:
            pad = max_len - feat.shape[0]
            feat = np.pad(feat, ((0, pad), (0, 0)), mode='wrap')
        padded.append(feat)

    return (
        torch.FloatTensor(np.array(padded)),
        torch.LongTensor(labels),
        paths,
    )


# ==================================================================
# METRICS
# ==================================================================

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer * 100, thresh


def calculate_min_dcf(y_true, y_scores, p_target=0.01, c_miss=1, c_fa=1):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    dcf = p_target * c_miss * fnr + (1 - p_target) * c_fa * fpr
    return float(np.min(dcf))


# ==================================================================
# TRAIN / EVAL
# ==================================================================

def train_epoch(model, loss_fn, loader, optimizer, device, grad_clip_norm=5.0):
    model.train()
    loss_fn.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (features, labels) in enumerate(loader):
        if features is None:
            continue

        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        emb = model(features)
        loss = loss_fn(emb, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), grad_clip_norm)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            logging.info('  Batch %d/%d  Loss: %.4f',
                         batch_idx + 1, len(loader), loss.item())

    return total_loss / max(n_batches, 1)


def evaluate_model(model, test_dataset, device, num_pairs=10000):
    logging.info('Evaluating... extracting embeddings from %d files.', len(test_dataset))
    model.eval()

    eval_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=eval_collate_fn,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
    )

    embeddings = {}
    speaker_labels = {}

    with torch.no_grad():
        for features, labels, filepaths in eval_loader:
            if features is None:
                continue
            embs = model(features.to(device))
            for i, fpath in enumerate(filepaths):
                embeddings[fpath] = embs[i].cpu()
                speaker_labels[fpath] = labels[i].item()

    logging.info('Extracted %d embeddings.', len(embeddings))

    if len(embeddings) < 2:
        logging.warning('Not enough embeddings for evaluation.')
        return 100.0, 1.0

    all_files = list(embeddings.keys())
    scores, y_true = [], []

    for _ in range(num_pairs):
        is_target = random.choice([True, False])
        f1 = random.choice(all_files)
        l1 = speaker_labels[f1]

        if is_target:
            same_speaker = [f for f in all_files if speaker_labels[f] == l1 and f != f1]
            if not same_speaker:
                continue
            f2 = random.choice(same_speaker)
        else:
            diff_speaker = [f for f in all_files if speaker_labels[f] != l1]
            if not diff_speaker:
                continue
            f2 = random.choice(diff_speaker)

        score = F.cosine_similarity(
            embeddings[f1].unsqueeze(0),
            embeddings[f2].unsqueeze(0),
        ).item()

        scores.append(score)
        y_true.append(1 if is_target else 0)

    if not scores:
        logging.warning('No valid pairs generated.')
        return 100.0, 1.0

    y_s = np.array(scores)
    y_t = np.array(y_true)

    eer, _ = calculate_eer(y_t, y_s)
    min_dcf = calculate_min_dcf(y_t, y_s,
                                CONFIG['dcf_p_target'],
                                CONFIG['dcf_c_miss'],
                                CONFIG['dcf_c_fa'])
    return eer, min_dcf


# ==================================================================
# MAIN
# ==================================================================

def main():
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

    setup_logging(CONFIG['log_dir'])
    set_seed(CONFIG['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Device: %s', device)
    logging.info('Config: %s', CONFIG)

    # --- Datasets  (KathbathDataset replaces VoxCelebDataset) ---
    logging.info('Loading training dataset...')
    train_dataset = KathbathDataset(CONFIG['train_dir'], unique_across_languages=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        drop_last=True,
    )

    logging.info('Loading test dataset...')
    test_dataset = KathbathDataset(CONFIG['test_dir'], unique_across_languages=True)

    num_classes = train_dataset.num_speakers
    logging.info('Speakers for training: %d', num_classes)

    # --- Model ---
    model = CTAEBranchformer(CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('Model parameters: %.2fM', n_params / 1e6)

    # --- Loss ---
    loss_fn = AAMSoftmax(
        n_class=num_classes,
        in_features=CONFIG['embedding_dim'],
        m=CONFIG['aam_margin'],
        s=CONFIG['aam_scale'],
    ).to(device)

    # --- Optimizer and scheduler ---
    all_params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = optim.Adam(all_params,
                           lr=CONFIG['learning_rate'],
                           weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=CONFIG['lr_decay'])

    # --- Training loop ---
    best_eer = 100.0
    best_epoch = 0

    logging.info('=== Starting Training ===')
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        logging.info('--- Epoch %d/%d ---', epoch, CONFIG['num_epochs'])

        avg_loss = train_epoch(model, loss_fn, train_loader, optimizer, device,
                               grad_clip_norm=CONFIG['grad_clip_norm'])
        logging.info('Epoch %d | Train Loss: %.4f', epoch, avg_loss)

        eer, min_dcf = evaluate_model(model, test_dataset, device, CONFIG['eval_pairs'])
        logging.info('Epoch %d | EER: %.4f%%  minDCF: %.4f', epoch, eer, min_dcf)

        if eer < best_eer:
            best_eer = eer
            best_epoch = epoch
            best_ckpt = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eer': eer,
                'min_dcf': min_dcf,
            }, best_ckpt)
            logging.info('New best model saved! EER: %.4f%%', eer)

        ckpt = os.path.join(CONFIG['checkpoint_dir'], 'epoch_%02d.pt' % epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss_fn_state_dict': loss_fn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'eer': eer,
            'min_dcf': min_dcf,
        }, ckpt)
        logging.info('Checkpoint saved to %s', ckpt)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logging.info('LR -> %.6f', current_lr)

    logging.info('=== Training Finished ===')
    logging.info('Best EER: %.4f%% at epoch %d', best_eer, best_epoch)


if __name__ == '__main__':
    main()