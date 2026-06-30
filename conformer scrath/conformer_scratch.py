import os
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
import glob
import random
import logging
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# --- Configuration ---
CONFIG = {
    # ------------------------------------------------------------------ #
    #  UPDATED: Kathbath dataset paths                                     #
    # ------------------------------------------------------------------ #
    'train_dir': '/home/user2/7th/dataset/voxceleb1/train',
    'test_dir':  '/home/user2/7th/dataset/voxceleb1/test',
    # ------------------------------------------------------------------ #
    'log_dir': '/home/user2/7th/parikshith_8th/logs_conformer_scratch1',
    'checkpoint_dir': '/home/user2/7th/parikshith_8th/checkpoints_conformer_scratch1',
    'batch_size': 128,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'lr_decay': 0.97,
    'num_workers': 8,
    'seed': 42,

    # Model & Data
    'in_channels': 80,
    'embedding_dim': 192,
    'model_channels': 512,
    'train_chunk_frames': 200,

    # Speed Perturbation (on-the-fly, no extra data)
    'speed_perturb_ratios': [0.9, 1.0, 1.1],

    # Conformer block config
    'conformer_ff_expansion': 4,
    'conformer_num_heads': 8,
    'conformer_conv_kernel': 31,

    # Multi-Head Attentive Statistics Pooling
    'pooling_heads': 8,
    'pooling_bottleneck': 128,

    # Sub-center AAMSoftmax
    'aam_margin': 0.2,
    'aam_scale': 30,
    'aam_sub_centers': 3,

    # Evaluation
    'eval_pairs': 25000,
    'dcf_p_target': 0.01,
    'dcf_c_miss': 1,
    'dcf_c_fa': 1,

    # AS-Norm
    'asnorm_cohort_size': 200,
    'asnorm_top_n': 200,

    # Memory optimizations
    'use_amp': True,
    'use_grad_checkpoint': True,
}


# ==================================================================
# LOGGING & SEED
# ==================================================================

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # deterministic=True blocks cuDNN algorithm search -> causes the error
    # Remove or set to False; slight non-determinism is acceptable for training
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ==================================================================
# 1. BASE ECAPA BLOCKS
# ==================================================================

class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, bias=False, scale=4):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size, stride,
                      padding, dilation, bias=bias)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(self.width) for _ in range(self.nums)
        ])

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = None
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.bns[i](F.relu(self.convs[i](sp)))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        assert channels % s == 0
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return x * out.unsqueeze(2)


def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
    return nn.Sequential(
        Conv1dReluBn(channels, channels, kernel_size=1),
        Res2Conv1dReluBn(channels, kernel_size, stride, padding,
                         dilation, scale=scale),
        Conv1dReluBn(channels, channels, kernel_size=1),
        SE_Connect(channels)
    )


# ==================================================================
# 2. CONFORMER BLOCKS
# ==================================================================

class ConformerFeedForward(nn.Module):
    """Macaron-style pre-norm feed-forward with 1/2 residual scaling."""
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.drop(F.silu(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return residual + 0.5 * x


class ConformerConvModule(nn.Module):
    """Pointwise -> GLU -> Depthwise conv -> BN -> Swish -> Pointwise."""
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        self.norm = nn.LayerNorm(dim)
        self.pw_conv1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.dw_conv  = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                   padding=(kernel_size - 1) // 2,
                                   groups=dim)
        self.bn       = nn.BatchNorm1d(dim)
        self.pw_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pw_conv1(x)
        x, gate = x.chunk(2, dim=1)
        x = x * torch.sigmoid(gate)
        x = F.silu(self.bn(self.dw_conv(x)))
        x = self.drop(self.pw_conv2(x))
        x = x.transpose(1, 2)
        return residual + x


class ConformerBlock(nn.Module):
    """Full Conformer block: FF (half) -> MHSA -> Conv -> FF (half) -> LN"""
    def __init__(self, dim, num_heads=8, ff_expansion=4,
                 conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1    = ConformerFeedForward(dim, ff_expansion, dropout)
        self.norm_a = nn.LayerNorm(dim)
        self.attn   = nn.MultiheadAttention(dim, num_heads,
                                             dropout=dropout,
                                             batch_first=True)
        self.drop_a = nn.Dropout(dropout)
        self.conv   = ConformerConvModule(dim, conv_kernel, dropout)
        self.ff2    = ConformerFeedForward(dim, ff_expansion, dropout)
        self.norm_o = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.transpose(1, 2)      # -> (batch, time, dim)

        x = self.ff1(x)

        residual = x
        x_n = self.norm_a(x)
        x_a, _ = self.attn(x_n, x_n, x_n)
        x = residual + self.drop_a(x_a)

        x = self.conv(x)
        x = self.ff2(x)
        x = self.norm_o(x)

        x = x.transpose(1, 2)      # -> (batch, dim, time)
        return x


# ==================================================================
# 3. MULTI-HEAD ATTENTIVE STATISTICS POOLING
# ==================================================================

class MultiHeadAttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.key_proj = nn.Conv1d(in_dim, bottleneck_dim * num_heads,
                                   kernel_size=1)
        self.val_proj = nn.Conv1d(bottleneck_dim * num_heads,
                                   num_heads, kernel_size=1)

    def forward(self, x):
        B, C, T = x.shape

        e = torch.tanh(self.key_proj(x))        # (B, bottleneck*H, T)
        alpha = self.val_proj(e)                 # (B, H, T)
        alpha = F.softmax(alpha, dim=2)          # (B, H, T)

        x_e     = x.unsqueeze(1)                 # (B, 1, C, T)
        alpha_e = alpha.unsqueeze(2)             # (B, H, 1, T)

        mean = (alpha_e * x_e).sum(dim=3)        # (B, H, C)
        sq   = (alpha_e * x_e.pow(2)).sum(dim=3)
        std  = (sq - mean.pow(2)).clamp(min=1e-9).sqrt()

        mean = mean.reshape(B, -1)
        std  = std.reshape(B, -1)
        return torch.cat([mean, std], dim=1)     # (B, H*C*2)


# ==================================================================
# 4. FULL ENHANCED MODEL (with optional gradient checkpointing)
# ==================================================================

class ECAPA_Conformer(nn.Module):
    def __init__(self, in_channels, channels, embd_dim,
                 conformer_heads=8, conformer_ff_exp=4,
                 conformer_kernel=31, pooling_heads=8,
                 pooling_bottleneck=128,
                 use_grad_checkpoint=True):
        super().__init__()
        self.use_grad_checkpoint = use_grad_checkpoint

        self.layer1 = Conv1dReluBn(in_channels, channels,
                                    kernel_size=5, padding=2)

        self.se_res2_2 = SE_Res2Block(channels, kernel_size=3, stride=1,
                                       padding=2, dilation=2, scale=8)
        self.conformer2 = ConformerBlock(channels, conformer_heads,
                                          conformer_ff_exp, conformer_kernel)

        self.se_res2_3 = SE_Res2Block(channels, kernel_size=3, stride=1,
                                       padding=3, dilation=3, scale=8)
        self.conformer3 = ConformerBlock(channels, conformer_heads,
                                          conformer_ff_exp, conformer_kernel)

        self.se_res2_4 = SE_Res2Block(channels, kernel_size=3, stride=1,
                                       padding=4, dilation=4, scale=8)
        self.conformer4 = ConformerBlock(channels, conformer_heads,
                                          conformer_ff_exp, conformer_kernel)

        cat_channels = channels * 3
        self.agg_conv = nn.Conv1d(cat_channels, 1536, kernel_size=1)

        self.pooling = MultiHeadAttentiveStatsPool(
            in_dim=1536,
            bottleneck_dim=pooling_bottleneck,
            num_heads=pooling_heads
        )

        pool_out_dim = 1536 * 2 * pooling_heads
        self.bn1    = nn.BatchNorm1d(pool_out_dim)
        self.linear = nn.Linear(pool_out_dim, embd_dim)
        self.bn2    = nn.BatchNorm1d(embd_dim)

    def _run_conformer(self, conformer_block, x):
        if self.use_grad_checkpoint and self.training:
            return grad_checkpoint(conformer_block, x, use_reentrant=False)
        return conformer_block(x)

    def forward(self, x):
        x = x.transpose(1, 2)   # -> (B, in_channels, T)

        out1 = self.layer1(x)

        out2 = self._run_conformer(
            self.conformer2,
            self.se_res2_2(out1) + out1
        )

        out3 = self._run_conformer(
            self.conformer3,
            self.se_res2_3(out1 + out2) + out1 + out2
        )

        out4 = self._run_conformer(
            self.conformer4,
            self.se_res2_4(out1 + out2 + out3) + out1 + out2 + out3
        )

        agg = torch.cat([out2, out3, out4], dim=1)
        agg = F.relu(self.agg_conv(agg))

        pooled = self.pooling(agg)
        out = self.bn1(pooled)
        out = self.linear(out)
        out = self.bn2(out)
        return out


# ==================================================================
# 5. SUB-CENTER AAMSoftmax
# ==================================================================

class SubCenterAAMSoftmax(nn.Module):
    def __init__(self, n_class, in_features, m, s, K=3):
        super().__init__()
        self.in_features = in_features
        self.n_class = n_class
        self.m = m
        self.s = s
        self.K = K

        self.weight = nn.Parameter(
            torch.FloatTensor(n_class * K, in_features)
        )
        nn.init.xavier_normal_(self.weight, gain=1)

        self.ce_loss = nn.CrossEntropyLoss()
        self.cos_m   = float(np.cos(m))
        self.sin_m   = float(np.sin(m))
        self.th      = float(np.cos(np.pi - m))
        self.mm      = float(np.sin(np.pi - m) * m)

    def forward(self, x, label):
        x_n = F.normalize(x, p=2, dim=1, eps=1e-8)
        w_n = F.normalize(self.weight, p=2, dim=1, eps=1e-8)

        cosine_all = F.linear(x_n, w_n)

        cosine = cosine_all.reshape(-1, self.n_class, self.K)
        cosine, _ = cosine.max(dim=2)

        sine  = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi   = cosine * self.cos_m - sine * self.sin_m
        phi   = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = one_hot * phi + (1.0 - one_hot) * cosine
        output = output * self.s

        return self.ce_loss(output, label)


# ==================================================================
# 6. DATASET  <<<  UPDATED FOR KATHBATH STRUCTURE  >>>
# ==================================================================
#
#  Kathbath folder layout:
#    <root>/
#      <language>/          e.g. "hindi", "tamil", ...
#        <speaker_id>/      e.g. "44", "107", ...
#          *.npy            feature files
#
#  Speaker identity:
#    By default speaker IDs are treated as GLOBAL across languages
#    (Kathbath guarantees unique speaker IDs in the released set).
#    If you ever need per-language uniqueness, change the `speaker_key`
#    line below from `sid` to `f"{lang}/{sid}"`.
# ==================================================================

class VoxCelebDataset(Dataset):
    """
    Dataset loader that supports both:
      - VoxCeleb layout  : <root>/<speaker_id>/<session>/<file>_clean.npy
      - Kathbath layout  : <root>/<language>/<speaker_id>/<file>.npy

    The loader auto-detects the layout by checking whether the immediate
    subdirectories of `data_dir` themselves contain subdirectories (Kathbath)
    or .npy files / leaf speaker dirs (VoxCeleb).

    For Kathbath the 'language' level is skipped; only the numeric speaker
    folder is used as the identity key.  Since Kathbath speaker IDs are
    unique across languages, speaker "44" in Hindi and "44" in Tamil
    (if it ever occurred) would map to the same label.  Change `speaker_key`
    to `f"{lang}/{sid}"` for strict per-language separation.
    """

    def __init__(self, data_dir_or_list):
        self.data_dirs = (
            [data_dir_or_list]
            if isinstance(data_dir_or_list, str)
            else data_dir_or_list
        )
        self.speaker_files    = []   # list of (npy_path, int_label)
        self.speaker_to_label = {}   # speaker_key -> int label
        self.num_speakers     = 0

        logging.info("Loading data from: %s", self.data_dirs)

        for data_dir in self.data_dirs:
            if not os.path.isdir(data_dir):
                logging.warning("Directory not found, skipping: %s", data_dir)
                continue

            # ----------------------------------------------------------------
            # Detect layout: list the immediate children of data_dir
            # ----------------------------------------------------------------
            immediate_children = sorted([
                d for d in glob.glob(os.path.join(data_dir, "*"))
                if os.path.isdir(d)
            ])

            if not immediate_children:
                logging.warning("No subdirectories found in %s", data_dir)
                continue

            # Heuristic: if grandchildren exist for the first child, it is
            # either VoxCeleb (speaker -> session -> files) or Kathbath
            # (language -> speaker -> files).
            # We distinguish them by checking whether the grandchild folders
            # contain .npy files directly (Kathbath) or have yet another
            # level (VoxCeleb session dir).
            first_child_children = [
                d for d in glob.glob(
                    os.path.join(immediate_children[0], "*"))
                if os.path.isdir(d)
            ]

            if first_child_children:
                # Check whether grandchildren hold .npy files directly
                first_grandchild_npys = glob.glob(
                    os.path.join(first_child_children[0], "*.npy"))
                if first_grandchild_npys:
                    # ---- Kathbath layout ----
                    logging.info(
                        "Detected KATHBATH layout (lang/speaker/files) "
                        "for %s", data_dir
                    )
                    self._load_kathbath(data_dir)
                else:
                    # ---- VoxCeleb layout ----
                    logging.info(
                        "Detected VOXCELEB layout (speaker/session/files) "
                        "for %s", data_dir
                    )
                    self._load_voxceleb(data_dir)
            else:
                # Immediate children directly contain .npy (flat structure)
                logging.info(
                    "Detected FLAT layout (speaker/files) for %s", data_dir
                )
                self._load_flat(data_dir)

        if not self.speaker_files:
            logging.error(
                "CRITICAL: No .npy files found in any of %s. "
                "Training will fail.", self.data_dirs
            )
        else:
            logging.info(
                "Loaded %d files from %d speakers.",
                len(self.speaker_files), self.num_speakers
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_label(self, speaker_key):
        """Return existing int label or allocate a new one."""
        if speaker_key not in self.speaker_to_label:
            self.speaker_to_label[speaker_key] = self.num_speakers
            self.num_speakers += 1
        return self.speaker_to_label[speaker_key]

    def _load_kathbath(self, data_dir):
        """
        Traverse: data_dir / <language> / <speaker_id> / *.npy

        Speaker key = speaker_id  (global across languages).
        Switch to f"{lang}/{sid}" for per-language uniqueness.
        """
        lang_dirs = sorted([
            d for d in glob.glob(os.path.join(data_dir, "*"))
            if os.path.isdir(d)
        ])
        file_count = 0
        for lang_dir in lang_dirs:
            lang = os.path.basename(lang_dir)
            speaker_dirs = sorted([
                d for d in glob.glob(os.path.join(lang_dir, "*"))
                if os.path.isdir(d)
            ])
            for speaker_dir in speaker_dirs:
                sid = os.path.basename(speaker_dir)

                # ---- speaker key: global (change to f"{lang}/{sid}" if needed) ----
                speaker_key = sid

                label = self._get_or_create_label(speaker_key)

                # Collect all .npy files directly inside the speaker folder
                npy_files = (
                    glob.glob(os.path.join(speaker_dir, "*_clean.npy")) +
                    glob.glob(os.path.join(speaker_dir, "*.npy"))
                )
                # De-duplicate in case both patterns overlap
                npy_files = list(dict.fromkeys(npy_files))

                for npy_file in npy_files:
                    self.speaker_files.append((npy_file, label))
                    file_count += 1

        logging.info(
            "  [Kathbath] %s: %d language dirs, %d files",
            data_dir, len(lang_dirs), file_count
        )

    def _load_voxceleb(self, data_dir):
        """
        Traverse: data_dir / <speaker_id> / <session> / *_clean.npy
        Mirrors original VoxCelebDataset behaviour exactly.
        """
        speaker_dirs = sorted([
            d for d in glob.glob(os.path.join(data_dir, "*"))
            if os.path.isdir(d)
        ])
        file_count = 0
        for speaker_dir in speaker_dirs:
            sid   = os.path.basename(speaker_dir)
            label = self._get_or_create_label(sid)
            for npy_file in glob.glob(
                    os.path.join(speaker_dir, "*", "*_clean.npy")):
                self.speaker_files.append((npy_file, label))
                file_count += 1

        logging.info(
            "  [VoxCeleb] %s: %d speaker dirs, %d files",
            data_dir, len(speaker_dirs), file_count
        )

    def _load_flat(self, data_dir):
        """
        Traverse: data_dir / <speaker_id> / *.npy  (no session level)
        """
        speaker_dirs = sorted([
            d for d in glob.glob(os.path.join(data_dir, "*"))
            if os.path.isdir(d)
        ])
        file_count = 0
        for speaker_dir in speaker_dirs:
            sid   = os.path.basename(speaker_dir)
            label = self._get_or_create_label(sid)
            npy_files = (
                glob.glob(os.path.join(speaker_dir, "*_clean.npy")) +
                glob.glob(os.path.join(speaker_dir, "*.npy"))
            )
            npy_files = list(dict.fromkeys(npy_files))
            for npy_file in npy_files:
                self.speaker_files.append((npy_file, label))
                file_count += 1

        logging.info(
            "  [Flat] %s: %d speaker dirs, %d files",
            data_dir, len(speaker_dirs), file_count
        )

    # ------------------------------------------------------------------
    # PyTorch Dataset interface  (unchanged)
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.speaker_files)

    def __getitem__(self, index):
        return self.speaker_files[index]


# ==================================================================
# 7. SPEED PERTURBATION
# ==================================================================

def speed_perturb(feat, ratio, target_frames):
    if ratio == 1.0:
        return feat
    T, F = feat.shape
    new_T = max(1, int(round(T / ratio)))
    x_old = np.linspace(0, T - 1, T)
    x_new = np.linspace(0, T - 1, new_T)
    resampled = np.stack([
        np.interp(x_new, x_old, feat[:, f]) for f in range(F)
    ], axis=1)
    return resampled


def load_and_clean(npy_path):
    feat = np.load(npy_path)   # stored as (F, T)
    feat = feat.T              # -> (T, F)
    if not np.isfinite(feat).all():
        finite_vals = feat[np.isfinite(feat)]
        median_val  = np.median(finite_vals) if len(finite_vals) > 0 else 0.0
        feat[~np.isfinite(feat)] = median_val
    return feat


def train_collate_fn(batch,
                     num_frames=CONFIG['train_chunk_frames'],
                     speed_ratios=CONFIG['speed_perturb_ratios']):
    features, labels = [], []
    for npy_path, label in batch:
        try:
            feat = load_and_clean(npy_path)
        except Exception:
            continue

        if np.std(feat) < 1e-6:
            continue

        ratio = random.choice(speed_ratios)
        feat  = speed_perturb(feat, ratio, num_frames)

        T = feat.shape[0]
        if T < num_frames:
            feat = np.pad(feat, ((0, num_frames - T), (0, 0)), mode='wrap')
        elif T > num_frames:
            start = random.randint(0, T - num_frames)
            feat  = feat[start: start + num_frames]

        features.append(feat)
        labels.append(label)

    if not features:
        return None, None

    return (torch.FloatTensor(np.array(features)),
            torch.LongTensor(labels))


def eval_collate_fn(batch):
    features, labels, filepaths = [], [], []
    loaded, max_len = [], 0
    for npy_path, label in batch:
        try:
            feat = load_and_clean(npy_path)
        except Exception as e:
            logging.warning("Eval load failed %s: %s", npy_path, e)
            continue
        loaded.append(feat)
        labels.append(label)
        filepaths.append(npy_path)
        max_len = max(max_len, feat.shape[0])

    if not loaded:
        return None, None, None

    for feat in loaded:
        pad = max_len - feat.shape[0]
        features.append(
            np.pad(feat, ((0, pad), (0, 0)), mode='wrap') if pad > 0 else feat
        )

    return (torch.FloatTensor(np.array(features)),
            torch.LongTensor(labels),
            filepaths)


# ==================================================================
# 8. METRICS
# ==================================================================

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer * 100, thresh


def calculate_min_dcf(y_true, y_scores,
                       p_target=0.01, c_miss=1, c_fa=1):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    dcf = p_target * c_miss * fnr + (1 - p_target) * c_fa * fpr
    return float(np.min(dcf))


# ==================================================================
# 9. AS-NORM
# ==================================================================

def build_asnorm_stats(embeddings_matrix, cohort_matrix, top_n):
    scores = torch.mm(
        F.normalize(embeddings_matrix, dim=1),
        F.normalize(cohort_matrix, dim=1).T
    )
    top_scores, _ = scores.topk(min(top_n, scores.size(1)), dim=1)
    mu  = top_scores.mean(dim=1)
    std = top_scores.std(dim=1).clamp(min=1e-9)
    return mu, std


def asnorm_score(raw_score, mu1, std1, mu2, std2):
    return 0.5 * ((raw_score - mu1) / std1 + (raw_score - mu2) / std2)


# ==================================================================
# 10. TRAINING EPOCH
# ==================================================================

def train_epoch(model, loss_fn, data_loader, optimizer, device, scaler,
                use_amp=True):
    model.train()
    loss_fn.train()
    total_loss, total_batches = 0.0, 0

    for features, labels in data_loader:
        if features is None:
            continue
        features = features.to(device, non_blocking=True)
        labels   = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast('cuda', enabled=use_amp):
            embeddings = model(features)
            loss = loss_fn(embeddings, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(loss_fn.parameters()), 5.0
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss   += loss.item()
        total_batches += 1
        if total_batches % 50 == 0:
            logging.info("  Batch %d/%d, Loss: %.4f",
                         total_batches, len(data_loader), loss.item())

    return total_loss / max(total_batches, 1)


# ==================================================================
# 11. EVALUATION (with AS-Norm)
# ==================================================================

def evaluate_model(model, test_dataset, device,
                   num_pairs=10000,
                   cohort_size=CONFIG['asnorm_cohort_size'],
                   top_n=CONFIG['asnorm_top_n'],
                   use_amp=True):
    logging.info("Evaluation: extracting embeddings from %d files...",
                 len(test_dataset))
    model.eval()

    eval_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=eval_collate_fn,
        num_workers=CONFIG['num_workers']
    )

    embeddings     = {}
    speaker_labels = {}

    with torch.no_grad():
        for features, labels, filepaths in eval_loader:
            if features is None:
                continue
            with autocast('cuda', enabled=use_amp):
                batch_emb = model(features.to(device, non_blocking=True))
            batch_emb = batch_emb.float()
            for i, fp in enumerate(filepaths):
                embeddings[fp]     = batch_emb[i].cpu()
                speaker_labels[fp] = labels[i].item()

    logging.info("Extracted %d embeddings.", len(embeddings))
    if len(embeddings) < 2:
        logging.warning("Not enough embeddings for evaluation.")
        return 0.0, 0.0

    all_files = list(embeddings.keys())

    cohort_size   = min(cohort_size, len(all_files))
    cohort_files  = random.sample(all_files, cohort_size)
    cohort_matrix = torch.stack([embeddings[f] for f in cohort_files])
    all_emb_matrix = torch.stack([embeddings[f] for f in all_files])

    logging.info("Computing AS-Norm stats over %d cohort utterances...",
                 cohort_size)
    file_to_idx = {f: i for i, f in enumerate(all_files)}
    mu_all, std_all = build_asnorm_stats(all_emb_matrix, cohort_matrix, top_n)

    logging.info("Generating %d random trial pairs...", num_pairs)
    scores, y_true = [], []

    for _ in range(num_pairs):
        is_target = random.choice([True, False])
        f1 = f2 = None

        attempts = 0
        while f1 == f2 or f1 is None or f2 is None:
            attempts += 1
            if attempts > 200:
                break
            if is_target:
                f1    = random.choice(all_files)
                label1 = speaker_labels[f1]
                same  = [f for f, l in speaker_labels.items() if l == label1]
                if len(same) < 2:
                    continue
                f2 = random.choice(same)
            else:
                f1    = random.choice(all_files)
                label1 = speaker_labels[f1]
                diff  = [f for f, l in speaker_labels.items() if l != label1]
                if not diff:
                    continue
                f2 = random.choice(diff)

        if f1 is None or f2 is None or f1 == f2:
            continue

        emb1 = embeddings[f1]
        emb2 = embeddings[f2]

        raw = F.cosine_similarity(emb1.unsqueeze(0),
                                   emb2.unsqueeze(0)).item()

        idx1 = file_to_idx[f1]
        idx2 = file_to_idx[f2]
        norm_score = asnorm_score(
            raw,
            mu_all[idx1].item(), std_all[idx1].item(),
            mu_all[idx2].item(), std_all[idx2].item()
        )

        scores.append(norm_score)
        y_true.append(1 if is_target else 0)

    if not scores:
        logging.error("No valid pairs generated.")
        return 0.0, 0.0

    y_scores_np = np.array(scores)
    y_true_np   = np.array(y_true)

    eer, _   = calculate_eer(y_true_np, y_scores_np)
    min_dcf  = calculate_min_dcf(
        y_true_np, y_scores_np,
        p_target=CONFIG['dcf_p_target'],
        c_miss=CONFIG['dcf_c_miss'],
        c_fa=CONFIG['dcf_c_fa']
    )
    return eer, min_dcf


# ==================================================================
# 12. MAIN
# ==================================================================

def main():
    os.makedirs(CONFIG['log_dir'],        exist_ok=True)
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    setup_logging(CONFIG['log_dir'])
    set_seed(CONFIG['seed'])

    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Device: %s", device)
    logging.info("Config: %s", CONFIG)

    logging.info("Loading training data...")
    train_dataset = VoxCelebDataset(CONFIG['train_dir'])
    train_loader  = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    logging.info("Loading test data...")
    test_dataset = VoxCelebDataset(CONFIG['test_dir'])
    num_classes  = train_dataset.num_speakers
    logging.info("Training speakers: %d", num_classes)

    logging.info("Initializing ECAPA-Conformer model...")
    model = ECAPA_Conformer(
        in_channels         = CONFIG['in_channels'],
        channels            = CONFIG['model_channels'],
        embd_dim            = CONFIG['embedding_dim'],
        conformer_heads     = CONFIG['conformer_num_heads'],
        conformer_ff_exp    = CONFIG['conformer_ff_expansion'],
        conformer_kernel    = CONFIG['conformer_conv_kernel'],
        pooling_heads       = CONFIG['pooling_heads'],
        pooling_bottleneck  = CONFIG['pooling_bottleneck'],
        use_grad_checkpoint = CONFIG['use_grad_checkpoint'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info("Model parameters: %.2fM", total_params)
    logging.info("AMP enabled: %s", CONFIG['use_amp'])
    logging.info("Gradient checkpointing: %s", CONFIG['use_grad_checkpoint'])

    loss_fn = SubCenterAAMSoftmax(
        n_class    = num_classes,
        in_features= CONFIG['embedding_dim'],
        m          = CONFIG['aam_margin'],
        s          = CONFIG['aam_scale'],
        K          = CONFIG['aam_sub_centers']
    ).to(device)

    all_params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer  = optim.Adam(all_params,
                             lr=CONFIG['learning_rate'],
                             weight_decay=2e-5)
    scheduler  = optim.lr_scheduler.StepLR(optimizer,
                                            step_size=1,
                                            gamma=CONFIG['lr_decay'])

    scaler = GradScaler(enabled=CONFIG['use_amp'])

    logging.info("=== Starting Training ===")
    best_eer = float('inf')

    for epoch in range(1, CONFIG['num_epochs'] + 1):
        logging.info("--- Epoch %d/%d ---", epoch, CONFIG['num_epochs'])

        avg_loss = train_epoch(
            model, loss_fn, train_loader, optimizer, device,
            scaler, use_amp=CONFIG['use_amp']
        )
        logging.info("Epoch %d | Avg Train Loss: %.4f", epoch, avg_loss)

        eer, min_dcf = evaluate_model(
            model, test_dataset, device,
            num_pairs   = CONFIG['eval_pairs'],
            cohort_size = CONFIG['asnorm_cohort_size'],
            top_n       = CONFIG['asnorm_top_n'],
            use_amp     = CONFIG['use_amp']
        )
        logging.info("Epoch %d | EER: %.4f%%  minDCF: %.4f",
                     epoch, eer, min_dcf)

        ckpt_path = os.path.join(CONFIG['checkpoint_dir'],
                                  f"epoch_{epoch}.pt")
        torch.save({
            'epoch':              epoch,
            'model_state_dict':   model.state_dict(),
            'loss_fn_state_dict': loss_fn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict':  scaler.state_dict(),
            'eer':                eer,
            'min_dcf':            min_dcf,
        }, ckpt_path)
        logging.info("Checkpoint saved: %s", ckpt_path)

        if eer < best_eer:
            best_eer = eer
            best_path = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pt')
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'eer':              eer,
                'min_dcf':          min_dcf,
            }, best_path)
            logging.info("New best EER: %.4f%% -- saved best_model.pt", best_eer)

        scheduler.step()
        logging.info("LR updated to: %.6f", scheduler.get_last_lr()[0])

    logging.info("=== Training Finished ===")


if __name__ == "__main__":
    main()