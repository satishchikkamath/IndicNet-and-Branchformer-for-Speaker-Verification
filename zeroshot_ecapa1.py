# -*- coding: utf-8 -*-
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

# --- Configuration ---
CONFIG = {
    'train_dir': '/home/user2/7th/parikshith_8th/Kathbath_train/',
    'test_dir': '/home/user2/7th/parikshith_8th/Kathbath_test/',
    'log_dir': '/home/user2/7th/parikshith_8th/logs_ecapakath_zeroshot',
    'checkpoint_dir': '/home/user2/7th/parikshith_8th/checkpoints2_ecapakath_zeroshot',
    'batch_size': 256,
    'num_epochs': 20,            # Fine-tuning needs far fewer epochs
    'learning_rate': 1e-4,       # Lower LR for fine-tuning
    'lr_decay': 0.97,
    'num_workers': 8,
    'seed': 42,

    # --- Model & Data Params ---
    'in_channels': 80,
    'embedding_dim': 192,
    'model_channels': 512,
    'train_chunk_frames': 200,

    # --- Pretrained Weights ---
    # Path to a pretrained ECAPA-TDNN checkpoint (e.g. trained on VoxCeleb2).
    # Set to None to skip loading pretrained weights.
    'pretrained_checkpoint': '/home/user2/pretrained/ecapa_voxceleb2.pt',

    # --- Fine-tuning Strategy ---
    # 'full'       : Update ALL parameters (backbone + classifier head).
    # 'head_only'  : Freeze backbone; only train the new AAMSoftmax head.
    # 'last_block' : Freeze everything except the last SE-Res2Block + head.
    'finetune_mode': 'full',

    # --- AAMSoftmax Params ---
    'aam_margin': 0.2,
    'aam_scale': 30,

    # --- Evaluation Params ---
    'eval_pairs': 10000,
    'dcf_p_target': 0.01,
    'dcf_c_miss': 1,
    'dcf_c_fa': 1
}

# --- Setup Logging ---
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

# --- Set Random Seed ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================================================================
# 1. MODEL DEFINITION  (unchanged architecture)
# ==================================================================

class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, scale=4):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.bns[i](F.relu(self.convs[i](sp)))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)

class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
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
        Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
        Conv1dReluBn(channels, channels, kernel_size=1),
        SE_Connect(channels)
    )

class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        alpha = torch.softmax(self.linear2(torch.tanh(self.linear1(x))), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        std = torch.sqrt((torch.sum(alpha * x ** 2, dim=2) - mean ** 2).clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)

class ECAPA_TDNN(nn.Module):
    def __init__(self, in_channels, channels, embd_dim):
        super().__init__()
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        self.conv = nn.Conv1d(channels * 3, 1536, kernel_size=1)
        self.pooling = AttentiveStatsPool(1536, 128)
        self.bn1 = nn.BatchNorm1d(3072)
        self.linear = nn.Linear(3072, embd_dim)
        self.bn2 = nn.BatchNorm1d(embd_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3
        out = F.relu(self.conv(torch.cat([out2, out3, out4], dim=1)))
        out = self.bn1(self.pooling(out))
        out = self.bn2(self.linear(out))
        return out

# ==================================================================
# 2. PRETRAINED WEIGHT LOADING
# ==================================================================

def load_pretrained_ecapa(model: ECAPA_TDNN, checkpoint_path: str) -> ECAPA_TDNN:
    """
    Load pretrained ECAPA-TDNN backbone weights, ignoring mismatched keys.

    The checkpoint may come from:
      - A full training checkpoint saved by this codebase
        (keys under 'model_state_dict')
      - A third-party pretrained model (bare state_dict)

    The final classification layer (linear / bn2) is intentionally excluded
    because the new dataset has a different number of speakers - those weights
    are re-initialised from scratch.
    """
    if not os.path.isfile(checkpoint_path):
        logging.warning(
            "Pretrained checkpoint not found at '%s'. "
            "Model will be initialised from scratch.", checkpoint_path
        )
        return model

    logging.info("Loading pretrained weights from '%s' ...", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Support both bare state_dicts and full training checkpoints
    if 'model_state_dict' in ckpt:
        src_state = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        src_state = ckpt['state_dict']
    else:
        src_state = ckpt  # assume bare state_dict

    # Strip any 'module.' prefix from DataParallel-saved models
    src_state = {k.replace('module.', ''): v for k, v in src_state.items()}

    dst_state = model.state_dict()
    matched, skipped = {}, []

    for k, v in src_state.items():
        if k in dst_state and dst_state[k].shape == v.shape:
            matched[k] = v
        else:
            skipped.append(k)

    dst_state.update(matched)
    model.load_state_dict(dst_state)

    logging.info(
        "Loaded %d / %d parameters. Skipped keys (%d): %s",
        len(matched), len(src_state), len(skipped),
        skipped[:10]  # show at most 10 skipped keys for brevity
    )
    return model

# ==================================================================
# 3. FINE-TUNING STRATEGY  (freeze / unfreeze layers)
# ==================================================================

def apply_finetune_strategy(model: ECAPA_TDNN, mode: str) -> None:
    """
    Freeze parts of the backbone depending on `mode`.

    Parameters
    ----------
    mode : str
        'full'       - All parameters trainable (full fine-tuning).
        'head_only'  - Only the projection head (linear + bn2) is trainable.
        'last_block' - layer4 + projection head are trainable; rest frozen.
    """
    if mode == 'full':
        for p in model.parameters():
            p.requires_grad = True
        logging.info("Fine-tune mode: FULL - all parameters trainable.")
        return

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    if mode == 'head_only':
        for p in model.linear.parameters():
            p.requires_grad = True
        for p in model.bn2.parameters():
            p.requires_grad = True
        logging.info("Fine-tune mode: HEAD_ONLY - only linear + bn2 trainable.")

    elif mode == 'last_block':
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.conv.parameters():
            p.requires_grad = True
        for p in model.pooling.parameters():
            p.requires_grad = True
        for p in model.bn1.parameters():
            p.requires_grad = True
        for p in model.linear.parameters():
            p.requires_grad = True
        for p in model.bn2.parameters():
            p.requires_grad = True
        logging.info("Fine-tune mode: LAST_BLOCK - layer4 + head trainable.")

    else:
        logging.warning("Unknown finetune_mode '%s'. Defaulting to FULL.", mode)
        for p in model.parameters():
            p.requires_grad = True

# ==================================================================
# 4. ZERO-SHOT INFERENCE HELPER
# ==================================================================

@torch.no_grad()
def extract_embedding(model: ECAPA_TDNN, npy_path: str, device: torch.device) -> torch.Tensor:
    """
    Extract a single L2-normalised speaker embedding from a .npy feature file.
    The full utterance is used (no chunking) for zero-shot inference.
    """
    feat = np.load(npy_path).T          # (time, features)
    feat_tensor = torch.FloatTensor(feat).unsqueeze(0).to(device)  # (1, T, F)
    emb = model(feat_tensor)            # (1, embd_dim)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.squeeze(0).cpu()         # (embd_dim,)


@torch.no_grad()
def zero_shot_verify(
    model: ECAPA_TDNN,
    npy_path_1: str,
    npy_path_2: str,
    device: torch.device,
    threshold: float = 0.25
) -> dict:
    """
    Zero-shot speaker verification between two utterances.

    Returns
    -------
    dict with keys:
        score      : float  cosine similarity score
        decision   : bool   True == same speaker
        threshold  : float  threshold used
    """
    model.eval()
    emb1 = extract_embedding(model, npy_path_1, device)
    emb2 = extract_embedding(model, npy_path_2, device)
    score = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    return {
        'score': score,
        'decision': score >= threshold,
        'threshold': threshold
    }

# ==================================================================
# 5. LOSS FUNCTION (AAMSoftmax - unchanged)
# ==================================================================

class AAMSoftmax(nn.Module):
    def __init__(self, n_class, in_features, m, s):
        super().__init__()
        self.m = m
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(n_class, in_features))
        nn.init.xavier_normal_(self.weight, gain=1)
        self.ce_loss = nn.CrossEntropyLoss()
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, x, label):
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
        w_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-8)
        cosine = F.linear(x_norm, w_norm)

        one_hot = torch.zeros_like(cosine).scatter_(1, label.view(-1, 1).long(), 1)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return self.ce_loss(output * self.s, label)

# ==================================================================
# 6. DATASET AND DATALOADER  (unchanged)
# ==================================================================

class KathbathDataset(Dataset):
    """
    root/
      <language>/
        <speaker_id>/
          *.npy
    """
    def __init__(self, root_dir, unique_across_languages=False):
        self.root_dirs = [root_dir] if isinstance(root_dir, str) else list(root_dir)
        self.unique_across_languages = unique_across_languages
        self.speaker_files = []
        self.speaker_to_label = {}
        self.num_speakers = 0

        logging.info("Loading Kathbath data from: %s", self.root_dirs)

        for root in self.root_dirs:
            language_dirs = sorted([d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)])
            if not language_dirs:
                logging.warning("No language sub-folders found in %s", root)
                continue

            for lang_dir in language_dirs:
                language = os.path.basename(lang_dir)
                speaker_dirs = sorted([d for d in glob.glob(os.path.join(lang_dir, "*")) if os.path.isdir(d)])

                for spk_dir in speaker_dirs:
                    speaker_id = os.path.basename(spk_dir)
                    speaker_key = speaker_id if self.unique_across_languages else "%s/%s" % (language, speaker_id)

                    if speaker_key not in self.speaker_to_label:
                        self.speaker_to_label[speaker_key] = self.num_speakers
                        self.num_speakers += 1

                    label = self.speaker_to_label[speaker_key]
                    for npy_file in glob.glob(os.path.join(spk_dir, "*.npy")):
                        self.speaker_files.append((npy_file, label))

        if not self.speaker_files:
            logging.error("CRITICAL: No .npy files found. Check paths and directory structure.")
        else:
            logging.info("Loaded %d utterances from %d speakers.", len(self.speaker_files), self.num_speakers)

    def __len__(self):
        return len(self.speaker_files)

    def __getitem__(self, index):
        return self.speaker_files[index]


def _safe_load_npy(npy_path):
    """Load and sanitise a .npy feature array. Returns None on failure."""
    try:
        feat = np.load(npy_path).T          # (time, features)
        if not np.isfinite(feat).all():
            finite_vals = feat[np.isfinite(feat)]
            feat[~np.isfinite(feat)] = np.median(finite_vals) if len(finite_vals) else 0.0
        if np.std(feat) < 1e-6:
            return None
        return feat
    except Exception:
        return None


def train_collate_fn(batch, num_frames=CONFIG['train_chunk_frames']):
    features, labels = [], []
    for npy_path, label in batch:
        feat = _safe_load_npy(npy_path)
        if feat is None:
            continue
        if feat.shape[0] < num_frames:
            feat = np.pad(feat, ((0, num_frames - feat.shape[0]), (0, 0)), mode='wrap')
        elif feat.shape[0] > num_frames:
            start = random.randint(0, feat.shape[0] - num_frames)
            feat = feat[start: start + num_frames]
        features.append(feat)
        labels.append(label)

    if not features:
        return None, None
    return torch.FloatTensor(np.array(features)), torch.LongTensor(labels)


def eval_collate_fn(batch):
    loaded, labels, filepaths, max_len = [], [], [], 0
    for npy_path, label in batch:
        feat = _safe_load_npy(npy_path)
        if feat is None:
            continue
        loaded.append(feat)
        labels.append(label)
        filepaths.append(npy_path)
        max_len = max(max_len, feat.shape[0])

    if not loaded:
        return None, None, None

    padded = [
        np.pad(f, ((0, max_len - f.shape[0]), (0, 0)), mode='wrap') if f.shape[0] < max_len else f
        for f in loaded
    ]
    return torch.FloatTensor(np.array(padded)), torch.LongTensor(labels), filepaths

# ==================================================================
# 7. METRICS  (unchanged)
# ==================================================================

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100, interp1d(fpr, thresholds)(eer)

def calculate_min_dcf(y_true, y_scores, p_target=0.01, c_miss=1, c_fa=1):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    return np.min(p_target * c_miss * fnr + (1 - p_target) * c_fa * fpr)

# ==================================================================
# 8. TRAIN / EVAL  (unchanged logic)
# ==================================================================

def train_epoch(model, loss_fn, data_loader, optimizer, device):
    model.train()
    loss_fn.train()
    total_loss, total_batches = 0, 0

    for features, labels in data_loader:
        if features is None:
            continue
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(features), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
        if total_batches % 50 == 0:
            logging.info("  Batch %d/%d, Loss: %.4f", total_batches, len(data_loader), loss.item())

    return total_loss / max(total_batches, 1)


def evaluate_model(model, test_dataset, device, num_pairs=10000):
    logging.info("Evaluating - extracting embeddings from %d test files.", len(test_dataset))
    model.eval()

    eval_loader = DataLoader(
        test_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
        collate_fn=eval_collate_fn, num_workers=CONFIG['num_workers']
    )

    embeddings, speaker_labels = {}, {}
    with torch.no_grad():
        for features, labels, filepaths in eval_loader:
            if features is None:
                continue
            batch_embs = model(features.to(device))
            for i, fpath in enumerate(filepaths):
                embeddings[fpath] = batch_embs[i].cpu()
                speaker_labels[fpath] = labels[i].item()

    logging.info("Extracted %d embeddings.", len(embeddings))
    if len(embeddings) < 2:
        logging.warning("Not enough embeddings for evaluation. Skipping.")
        return 0, 0

    all_files = list(embeddings.keys())
    scores, y_true = [], []

    for _ in range(num_pairs):
        is_target = random.choice([True, False])
        f1 = f2 = None

        while f1 == f2:
            f1 = random.choice(all_files)
            label1 = speaker_labels[f1]
            if is_target:
                pool = [f for f, l in speaker_labels.items() if l == label1]
                if len(pool) < 2:
                    continue
            else:
                pool = [f for f, l in speaker_labels.items() if l != label1]
                if not pool:
                    continue
            f2 = random.choice(pool)

        if f1 is None or f2 is None:
            continue

        score = F.cosine_similarity(
            embeddings[f1].unsqueeze(0), embeddings[f2].unsqueeze(0)
        ).item()
        scores.append(score)
        y_true.append(1 if is_target else 0)

    if not scores:
        logging.error("No valid pairs generated. Skipping metrics.")
        return 0, 0

    y_scores_np = np.array(scores)
    y_true_np = np.array(y_true)
    eer, _ = calculate_eer(y_true_np, y_scores_np)
    min_dcf = calculate_min_dcf(
        y_true_np, y_scores_np,
        p_target=CONFIG['dcf_p_target'],
        c_miss=CONFIG['dcf_c_miss'],
        c_fa=CONFIG['dcf_c_fa']
    )
    return eer, min_dcf

# ==================================================================
# 9. MAIN - zero-shot evaluation + optional fine-tuning
# ==================================================================

def main():
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    setup_logging(CONFIG['log_dir'])
    set_seed(CONFIG['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Device: %s", device)
    logging.info("Config: %s", CONFIG)

    # ---- Build model ----
    model = ECAPA_TDNN(
        in_channels=CONFIG['in_channels'],
        channels=CONFIG['model_channels'],
        embd_dim=CONFIG['embedding_dim']
    ).to(device)

    # ---- Load pretrained backbone ----
    if CONFIG.get('pretrained_checkpoint'):
        model = load_pretrained_ecapa(model, CONFIG['pretrained_checkpoint'])

    # ---- Zero-shot evaluation (no training) ----
    logging.info("=== Zero-Shot Evaluation (pretrained model, no fine-tuning) ===")
    test_dataset = KathbathDataset(CONFIG['test_dir'], unique_across_languages=False)
    eer_zs, dcf_zs = evaluate_model(model, test_dataset, device, num_pairs=CONFIG['eval_pairs'])
    logging.info("Zero-Shot  |  EER: %.4f%%  |  minDCF: %.4f", eer_zs, dcf_zs)

    # ---- Fine-tuning ----
    logging.info("=== Fine-Tuning on Kathbath ===")

    # Freeze / unfreeze backbone layers per chosen strategy
    apply_finetune_strategy(model, CONFIG['finetune_mode'])

    # Fresh classification head for the new speaker set
    train_dataset = KathbathDataset(CONFIG['train_dir'], unique_across_languages=False)
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
        collate_fn=train_collate_fn, num_workers=CONFIG['num_workers'], pin_memory=True
    )

    num_classes = train_dataset.num_speakers
    logging.info("Fine-tuning on %d Kathbath speakers.", num_classes)

    loss_fn = AAMSoftmax(
        n_class=num_classes,
        in_features=CONFIG['embedding_dim'],
        m=CONFIG['aam_margin'],
        s=CONFIG['aam_scale']
    ).to(device)

    # Only pass parameters that require gradients to the optimiser
    trainable = [p for p in model.parameters() if p.requires_grad] + list(loss_fn.parameters())
    optimizer = optim.Adam(trainable, lr=CONFIG['learning_rate'], weight_decay=2e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=CONFIG['lr_decay'])

    best_eer = float('inf')

    for epoch in range(1, CONFIG['num_epochs'] + 1):
        logging.info("--- Epoch %d/%d ---", epoch, CONFIG['num_epochs'])

        avg_loss = train_epoch(model, loss_fn, train_loader, optimizer, device)
        logging.info("Epoch %d  |  Train Loss: %.4f", epoch, avg_loss)

        eer, min_dcf = evaluate_model(model, test_dataset, device, num_pairs=CONFIG['eval_pairs'])
        logging.info("Epoch %d  |  EER: %.4f%%  |  minDCF: %.4f", epoch, eer, min_dcf)

        # Save checkpoint
        ckpt_path = os.path.join(CONFIG['checkpoint_dir'], "epoch_%d.pt" % epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss_fn_state_dict': loss_fn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'eer': eer,
            'min_dcf': min_dcf
        }, ckpt_path)
        logging.info("Saved checkpoint: %s", ckpt_path)

        # Save best model separately
        if eer < best_eer:
            best_eer = eer
            best_path = os.path.join(CONFIG['checkpoint_dir'], "best_model.pt")
            torch.save({'model_state_dict': model.state_dict(), 'eer': eer}, best_path)
            logging.info("New best EER %.4f%% - saved to %s", best_eer, best_path)

        scheduler.step()
        logging.info("LR updated to: %f", scheduler.get_last_lr()[0])

    logging.info("=== Fine-Tuning Finished. Best EER: %.4f%% ===", best_eer)


if __name__ == "__main__":
    main()