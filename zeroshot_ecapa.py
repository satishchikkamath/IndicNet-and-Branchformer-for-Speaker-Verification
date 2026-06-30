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
    'test_dir':  '/home/user2/7th/parikshith_8th/Kathbath_test/',
    'log_dir':   '/home/user2/7th/parikshith_8th/logs_ecapa_zeroshot_final',
    'checkpoint_dir': '/home/user2/7th/parikshith_8th/checkpoints_ecapa_zeroshot_final',

    # Pretrained weights (already downloaded)
    'pretrained_ckpt': '/home/user2/7th-2/pretrained_ecapa/embedding_model.ckpt',

    # ECAPA architecture (must match the pretrained checkpoint)
    'in_channels':    80,
    'model_channels': 512,
    'embedding_dim':  192,

    # Training
    'batch_size':        128,
    'num_epochs':        50,
    'learning_rate':     0.0001,  # Lower LR than scratch - we start from good weights
    'lr_decay':          0.97,
    'num_workers':       8,
    'seed':              42,
    'train_chunk_frames': 200,

    # AAMSoftmax
    'aam_margin': 0.2,
    'aam_scale':  30,

    # Evaluation
    'eval_pairs':   10000,
    'dcf_p_target': 0.01,
    'dcf_c_miss':   1,
    'dcf_c_fa':     1
}

# ==================================================================
# LOGGING + SEED
# ==================================================================

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'finetune.log')),
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
# 1. ECAPA-TDNN ARCHITECTURE
# ==================================================================

class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, bias=False, scale=4):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        self.nums  = scale if scale == 1 else scale - 1
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
        sp  = None
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
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = torch.sigmoid(self.linear2(F.relu(self.linear1(out))))
        return x * out.unsqueeze(2)


def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
    return nn.Sequential(
        Conv1dReluBn(channels, channels, kernel_size=1),
        Res2Conv1dReluBn(channels, kernel_size, stride, padding,
                         dilation, scale=scale),
        Conv1dReluBn(channels, channels, kernel_size=1),
        SE_Connect(channels)
    )


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        alpha = torch.softmax(
            self.linear2(torch.tanh(self.linear1(x))), dim=2
        )
        mean = torch.sum(alpha * x, dim=2)
        std  = torch.sqrt(
            (torch.sum(alpha * x ** 2, dim=2) - mean ** 2).clamp(min=1e-9)
        )
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    def __init__(self, in_channels, channels, embd_dim):
        super().__init__()
        self.layer1  = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2  = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3  = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4  = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)
        self.conv    = nn.Conv1d(channels * 3, 1536, kernel_size=1)
        self.pooling = AttentiveStatsPool(1536, 128)
        self.bn1     = nn.BatchNorm1d(3072)
        self.linear  = nn.Linear(3072, embd_dim)
        self.bn2     = nn.BatchNorm1d(embd_dim)

    def forward(self, x):
        x    = x.transpose(1, 2)
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3
        out  = F.relu(self.conv(torch.cat([out2, out3, out4], dim=1)))
        out  = self.bn2(self.linear(self.bn1(self.pooling(out))))
        return out

# ==================================================================
# 2. PRETRAINED WEIGHT LOADER
#    Loads SpeechBrain embedding_model.ckpt into ECAPA_TDNN.
#    ALL weights are loaded and then ALL are trainable (full fine-tune).
# ==================================================================

def load_pretrained_ecapa(cfg, device):
    model = ECAPA_TDNN(
        in_channels=cfg['in_channels'],
        channels=cfg['model_channels'],
        embd_dim=cfg['embedding_dim']
    )

    ckpt_path = cfg['pretrained_ckpt']
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "Pretrained checkpoint not found at %s.\n"
            "Download it with:\n"
            "  python3 -c \""
            "from huggingface_hub import hf_hub_download; "
            "hf_hub_download(repo_id='speechbrain/spkrec-ecapa-voxceleb', "
            "filename='embedding_model.ckpt', "
            "local_dir='/home/user2/7th-2/pretrained_ecapa/')\"" % ckpt_path
        )

    logging.info("Loading pretrained weights from %s ...", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # SpeechBrain saves the state dict directly or under '0' key
    if isinstance(ckpt, dict) and '0' in ckpt:
        state_dict = ckpt['0']
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # Strip common prefixes
    cleaned = {}
    for k, v in state_dict.items():
        for prefix in ('module.', 'encoder.', 'model.', '0.'):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    logging.info(
        "Pretrained weights loaded. Missing: %d keys, Unexpected: %d keys.",
        len(missing), len(unexpected)
    )
    if missing:
        logging.warning("Missing keys (will be randomly init): %s", missing)

    # All parameters remain trainable - full fine-tuning
    model = model.to(device)
    logging.info("Model ready for fine-tuning on Kathbath.")
    return model

# ==================================================================
# 3. LOSS FUNCTION (AAMSoftmax) - same as scratch
# ==================================================================

class AAMSoftmax(nn.Module):
    def __init__(self, n_class, in_features, m, s):
        super().__init__()
        self.m  = m
        self.s  = s
        self.weight  = nn.Parameter(torch.FloatTensor(n_class, in_features))
        nn.init.xavier_normal_(self.weight, gain=1)
        self.ce_loss = nn.CrossEntropyLoss()
        self.cos_m   = np.cos(m)
        self.sin_m   = np.sin(m)
        self.th      = np.cos(np.pi - m)
        self.mm      = np.sin(np.pi - m) * m

    def forward(self, x, label):
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
        w_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-8)
        cosine = F.linear(x_norm, w_norm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        sine  = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi   = cosine * self.cos_m - sine * self.sin_m
        phi   = torch.where(cosine > self.th, phi, cosine - self.mm)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return self.ce_loss(output, label)

# ==================================================================
# 4. DATASET - same as scratch (Kathbath structure)
# ==================================================================

class KathbathDataset(Dataset):
    def __init__(self, root_dir, unique_across_languages=False):
        self.root_dirs = [root_dir] if isinstance(root_dir, str) else list(root_dir)
        self.unique_across_languages = unique_across_languages
        self.speaker_files   = []
        self.speaker_to_label = {}
        self.num_speakers    = 0

        logging.info("Loading Kathbath data from: %s", self.root_dirs)

        for root in self.root_dirs:
            language_dirs = sorted([
                d for d in glob.glob(os.path.join(root, '*'))
                if os.path.isdir(d)
            ])
            if not language_dirs:
                logging.warning("No language sub-folders found in %s - skipping.", root)
                continue

            for lang_dir in language_dirs:
                language = os.path.basename(lang_dir)
                speaker_dirs = sorted([
                    d for d in glob.glob(os.path.join(lang_dir, '*'))
                    if os.path.isdir(d)
                ])
                for spk_dir in speaker_dirs:
                    speaker_id  = os.path.basename(spk_dir)
                    speaker_key = speaker_id if self.unique_across_languages \
                                  else '%s/%s' % (language, speaker_id)
                    if speaker_key not in self.speaker_to_label:
                        self.speaker_to_label[speaker_key] = self.num_speakers
                        self.num_speakers += 1
                    label = self.speaker_to_label[speaker_key]
                    for npy_file in glob.glob(os.path.join(spk_dir, '*.npy')):
                        self.speaker_files.append((npy_file, label))

        if not self.speaker_files:
            logging.error("CRITICAL: No .npy files found. Check path and structure.")
        else:
            logging.info("Loaded %d utterances from %d unique speakers.",
                         len(self.speaker_files), self.num_speakers)

    def __len__(self):
        return len(self.speaker_files)

    def __getitem__(self, index):
        return self.speaker_files[index]

# ==================================================================
# 5. COLLATE FUNCTIONS
# ==================================================================

def train_collate_fn(batch, num_frames=CONFIG['train_chunk_frames']):
    features, labels = [], []
    for npy_path, label in batch:
        try:
            feat = np.load(npy_path).T   # (T, F)
            if not np.isfinite(feat).all():
                fv = feat[np.isfinite(feat)]
                feat[~np.isfinite(feat)] = np.median(fv) if len(fv) > 0 else 0
            if np.std(feat) < 1e-6:
                continue
        except Exception:
            continue

        if feat.shape[0] < num_frames:
            feat = np.pad(feat, ((0, num_frames - feat.shape[0]), (0, 0)), mode='wrap')
        elif feat.shape[0] > num_frames:
            start = random.randint(0, feat.shape[0] - num_frames)
            feat  = feat[start: start + num_frames]

        features.append(feat)
        labels.append(label)

    if not features:
        return None, None
    return torch.FloatTensor(np.array(features)), torch.LongTensor(labels)


def eval_collate_fn(batch):
    loaded, labels, filepaths, max_len = [], [], [], 0
    for npy_path, label in batch:
        try:
            feat = np.load(npy_path).T
            if not np.isfinite(feat).all():
                fv = feat[np.isfinite(feat)]
                feat[~np.isfinite(feat)] = np.median(fv) if len(fv) > 0 else 0
            loaded.append(feat)
            labels.append(label)
            filepaths.append(npy_path)
            if feat.shape[0] > max_len:
                max_len = feat.shape[0]
        except Exception as e:
            logging.warning("Could not load %s: %s", npy_path, e)

    if not loaded:
        return None, None, None

    features = []
    for feat in loaded:
        pad = max_len - feat.shape[0]
        features.append(
            np.pad(feat, ((0, pad), (0, 0)), mode='wrap') if pad > 0 else feat
        )
    return (
        torch.FloatTensor(np.array(features)),
        torch.LongTensor(labels),
        filepaths
    )

# ==================================================================
# 6. METRICS
# ==================================================================

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer   = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer * 100, thresh

def calculate_min_dcf(y_true, y_scores, p_target=0.01, c_miss=1, c_fa=1):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    return np.min(p_target * c_miss * fnr + (1 - p_target) * c_fa * fpr)

# ==================================================================
# 7. TRAIN + EVAL FUNCTIONS
# ==================================================================

def train_epoch(model, loss_fn, loader, optimizer, device):
    model.train()
    loss_fn.train()
    total_loss, total_batches = 0, 0

    for features, labels in loader:
        if features is None:
            continue
        features = features.to(device)
        labels   = labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(features), labels)
        loss.backward()
        optimizer.step()
        total_loss   += loss.item()
        total_batches += 1
        if total_batches % 50 == 0:
            logging.info("  Batch %d/%d, Loss: %.4f",
                         total_batches, len(loader), loss.item())

    return total_loss / max(total_batches, 1)


def evaluate_model(model, test_dataset, device, num_pairs=10000):
    logging.info("Evaluating on %d utterances...", len(test_dataset))
    model.eval()
    eval_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=eval_collate_fn,
        num_workers=CONFIG['num_workers']
    )

    embeddings, speaker_labels = {}, {}
    with torch.no_grad():
        for batch_idx, (features, labels, filepaths) in enumerate(eval_loader):
            if features is None:
                continue
            batch_emb = F.normalize(model(features.to(device)), p=2, dim=1)
            for i, fpath in enumerate(filepaths):
                embeddings[fpath]     = batch_emb[i].cpu()
                speaker_labels[fpath] = labels[i].item()
            if (batch_idx + 1) % 20 == 0:
                logging.info("  %d/%d batches, %d embeddings",
                             batch_idx + 1, len(eval_loader), len(embeddings))

    logging.info("Extracted %d embeddings.", len(embeddings))
    if len(embeddings) < 2:
        return 0.0, 0.0

    speaker_to_files = {}
    for fpath, spk in speaker_labels.items():
        speaker_to_files.setdefault(spk, []).append(fpath)

    all_files = list(embeddings.keys())
    scores, y_true = [], []

    for _ in range(num_pairs):
        is_target = random.choice([True, False])
        if is_target:
            spk   = random.choice(list(speaker_to_files.keys()))
            files = speaker_to_files[spk]
            if len(files) < 2:
                continue
            f1, f2 = random.sample(files, 2)
        else:
            f1   = random.choice(all_files)
            spk1 = speaker_labels[f1]
            imp  = [s for s in speaker_to_files if s != spk1]
            if not imp:
                continue
            f2 = random.choice(speaker_to_files[random.choice(imp)])

        score = F.cosine_similarity(
            embeddings[f1].unsqueeze(0), embeddings[f2].unsqueeze(0)
        ).item()
        scores.append(score)
        y_true.append(1 if is_target else 0)

    if not scores:
        return 0.0, 0.0

    y_true_np   = np.array(y_true)
    y_scores_np = np.array(scores)
    eer, _  = calculate_eer(y_true_np, y_scores_np)
    min_dcf = calculate_min_dcf(
        y_true_np, y_scores_np,
        p_target=CONFIG['dcf_p_target'],
        c_miss=CONFIG['dcf_c_miss'],
        c_fa=CONFIG['dcf_c_fa']
    )
    return eer, min_dcf

# ==================================================================
# 8. MAIN
# ==================================================================

def main():
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    setup_logging(CONFIG['log_dir'])
    set_seed(CONFIG['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using device: %s", device)
    logging.info("Config: %s", CONFIG)

    # --- Datasets ---
    logging.info("Loading training data...")
    train_dataset = KathbathDataset(CONFIG['train_dir'], unique_across_languages=False)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    logging.info("Loading test data...")
    test_dataset = KathbathDataset(CONFIG['test_dir'], unique_across_languages=False)

    num_classes = train_dataset.num_speakers
    logging.info("Training on %d speakers.", num_classes)

    # --- Model: pretrained ECAPA backbone ---
    model = load_pretrained_ecapa(CONFIG, device)

    # --- Loss: fresh AAMSoftmax head for Kathbath speakers ---
    loss_fn = AAMSoftmax(
        n_class=num_classes,
        in_features=CONFIG['embedding_dim'],
        m=CONFIG['aam_margin'],
        s=CONFIG['aam_scale']
    ).to(device)

    # --- Optimizer: lower LR since backbone is already pretrained ---
    all_params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer  = optim.Adam(all_params, lr=CONFIG['learning_rate'], weight_decay=2e-5)
    scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=CONFIG['lr_decay'])

    # --- Training Loop: 50 epochs ---
    logging.info("=== Starting Fine-tuning (pretrained -> Kathbath) ===")
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        logging.info("--- Epoch %d/%d ---", epoch, CONFIG['num_epochs'])

        avg_loss = train_epoch(model, loss_fn, train_loader, optimizer, device)
        logging.info("Epoch %d - Avg Train Loss: %.4f", epoch, avg_loss)

        eer, min_dcf = evaluate_model(
            model, test_dataset, device, num_pairs=CONFIG['eval_pairs']
        )
        logging.info("Epoch %d - EER: %.4f%%, minDCF: %.4f", epoch, eer, min_dcf)

        ckpt_path = os.path.join(
            CONFIG['checkpoint_dir'], 'epoch_%d.pt' % epoch
        )
        torch.save({
            'epoch':              epoch,
            'model_state_dict':   model.state_dict(),
            'loss_fn_state_dict': loss_fn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'eer':     eer,
            'min_dcf': min_dcf,
            'avg_loss': avg_loss
        }, ckpt_path)
        logging.info("Checkpoint saved: %s", ckpt_path)

        scheduler.step()
        logging.info("LR updated to: %f", scheduler.get_last_lr()[0])

    logging.info("=== Fine-tuning Complete ===")


if __name__ == '__main__':
    main()