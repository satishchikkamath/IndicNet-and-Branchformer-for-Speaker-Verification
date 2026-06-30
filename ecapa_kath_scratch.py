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
    'log_dir': '/home/user2/7th/parikshith_8th/logs_ecapakath_scratch',
    'checkpoint_dir': '/home/user2/7th/parikshith_8th/checkpoints_ecapa_scratch',
    'batch_size': 512,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'lr_decay': 0.97,       # Scheduler decay rate
    'num_workers': 8,
    'seed': 42,

    # --- Model & Data Params ---
    'in_channels': 80,      # Assuming 80-dim fbank/MFCC features in .npy
    'embedding_dim': 192,
    'model_channels': 512,
    'train_chunk_frames': 200, # 2-second chunks for training (assuming 10ms hop)

    # --- AAMSoftmax Params ---
    'aam_margin': 0.2,
    'aam_scale': 30,

    # --- Evaluation Params ---
    'eval_pairs': 10000,      # Number of pairs to sample for EER/minDCF
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

# --- Set Random Seed for Reproducibility ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================================================================
# 1. MODEL DEFINITION
# ==================================================================

class Res2Conv1dReluBn(nn.Module):
    '''
    in_channels == out_channels == channels
    '''
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out

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
        assert channels % s == 0, "{} % {} != 0".format(channels, s)
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out

def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
    return nn.Sequential(
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        SE_Connect(channels)
    )

class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)

class ECAPA_TDNN(nn.Module):
    def __init__(self, in_channels, channels, embd_dim):
        super().__init__()
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, 1536, kernel_size=1)
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

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn1(self.pooling(out))
        out = self.linear(out)
        out = self.bn2(out)
        return out

# ==================================================================
# 2. LOSS FUNCTION (AAMSoftmax)
# ==================================================================

class AAMSoftmax(nn.Module):
    def __init__(self, n_class, in_features, m, s):
        super().__init__()
        self.in_features = in_features
        self.n_class = n_class
        self.m = m
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(n_class, in_features))
        nn.init.xavier_normal_(self.weight, gain=1)
        self.ce_loss = nn.CrossEntropyLoss()
        self.cos_m = np.cos(self.m)
        self.sin_m = np.sin(self.m)
        self.th = np.cos(np.pi - self.m)
        self.mm = np.sin(np.pi - self.m) * self.m

    def forward(self, x, label):
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
        w_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-8)
        cosine = F.linear(x_norm, w_norm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        loss = self.ce_loss(output, label)
        return loss

# ==================================================================
# 3. DATASET AND DATALOADER
# ==================================================================
#
# KEY CHANGE: KathbathDataset replaces VoxCelebDataset.
#
# Kathbath folder structure:
#   <root>/
#     <language>/        e.g. hindi, english, ...
#       <speaker_id>/    e.g. 44, 102, ...
#         *.npy          flat - all utterance .npy files live here
#
# Speaker identity is keyed as "<language>/<speaker_id>" so that
# speaker "44" in "hindi" and speaker "44" in "english" are treated
# as DISTINCT identities.  If your dataset guarantees globally-unique
# numeric IDs across languages, change the key to just `speaker_id`
# (see the comment inside the class).
# ==================================================================

class KathbathDataset(Dataset):
    """
    Dataset loader for the Kathbath dataset arranged in VoxCeleb-style
    npy files but with an extra language-folder level:

        root/
          <language>/
            <speaker_id>/
              *.npy

    Parameters
    ----------
    root_dir : str or list of str
        Root directory (or list of roots) containing language sub-folders.
    unique_across_languages : bool
        If True  -> speaker key = numeric ID only (IDs are globally unique).
        If False -> speaker key = "language/ID"   (same number in two
                   languages = two different speakers).
        Default: False (safe / conservative).
    """

    def __init__(self, root_dir, unique_across_languages=False):
        if isinstance(root_dir, str):
            self.root_dirs = [root_dir]
        else:
            self.root_dirs = list(root_dir)

        self.unique_across_languages = unique_across_languages
        self.speaker_files = []      # [(npy_path, int_label), ...]
        self.speaker_to_label = {}   # {speaker_key: int_label}
        self.num_speakers = 0

        logging.info("Loading Kathbath data from: %s", self.root_dirs)

        for root in self.root_dirs:
            # Level 1: language folders  (hindi, english, ...)
            language_dirs = sorted([
                d for d in glob.glob(os.path.join(root, "*"))
                if os.path.isdir(d)
            ])

            if not language_dirs:
                logging.warning("No language sub-folders found in %s - skipping.", root)
                continue

            for lang_dir in language_dirs:
                language = os.path.basename(lang_dir)

                # Level 2: speaker ID folders  (44, 102, ...)
                speaker_dirs = sorted([
                    d for d in glob.glob(os.path.join(lang_dir, "*"))
                    if os.path.isdir(d)
                ])

                if not speaker_dirs:
                    logging.warning("No speaker folders found in %s - skipping.", lang_dir)
                    continue

                for spk_dir in speaker_dirs:
                    speaker_id = os.path.basename(spk_dir)

                    # Build the key used for the global speaker map
                    if self.unique_across_languages:
                        # Numeric IDs are unique across all languages
                        speaker_key = speaker_id
                    else:
                        # Treat same numeric ID in different languages as different
                        speaker_key = "%s/%s" % (language, speaker_id)

                    if speaker_key not in self.speaker_to_label:
                        self.speaker_to_label[speaker_key] = self.num_speakers
                        self.num_speakers += 1

                    label = self.speaker_to_label[speaker_key]

                    # Level 3: all .npy utterance files directly inside speaker folder
                    npy_files = glob.glob(os.path.join(spk_dir, "*.npy"))

                    if not npy_files:
                        logging.warning("No .npy files found for speaker %s - skipping.", spk_dir)
                        continue

                    for npy_file in npy_files:
                        self.speaker_files.append((npy_file, label))

        if not self.speaker_files:
            logging.error(
                "CRITICAL: No .npy files were found under any language/speaker folder. "
                "Check the root path and directory structure."
            )
        else:
            logging.info(
                "Loaded %d utterances from %d unique speakers.",
                len(self.speaker_files), self.num_speakers
            )

    def __len__(self):
        return len(self.speaker_files)

    def __getitem__(self, index):
        npy_path, label = self.speaker_files[index]
        return npy_path, label


def train_collate_fn(batch, num_frames=CONFIG['train_chunk_frames']):
    features = []
    labels = []

    for npy_path, label in batch:
        try:
            feat = np.load(npy_path)  # (features, time)
            feat = feat.T             # -> (time, features)

            if not np.isfinite(feat).all():
                finite_vals = feat[np.isfinite(feat)]
                if len(finite_vals) > 0:
                    median_val = np.median(finite_vals)
                    feat[~np.isfinite(feat)] = median_val
                else:
                    feat = np.zeros_like(feat)

            if np.std(feat) < 1e-6:
                continue

        except Exception:
            continue

        if feat.shape[0] < num_frames:
            pad_len = num_frames - feat.shape[0]
            feat = np.pad(feat, ((0, pad_len), (0, 0)), mode='wrap')
        elif feat.shape[0] > num_frames:
            start = random.randint(0, feat.shape[0] - num_frames)
            feat = feat[start: start + num_frames, :]

        features.append(feat)
        labels.append(label)

    if not features:
        return None, None

    features_tensor = torch.FloatTensor(np.array(features))
    labels_tensor = torch.LongTensor(labels)
    return features_tensor, labels_tensor


def eval_collate_fn(batch):
    features = []
    labels = []
    filepaths = []
    max_len = 0

    loaded_features = []
    for npy_path, label in batch:
        try:
            feat = np.load(npy_path)
            feat = feat.T

            if not np.isfinite(feat).all():
                finite_vals = feat[np.isfinite(feat)]
                if len(finite_vals) > 0:
                    median_val = np.median(finite_vals)
                    feat[~np.isfinite(feat)] = median_val
                else:
                    feat = np.zeros_like(feat)

            loaded_features.append(feat)
            labels.append(label)
            filepaths.append(npy_path)
            if feat.shape[0] > max_len:
                max_len = feat.shape[0]

        except Exception as e:
            logging.warning("Could not load or process %s (eval): %s", npy_path, e)
            continue

    if not loaded_features:
        return None, None, None

    for feat in loaded_features:
        pad_len = max_len - feat.shape[0]
        if pad_len > 0:
            padded_feat = np.pad(feat, ((0, pad_len), (0, 0)), mode='wrap')
        else:
            padded_feat = feat
        features.append(padded_feat)

    features_tensor = torch.FloatTensor(np.array(features))
    labels_tensor = torch.LongTensor(labels)
    return features_tensor, labels_tensor, filepaths

# ==================================================================
# 4. METRICS CALCULATION (EER and minDCF)
# ==================================================================

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer * 100, thresh

def calculate_min_dcf(y_true, y_scores, p_target=0.01, c_miss=1, c_fa=1):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    dcf_costs = p_target * c_miss * fnr + (1 - p_target) * c_fa * fpr
    min_dcf = np.min(dcf_costs)
    return min_dcf

# ==================================================================
# 5. TRAINING AND EVALUATION FUNCTIONS
# ==================================================================

def train_epoch(model, loss_fn, data_loader, optimizer, device):
    model.train()
    loss_fn.train()
    total_loss = 0
    total_batches = 0

    for features, labels in data_loader:
        if features is None:
            continue

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings = model(features)
        loss = loss_fn(embeddings, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if total_batches % 50 == 0:
            logging.info("  Batch %d/%d, Loss: %.4f", total_batches, len(data_loader), loss.item())

    return total_loss / total_batches


def evaluate_model(model, test_dataset, device, num_pairs=10000):
    logging.info("Starting evaluation... extracting embeddings from %d test files.", len(test_dataset))
    model.eval()

    eval_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=eval_collate_fn,
        num_workers=CONFIG['num_workers']
    )

    embeddings = {}
    speaker_labels = {}

    with torch.no_grad():
        for features, labels, filepaths in eval_loader:
            if features is None:
                continue

            features = features.to(device)
            batch_embeddings = model(features)

            for i, fpath in enumerate(filepaths):
                embeddings[fpath] = batch_embeddings[i].cpu()
                speaker_labels[fpath] = labels[i].item()

    logging.info("Extracted %d embeddings.", len(embeddings))
    if len(embeddings) < 2:
        logging.warning("Not enough embeddings extracted to perform evaluation. Skipping.")
        return 0, 0

    logging.info("Generating %d random trial pairs...", num_pairs)
    all_files = list(embeddings.keys())
    scores = []
    y_true = []

    for _ in range(num_pairs):
        is_target = random.choice([True, False])
        f1, f2 = None, None

        while f1 == f2:
            if is_target:
                f1 = random.choice(all_files)
                label1 = speaker_labels[f1]
                target_files = [f for f, l in speaker_labels.items() if l == label1]
                if len(target_files) < 2:
                    continue
                f2 = random.choice(target_files)
            else:
                f1 = random.choice(all_files)
                label1 = speaker_labels[f1]
                impostor_files = [f for f, l in speaker_labels.items() if l != label1]
                if not impostor_files:
                    continue
                f2 = random.choice(impostor_files)

        if f1 is None or f2 is None:
            continue

        emb1 = embeddings[f1]
        emb2 = embeddings[f2]
        score = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        scores.append(score)
        y_true.append(1 if is_target else 0)

    if not scores:
        logging.error("No valid pairs were generated. Skipping metrics calculation.")
        return 0, 0

    logging.info("Calculating metrics...")
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
# 6. MAIN ORCHESTRATION
# ==================================================================

def main():
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    setup_logging(CONFIG['log_dir'])
    set_seed(CONFIG['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using device: %s", device)
    logging.info("Starting training with config: %s", CONFIG)

    # --- Datasets ---
    logging.info("Loading training data (Kathbath)...")
    train_dataset = KathbathDataset(
        CONFIG['train_dir'],
        unique_across_languages=False   # Set True if speaker IDs are globally unique
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    logging.info("Loading test data...")
    # For evaluation, reuse KathbathDataset if test set has the same structure,
    # or swap back to VoxCelebDataset if you evaluate on VoxCeleb1-O.
    test_dataset = KathbathDataset(CONFIG['test_dir'], unique_across_languages=False)

    num_classes = train_dataset.num_speakers
    logging.info("Found %d speakers for training.", num_classes)

    # --- Model, Loss, Optimizer ---
    logging.info("Initializing model...")
    model = ECAPA_TDNN(
        in_channels=CONFIG['in_channels'],
        channels=CONFIG['model_channels'],
        embd_dim=CONFIG['embedding_dim']
    ).to(device)

    loss_fn = AAMSoftmax(
        n_class=num_classes,
        in_features=CONFIG['embedding_dim'],
        m=CONFIG['aam_margin'],
        s=CONFIG['aam_scale']
    ).to(device)

    all_params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = optim.Adam(all_params, lr=CONFIG['learning_rate'], weight_decay=2e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=CONFIG['lr_decay'])

    # --- Training Loop ---
    logging.info("=== Starting Training ===")
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        logging.info("--- Epoch %d/%d ---", epoch, CONFIG['num_epochs'])

        avg_loss = train_epoch(model, loss_fn, train_loader, optimizer, device)
        logging.info("Epoch %d Complete. Average Train Loss: %.4f", epoch, avg_loss)

        eer, min_dcf = evaluate_model(model, test_dataset, device, num_pairs=CONFIG['eval_pairs'])
        logging.info("Epoch %d Evaluation. EER: %.4f%%, minDCF: %.4f", epoch, eer, min_dcf)

        checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], "epoch_%d.pt" % epoch)
        torch.save({
            'model_state_dict': model.state_dict(),
            'loss_fn_state_dict': loss_fn.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'eer': eer,
            'min_dcf': min_dcf
        }, checkpoint_path)
        logging.info("Saved checkpoint to %s", checkpoint_path)

        scheduler.step()
        logging.info("Updated learning rate to: %f", scheduler.get_last_lr()[0])

    logging.info("=== Training Finished ===")


if __name__ == "__main__":
    main()