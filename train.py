# ================== IMPORTS & CONFIG ==================
import os, json, random, warnings, re, copy
import heapq
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GroupShuffleSplit
from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, precision_recall_curve, auc, roc_curve,
    brier_score_loss, matthews_corrcoef, cohen_kappa_score,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from statsmodels.stats.contingency_tables import mcnemar

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.backends.cudnn as cudnn
torch.set_float32_matmul_precision('high')
cudnn.allow_tf32       = True
torch.backends.cuda.matmul.allow_tf32 = True


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.deterministic = True
    cudnn.benchmark     = False


set_all_seeds(42)

CFG = {
    "base_dir"       : r"C:\Users\Sergio Pintado\OneDrive\Documentos\ENTRENAMIENTO - copia\DATA",
    "classes"        : ('SANO', 'GINGIVITIS', 'PERIODONTITIS'),
    "domains"        : ('MIO', 'MENDELEY'),

    "img_size"       : 288,
    "batch_size"     : 8,

    "epochs_phase1"  : 25,
    "epochs_phase2"  : 50,
    "lr_phase1"      : 3e-4,
    "lr_phase2"      : 1e-5,

    "mixup_alpha"    : 0.4,
    "label_smoothing": 0.1,
    "cutmix_prob"    : 0.5,
    "seed"           : 42,
    "num_workers"    : 2,
    "effective_batch": 16,

    "use_tta"        : True,
    "tta_n"          : 5,
    "save_cams_errors": True,
    "cams_per_class" : 12,
    "show_plots"     : False,

    # ── AI-journal extras ──────────────────────────────
    "n_boot"         : 2000,
    "kfold_k"        : 5,
    "run_kfold"      : True,        # ← ACTIVADO (Revisor 2)
    "run_ablation"   : True,
    "run_tsne"       : True,
    "run_lodo"       : True,        # ← NUEVO: Leave-One-Domain-Out (Revisor 2)
    "run_baseline"   : True,        # ← NUEVO: Comparación ResNet-50 (Revisor 2 & 4)
}

# =================== TRANSFORMS =======================
INV_NORMALIZE = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std =[1/0.229,       1/0.224,       1/0.225],
)


def crop_band_central(image, **kwargs):
    h      = image.shape[0]
    top    = int(h * 0.20)
    bottom = int(h * 0.80)
    return image[top:bottom, :, :]


def get_transforms(img_size):
    train_tf = A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.90, 1.0),
                            ratio=(0.9, 1.1), p=1.0),
        A.Lambda(image=crop_band_central, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.25),
        A.ColorJitter(brightness=0.35, contrast=0.35,
                      saturation=0.35, hue=0.08, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.25,
                                   contrast_limit=0.25, p=0.5),
        A.RandomGamma(gamma_limit=(70, 130), p=0.4),
        A.ImageCompression(quality_lower=35, quality_upper=80, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    test_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Lambda(image=crop_band_central, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_tf, test_tf


def get_tta_transforms(img_size):
    return [
        A.Compose([A.HorizontalFlip(p=1.0), A.Resize(img_size, img_size),
                   A.Lambda(image=crop_band_central, p=1.0),
                   A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                   ToTensorV2()]),
        A.Compose([A.RandomGamma(gamma_limit=(90,110), p=1.0), A.Resize(img_size, img_size),
                   A.Lambda(image=crop_band_central, p=1.0),
                   A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                   ToTensorV2()]),
        A.Compose([A.RandomBrightnessContrast(0.1, 0.1, p=1.0), A.Resize(img_size, img_size),
                   A.Lambda(image=crop_band_central, p=1.0),
                   A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                   ToTensorV2()]),
    ]


# =================== PATIENT-LEVEL GROUPING =====================
# ── FIX Revisor 2: garantizar independencia a nivel de paciente ──
def extract_patient_id(filepath, domain):
    """
    Extrae un identificador de paciente del nombre del archivo.
    - MIO numerico: "001 - SI.JPG" → "MIO_num_001"
    - MIO con nombre: "Aguilar Bravo Miguel Ángel-5.jpg" → "MIO_name_aguilar bravo miguel ángel"
    - MIO con codigo: "LC1031-F-5.jpg" → "MIO_code_LC1031"
    - MENDELEY: "00001.jpg" → "MEND_00001" (1 img = 1 paciente)
    """
    fname = os.path.basename(filepath)
    name_no_ext = os.path.splitext(fname)[0]

    if domain == "MENDELEY":
        return f"MEND_{name_no_ext}"

    # MIO numeric pattern: "001 - SI" or "002 - G"
    m = re.match(r'^(\d{2,4})\s*-?\s*(SI|G|P)', name_no_ext, re.IGNORECASE)
    if m:
        return f"MIO_num_{m.group(1).strip()}"

    # MIO code pattern: "LC1031-F-5" or "LD48-F-5" or "JD527-f-5"
    m = re.match(r'^([A-Z]{1,3}\d{1,5})', name_no_ext, re.IGNORECASE)
    if m:
        return f"MIO_code_{m.group(1).upper()}"

    # MIO "Foto-Name-..." or "Fotos Name..."
    cleaned = re.sub(r'[-_]?\d{1,5}$', '', name_no_ext)  # remove trailing number
    cleaned = re.sub(r'^(Fotos?\s*[-_]?\s*)', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace('-', ' ').replace('_', ' ').strip().lower()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    if cleaned:
        return f"MIO_name_{cleaned}"

    return f"MIO_unk_{name_no_ext}"


def check_cross_class_patients(paths, labels, domains, classes):
    """Detecta pacientes con imágenes en múltiples clases (problema de calidad)."""
    patient_classes = defaultdict(set)
    for p, l, d in zip(paths, labels, domains):
        pid = extract_patient_id(p, d)
        patient_classes[pid].add(classes[l])

    conflicts = {pid: cls for pid, cls in patient_classes.items() if len(cls) > 1}
    if conflicts:
        print(f"\n⚠️  ALERTA: {len(conflicts)} pacientes aparecen en múltiples clases:")
        for pid, cls in conflicts.items():
            print(f"   {pid} → {cls}")
        print("   Estas imágenes se mantendrán pero se agruparán en el mismo split.\n")
    return conflicts


def patient_stratified_split(paths, labels, domains, classes, test_size=0.20, val_size=0.10, seed=42):
    """
    Split estratificado a nivel de paciente.
    Garantiza que todas las imágenes de un mismo paciente van al mismo split.
    """
    # Asignar patient IDs
    patient_ids = np.array([extract_patient_id(p, d) for p, d in zip(paths, domains)])

    # Crear mapping paciente → clase mayoritaria (para estratificación)
    unique_patients = np.unique(patient_ids)
    patient_majority_class = {}
    patient_domain = {}
    for pid in unique_patients:
        mask = patient_ids == pid
        majority = Counter(labels[mask]).most_common(1)[0][0]
        patient_majority_class[pid] = majority
        patient_domain[pid] = domains[mask][0]

    # Crear arrays a nivel de paciente
    pat_arr = np.array(list(patient_majority_class.keys()))
    pat_labels = np.array([patient_majority_class[p] for p in pat_arr])
    pat_domains = np.array([patient_domain[p] for p in pat_arr])
    pat_strat_key = np.array([f"{l}_{d}" for l, d in zip(pat_labels, pat_domains)])

    # Split 1: separar test
    gss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_pidx, test_pidx = next(gss1.split(pat_arr, pat_strat_key))

    # Split 2: separar val del train
    adjusted_val = val_size / (1 - test_size)
    gss2 = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val, random_state=seed)
    trainval_strat = pat_strat_key[trainval_pidx]
    train_pidx_rel, val_pidx_rel = next(gss2.split(pat_arr[trainval_pidx], trainval_strat))
    train_pidx = trainval_pidx[train_pidx_rel]
    val_pidx = trainval_pidx[val_pidx_rel]

    # Expandir de pacientes a imágenes
    train_pats = set(pat_arr[train_pidx])
    val_pats   = set(pat_arr[val_pidx])
    test_pats  = set(pat_arr[test_pidx])

    train_mask = np.array([pid in train_pats for pid in patient_ids])
    val_mask   = np.array([pid in val_pats   for pid in patient_ids])
    test_mask  = np.array([pid in test_pats  for pid in patient_ids])

    # Verificar que no hay leakage
    assert len(train_pats & val_pats) == 0, "Leakage train-val!"
    assert len(train_pats & test_pats) == 0, "Leakage train-test!"
    assert len(val_pats & test_pats) == 0, "Leakage val-test!"

    n = len(paths)
    print(f"\n✅ Patient-level split:")
    print(f"   Patients: Train={len(train_pats)}, Val={len(val_pats)}, Test={len(test_pats)}")
    print(f"   Images:   Train={train_mask.sum()} ({100*train_mask.sum()/n:.1f}%), "
          f"Val={val_mask.sum()} ({100*val_mask.sum()/n:.1f}%), "
          f"Test={test_mask.sum()} ({100*test_mask.sum()/n:.1f}%)")

    return (
        paths[train_mask], labels[train_mask], domains[train_mask],
        paths[val_mask],   labels[val_mask],   domains[val_mask],
        paths[test_mask],  labels[test_mask],  domains[test_mask],
        patient_ids
    )


# =================== DATA LOADING =====================
def cargar_imagenes_por_dominio(base_dir, clases, dominios):
    paths, labels, domains = [], [], []
    for dom in dominios:
        for i, c in enumerate(clases):
            ddir = os.path.join(base_dir, dom, c)
            if not os.path.isdir(ddir):
                continue
            for f in os.listdir(ddir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(ddir, f))
                    labels.append(i)
                    domains.append(dom)
    return np.array(paths), np.array(labels), np.array(domains)


class CustomDentalDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths   = np.array(paths)
        self.labels  = np.array(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]).convert('RGB'))
        img = self.transform(image=img)['image']
        return img, int(self.labels[idx])


def seed_worker(worker_id):
    s = (CFG["seed"] + worker_id) % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


g_torch = torch.Generator()
g_torch.manual_seed(CFG["seed"])


def make_loaders(P_train, y_train, d_train, P_val, y_val, train_tf, test_tf):
    """Builds train/val DataLoaders with balanced sampler."""
    train_ds = CustomDentalDataset(P_train, y_train, transform=train_tf)
    val_ds   = CustomDentalDataset(P_val,   y_val,   transform=test_tf)

    keys_cd  = np.array([f"{c}_{d}" for c, d in zip(y_train, d_train)])
    uniq, counts = np.unique(keys_cd, return_counts=True)
    inv      = {k: 1.0 / c for k, c in zip(uniq, counts)}
    sample_w = np.array([inv[k] for k in keys_cd], dtype=np.float64)

    dl_kw = dict(num_workers=CFG["num_workers"], pin_memory=True,
                 worker_init_fn=seed_worker, generator=g_torch)
    try:
        dl_kw.update(dict(persistent_workers=True, prefetch_factor=2))
    except Exception:
        pass

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"],
        sampler=WeightedRandomSampler(
            torch.tensor(sample_w, dtype=torch.double),
            num_samples=len(sample_w), replacement=True),
        drop_last=True,   # ← FIX: evita batch=1 que crashea BatchNorm1d
        **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"],
                            shuffle=False, **dl_kw)
    return train_loader, val_loader


# =================== MODELS =====================

# ── SafeBatchNorm1d: no crashea con batch=1 ─────────────
class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d que usa running stats cuando batch_size == 1."""
    def forward(self, x):
        if x.size(0) == 1 and self.training:
            return F.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias, False,      # training=False
                self.momentum, self.eps)
        return super().forward(x)


# ── EfficientNet-B2 (propuesto) ──────────────────────────
class MyEfficientNetB2(nn.Module):
    def __init__(self, n=3, dropout=0.4):
        super().__init__()
        w = EfficientNet_B2_Weights.IMAGENET1K_V1
        self.base_model = efficientnet_b2(weights=w)
        in_f = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            SafeBatchNorm1d(in_f),
            nn.Dropout(p=dropout),
            nn.Linear(in_f, 512),
            nn.SiLU(),
            SafeBatchNorm1d(512),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, n),
        )

    def forward(self, x):
        return self.base_model(x)

    def freeze_backbone(self):
        """FIX Revisor 2/3: Congela el backbone, solo entrena el classifier."""
        for param in self.base_model.features.parameters():
            param.requires_grad = False
        # Mantener classifier entrenándose
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True
        frozen = sum(1 for p in self.base_model.features.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  🔒 Backbone FROZEN: {trainable:,} trainable / {total:,} total params")

    def unfreeze_all(self):
        """Descongela todo el modelo para fine-tuning completo."""
        for p in self.base_model.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  🔓 All layers UNFROZEN: {trainable:,} trainable params")

    def get_cam_target_layer(self):
        return self.base_model.features[-1]


# ── ResNet-50 (baseline comparativo — Revisor 2 & 4) ────
class MyResNet50(nn.Module):
    def __init__(self, n=3, dropout=0.4):
        super().__init__()
        w = ResNet50_Weights.IMAGENET1K_V2
        self.base_model = resnet50(weights=w)
        in_f = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            SafeBatchNorm1d(in_f),
            nn.Dropout(p=dropout),
            nn.Linear(in_f, 512),
            nn.ReLU(),
            SafeBatchNorm1d(512),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, n),
        )

    def forward(self, x):
        return self.base_model(x)

    def freeze_backbone(self):
        # Freeze everything except fc
        for name, param in self.base_model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  🔒 ResNet50 backbone FROZEN: {trainable:,} / {total:,} params")

    def unfreeze_all(self):
        for p in self.base_model.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  🔓 ResNet50 all UNFROZEN: {trainable:,} trainable params")

    def get_cam_target_layer(self):
        return self.base_model.layer4[-1]


# =================== LOSS =====================
class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        n_classes = inputs.size(1)
        with torch.no_grad():
            smooth_targets = torch.full_like(inputs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_prob = F.log_softmax(inputs, dim=1)
        ce       = -(smooth_targets * log_prob).sum(dim=1)

        prob    = torch.softmax(inputs, dim=1)
        pt      = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_w = (1 - pt) ** self.gamma

        if self.alpha is not None:
            at      = self.alpha.gather(0, targets)
            focal_w = at * focal_w

        loss = focal_w * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()


def class_balanced_alpha(counts, beta=0.9999):
    eff = (1 - np.power(beta, counts)) / (1 - beta)
    w   = 1.0 / eff
    w   = w / w.mean()
    if len(w) >= 2:
        w[1] = max(w[1], 0.8)
    return w.astype(np.float32)


def build_criterion(y_train):
    cls_counts = np.bincount(y_train, minlength=len(CFG["classes"]))
    alpha_vec  = class_balanced_alpha(cls_counts.astype(float))
    return FocalLossWithLabelSmoothing(
        alpha=torch.tensor(alpha_vec, device=DEVICE),
        gamma=1.5, smoothing=CFG["label_smoothing"]
    ), alpha_vec


# =================== AUGMENTATION =====================
def apply_mixup(inputs, targets, alpha=0.3):
    if alpha <= 0 or inputs.size(0) <= 1:
        return inputs, targets, targets, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(inputs.size(0), device=inputs.device)
    return lam * inputs + (1 - lam) * inputs[idx], targets, targets[idx], lam


def apply_cutmix(inputs, targets, alpha=1.0):
    if alpha <= 0 or inputs.size(0) <= 1:
        return inputs, targets, targets, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(inputs.size(0), device=inputs.device)
    B, C, H, W = inputs.shape
    cut_w = int(W * np.sqrt(1.0 - lam))
    cut_h = int(H * np.sqrt(1.0 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    mixed = inputs.clone()
    mixed[:, :, y1:y2, x1:x2] = inputs[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed, targets, targets[idx], lam


def apply_augmentation(inputs, targets, alpha=0.3, cutmix_prob=0.5):
    if np.random.rand() < cutmix_prob:
        return apply_cutmix(inputs, targets, alpha=1.0)
    return apply_mixup(inputs, targets, alpha=alpha)


# =================== TRAINING =========================
def train_phase(model, train_loader, val_loader, criterion, optimizer,
                epochs, scheduler=None, phase=1, prefix=""):
    from torch.optim.lr_scheduler import OneCycleLR

    best_vacc   = 0.0
    scaler      = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    ACCUM_STEPS = max(1, int(CFG["effective_batch"] // CFG["batch_size"]))
    step_per_batch = isinstance(scheduler, OneCycleLR)
    top_k = []

    best_ckpt = f"{prefix}best_model_phase{phase}.pth"

    print(f"[Phase {phase}{' | ' + prefix.strip('_') if prefix else ''}] "
          f"Accumulation: {ACCUM_STEPS}x  "
          f"(effective batch = {ACCUM_STEPS * CFG['batch_size']})")

    trA, vaA, trL, vaL, vaROC = [], [], [], [], []

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_run, correct, total = 0.0, 0, 0

        for bi, (xb, yb) in enumerate(train_loader):
            xb = xb.to(DEVICE, non_blocking=True).contiguous(
                memory_format=torch.channels_last)
            yb = yb.to(DEVICE)
            xb, ya, yb2, lam = apply_augmentation(
                xb, yb, alpha=CFG["mixup_alpha"], cutmix_prob=CFG["cutmix_prob"])

            with torch.amp.autocast(device_type=DEVICE.type,
                                    enabled=(DEVICE.type != 'cpu')):
                out  = model(xb)
                loss = (lam * criterion(out, ya) +
                        (1 - lam) * criterion(out, yb2)) / ACCUM_STEPS

            scaler.scale(loss).backward()
            if ((bi + 1) % ACCUM_STEPS) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if step_per_batch and scheduler is not None:
                    scheduler.step()

            loss_run += loss.item() * ACCUM_STEPS
            preds     = out.argmax(1)
            correct  += (lam * (preds == ya).sum() +
                         (1 - lam) * (preds == yb2).sum()).item()
            total    += ya.size(0)

        train_loss = loss_run / len(train_loader)
        train_acc  = 100.0 * correct / total
        trA.append(train_acc); trL.append(train_loss)

        # ── Validation ──
        model.eval()
        vloss, vcorrect, vtotal = 0.0, 0, 0
        all_p, all_t = [], []

        with torch.no_grad():
            for xv, yv in val_loader:
                xv = xv.to(DEVICE, non_blocking=True).contiguous(
                    memory_format=torch.channels_last)
                yv  = yv.to(DEVICE)
                out = model(xv)
                vloss    += F.cross_entropy(out, yv).item()
                prob      = torch.softmax(out, dim=1)
                vcorrect += (prob.argmax(1) == yv).sum().item()
                vtotal   += yv.size(0)
                all_p.append(prob.cpu().numpy())
                all_t.append(yv.cpu().numpy())

        vloss /= len(val_loader)
        vacc   = 100.0 * vcorrect / vtotal
        yb_bin = label_binarize(np.concatenate(all_t), classes=[0, 1, 2])
        vauc   = roc_auc_score(yb_bin, np.concatenate(all_p),
                               multi_class='ovr') if vtotal > 0 else np.nan
        vaA.append(vacc); vaL.append(vloss); vaROC.append(vauc)

        if (scheduler is not None) and (not step_per_batch):
            scheduler.step()

        print(f"Phase {phase} | Ep {ep+1:3d}/{epochs} | "
              f"TrLoss {train_loss:.4f} | VaLoss {vloss:.4f} | "
              f"TrAcc {train_acc:.2f}% | VaAcc {vacc:.2f}% | AUC {vauc:.4f}")

        # ── Checkpoint management ──
        if vacc > best_vacc:
            best_vacc = vacc
            torch.save(model.state_dict(), best_ckpt)

        save_path = f"{prefix}ckpt_ph{phase}_ep{ep+1}_acc{vacc:.2f}.pth"
        torch.save(model.state_dict(), save_path)
        heapq.heappush(top_k, (vacc, ep, save_path))
        if len(top_k) > 3:
            _, _, old = heapq.heappop(top_k)
            if os.path.exists(old):
                os.remove(old)

    top3_paths = [p for _, _, p in sorted(top_k, reverse=True)]
    return best_vacc, trA, vaA, trL, vaL, vaROC, top3_paths


# =================== SWA ==================================
def run_swa_minicycle(model, train_loader, epochs=2, lr=3e-5):
    from torch.optim.swa_utils import AveragedModel
    base      = model.to(DEVICE)
    swa_model = AveragedModel(base).to(DEVICE)
    opt       = torch.optim.AdamW(base.parameters(), lr=lr)

    for _ in range(epochs):
        base.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True).contiguous(
                memory_format=torch.channels_last)
            yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type,
                                    enabled=(DEVICE.type != 'cpu')):
                loss = F.cross_entropy(base(xb), yb)
            loss.backward(); opt.step()
        swa_model.update_parameters(base)

    swa_model.train()
    with torch.no_grad():
        for xb, _ in train_loader:
            xb = xb.to(DEVICE, non_blocking=True).contiguous(
                memory_format=torch.channels_last)
            swa_model(xb)

    base.load_state_dict(swa_model.module.state_dict(), strict=True)
    torch.save(swa_model.module.state_dict(), "best_model_swa.pth")
    print("[SWA] Saved best_model_swa.pth")
    return base


# =================== TEMPERATURE SCALING ==================
@torch.no_grad()
def temperature_scale(model, loader):
    model.eval()
    logits_all, labels_all = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True).contiguous(
            memory_format=torch.channels_last)
        logits_all.append(model(xb).detach().cpu())
        labels_all.append(yb.clone())
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)

    T   = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    def _eval():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    for _ in range(50):
        opt.step(_eval)
    return float(T.detach().item())


# =================== INFERENCE ============================
@torch.no_grad()
def predict_probs(model, loader, temperature=1.0):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True).contiguous(
            memory_format=torch.channels_last)
        pr = torch.softmax(model(xb) / temperature, dim=1)
        ys.extend(yb.numpy())
        ps.append(pr.cpu().numpy())
    return np.array(ys), np.concatenate(ps, axis=0)


@torch.no_grad()
def predict_probs_tta(model, paths, labels, base_tf, tta_tfs, temperature=1.0):
    model.eval()
    probs, ys = [], []
    for i, p in enumerate(paths):
        img     = np.array(Image.open(p).convert('RGB'))
        tensors = [base_tf(image=img)['image']] + [tf(image=img)['image'] for tf in tta_tfs]
        xb      = torch.stack(tensors, dim=0).to(DEVICE)
        pr      = torch.softmax(model(xb) / temperature, dim=1).mean(0).cpu().numpy()
        probs.append(pr)
        ys.append(int(labels[i]))
    return np.array(ys), np.vstack(probs)


def evaluate_argmax_from_probs(y_true, probas):
    return y_true, np.argmax(probas, axis=1)


# =================== EVAL & PLOTS ========================
def print_metrics(y_true, y_pred, classes):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    present = np.union1d(y_true, y_pred).astype(int)
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, labels=present.tolist(),
                                target_names=[classes[i] for i in present],
                                zero_division=0))
    print(f"F1 Macro  : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred, average='macro'):.4f}")


def save_confmat(y_true, y_pred, classes, title):
    present = np.union1d(y_true, y_pred).astype(int)
    cm   = confusion_matrix(y_true, y_pred, labels=present.tolist())
    disp = ConfusionMatrixDisplay(cm, display_labels=[classes[i] for i in present])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(title); plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=200)
    if CFG["show_plots"]: plt.show(block=True)
    plt.close()


def plot_multiclass_roc(y_true, probas, classes, fname="ROC_multiclass.png"):
    y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    plt.figure(); aucs = []
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probas[:, i])
        ra = auc(fpr, tpr); aucs.append(ra)
        plt.plot(fpr, tpr, lw=2, label=f"{classes[i]} (AUC={ra:.3f})")
    macro = float(np.mean(aucs))
    plt.plot([0,1],[0,1],"k--",lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"Multiclass ROC (macro AUC={macro:.3f})")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(fname, dpi=300)
    if CFG["show_plots"]: plt.show(block=True)
    plt.close()
    print(f"ROC macro AUC: {macro:.4f}")


def plot_multiclass_pr(y_true, probas, classes, fname="PR_multiclass.png"):
    y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    plt.figure(); pr_aucs = []
    for i in range(len(classes)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], probas[:, i])
        pa = auc(rec, prec); pr_aucs.append(pa)
        plt.plot(rec, prec, lw=2, label=f"{classes[i]} (PR-AUC={pa:.3f})")
    macro = float(np.mean(pr_aucs))
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall (macro PR-AUC={macro:.3f})")
    plt.legend(loc="lower left"); plt.tight_layout()
    plt.savefig(fname, dpi=300)
    if CFG["show_plots"]: plt.show(block=True)
    plt.close()
    print(f"PR macro AUC: {macro:.4f}")


def plot_calibration_curves(y_true, probas, classes, n_bins=10,
                            outdir="calibration_curves"):
    os.makedirs(outdir, exist_ok=True)
    for i, cname in enumerate(classes):
        y_bin  = (y_true == i).astype(int)
        brier  = brier_score_loss(y_bin, probas[:, i])
        frac, mean_p = calibration_curve(y_bin, probas[:, i],
                                         n_bins=n_bins, strategy="uniform")
        plt.figure()
        plt.plot(mean_p, frac, "s-", label="Model")
        plt.plot([0,1],[0,1],"k--", label="Perfect")
        plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
        plt.title(f"Calibration - {cname}  (Brier={brier:.3f})")
        plt.legend(loc="upper left"); plt.tight_layout()
        fname = os.path.join(outdir, f"calibration_{cname}.png")
        plt.savefig(fname, dpi=300)
        if CFG["show_plots"]: plt.show(block=True)
        plt.close()
        print(f"  {cname}: Brier={brier:.4f}")


def plot_training_curves(csv_path, phase1_epochs, fname="training_curves.png"):
    df     = pd.read_csv(csv_path)
    epochs = range(1, len(df) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    specs = [
        ("Train_Loss",     "Val_Loss",     "Loss",         "Loss"),
        ("Train_Accuracy", "Val_Accuracy", "Accuracy (%)", "Accuracy"),
        ("Val_AUC",        None,           "AUC-ROC (val)","AUC"),
    ]
    for ax, (y1, y2, title, ylabel) in zip(axes, specs):
        ax.plot(epochs, df[y1], label="Train")
        if y2 and y2 in df.columns:
            ax.plot(epochs, df[y2], label="Val")
        ax.axvline(phase1_epochs, color="gray", ls="--",
                   alpha=0.6, label="Phase 1->2")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    if CFG["show_plots"]: plt.show(block=True)
    plt.close()
    print(f"Training curves saved: {fname}")


# =================== AI-JOURNAL METRICS ==================

def bootstrap_metrics(y_true, y_pred, y_proba, classes, n_boot=2000, seed=42):
    rng       = np.random.default_rng(seed)
    n         = len(y_true)
    n_classes = len(classes)
    store = {
        "f1_macro":  [], "auc_macro": [], "mcc":       [],
        **{f"sens_{c}": [] for c in classes},
        **{f"spec_{c}": [] for c in classes},
        **{f"auc_{c}":  [] for c in classes},
        **{f"f1_{c}":   [] for c in classes},
    }

    for _ in range(n_boot):
        idx  = rng.choice(n, size=n, replace=True)
        yt   = y_true[idx]; yp = y_pred[idx]; ypr = y_proba[idx]
        if len(np.unique(yt)) < n_classes:
            continue
        store["f1_macro"].append(f1_score(yt, yp, average="macro", zero_division=0))
        store["mcc"].append(matthews_corrcoef(yt, yp))
        yb_all = label_binarize(yt, classes=list(range(n_classes)))
        store["auc_macro"].append(roc_auc_score(yb_all, ypr, multi_class="ovr"))
        cm = confusion_matrix(yt, yp, labels=list(range(n_classes)))
        for i, c in enumerate(classes):
            tp = cm[i, i]; fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp; tn = cm.sum() - (tp + fn + fp)
            store[f"sens_{c}"].append(tp/(tp+fn) if (tp+fn)>0 else 0)
            store[f"spec_{c}"].append(tn/(tn+fp) if (tn+fp)>0 else 0)
            store[f"f1_{c}"].append(
                f1_score(yt, yp, labels=[i], average="macro", zero_division=0))
            try:
                store[f"auc_{c}"].append(roc_auc_score((yt==i).astype(int), ypr[:, i]))
            except Exception:
                pass

    results = {}
    for key, vals in store.items():
        arr = np.array(vals)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        results[key] = {
            "mean": float(arr.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "str":  f"{arr.mean():.3f} [{lo:.3f}-{hi:.3f}]",
        }
    return results


def compute_kappa(y_true, y_pred):
    k_lin = cohen_kappa_score(y_true, y_pred, weights=None)
    k_wgt = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    print(f"\nCohen's Kappa (linear)   : {k_lin:.4f}")
    print(f"Cohen's Kappa (quadratic): {k_wgt:.4f}")
    return k_lin, k_wgt


def compute_mcc(y_true, y_pred):
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"MCC: {mcc:.4f}")
    return mcc


def mcnemar_test(y_true, y_pred_A, y_pred_B, name_A="Proposed", name_B="Baseline"):
    cA = (y_pred_A == y_true)
    cB = (y_pred_B == y_true)
    n10 = int(np.sum( cA & ~cB))
    n01 = int(np.sum(~cA &  cB))
    table  = [[int(np.sum(cA & cB)), n10],
              [n01, int(np.sum(~cA & ~cB))]]
    result = mcnemar(table, exact=True)
    print(f"\nMcNemar Test  {name_A} vs {name_B}")
    print(f"  n10={n10}  n01={n01}")
    print(f"  statistic={result.statistic:.4f}  p={result.pvalue:.4f}")
    sig = "Significant (p<0.05)" if result.pvalue < 0.05 else "Not significant"
    print(f"  {sig}")
    return result


def tabla_publicacion(y_true, y_pred, y_proba, classes,
                      boot, kappas, mcc_val,
                      fname="table_metrics_publication.csv"):
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    rows = []
    for i, c in enumerate(classes):
        tp = cm[i, i]; fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp; tn = cm.sum() - (tp + fn + fp)
        ppv = tp/(tp+fp) if (tp+fp)>0 else 0
        npv = tn/(tn+fn) if (tn+fn)>0 else 0
        rows.append({
            "Class": c,
            "Sensitivity": boot[f"sens_{c}"]["str"],
            "Specificity": boot[f"spec_{c}"]["str"],
            "PPV (Precision)": f"{ppv:.3f}",
            "NPV": f"{npv:.3f}",
            "F1-score": boot[f"f1_{c}"]["str"],
            "ROC AUC": boot[f"auc_{c}"]["str"],
            "Cohen k (linear)": "-",
            "Cohen k (quadratic)": "-",
            "MCC": "-",
        })
    rows.append({
        "Class": "Overall (macro)",
        "Sensitivity": "-", "Specificity": "-",
        "PPV (Precision)": "-", "NPV": "-",
        "F1-score": boot["f1_macro"]["str"],
        "ROC AUC": boot["auc_macro"]["str"],
        "Cohen k (linear)": f"{kappas[0]:.3f}",
        "Cohen k (quadratic)": f"{kappas[1]:.3f}",
        "MCC": f"{mcc_val:.3f}",
    })
    df = pd.DataFrame(rows)
    df.to_csv(fname, index=False)
    print(f"\nPublication table saved: '{fname}'")
    print(df.to_string(index=False))
    return df


# ── t-SNE embedding ──
def plot_tsne_features(model, loader, classes, fname="tsne_features.png"):
    model.eval()
    hook_out, labels_all = [], []
    def _hook(module, inp, out):
        hook_out.append(out.detach().cpu().flatten(start_dim=1))
    handle = model.base_model.avgpool.register_forward_hook(_hook)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            model(xb)
            labels_all.extend(yb.numpy())
    handle.remove()
    feats = torch.cat(hook_out, dim=0).numpy()
    print("Computing t-SNE ...")
    emb = TSNE(n_components=2, perplexity=30, random_state=42,
               n_iter=1000).fit_transform(feats)
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    plt.figure(figsize=(8, 6))
    for i, (cls, col) in enumerate(zip(classes, colors)):
        mask = np.array(labels_all) == i
        plt.scatter(emb[mask, 0], emb[mask, 1], c=col,
                    label=cls, alpha=0.7, s=20, edgecolors='none')
    plt.title("t-SNE of EfficientNet-B2 embeddings")
    plt.legend(); plt.axis("off"); plt.tight_layout()
    plt.savefig(fname, dpi=300)
    if CFG["show_plots"]: plt.show(block=True)
    plt.close()
    print(f"t-SNE saved: '{fname}'")


# ── k-Fold cross-validation (con patient-level split) ──
def run_kfold_evaluation(paths, labels, domains, k=5):
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    # Patient-level groups for GroupKFold
    patient_ids = np.array([extract_patient_id(p, d) for p, d in zip(paths, domains)])
    unique_pats = np.unique(patient_ids)
    pat_labels = np.array([Counter(labels[patient_ids == p]).most_common(1)[0][0] for p in unique_pats])
    pat_domains = np.array([domains[patient_ids == p][0] for p in unique_pats])
    pat_strat = np.array([f"{l}_{d}" for l, d in zip(pat_labels, pat_domains)])

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=CFG["seed"])
    train_tf, test_tf = get_transforms(CFG["img_size"])
    fold_results = []

    for fold, (tr_pat_idx, te_pat_idx) in enumerate(skf.split(unique_pats, pat_strat)):
        print(f"\n{'='*50}  FOLD {fold+1}/{k}  {'='*50}")
        set_all_seeds(CFG["seed"] + fold)

        train_pats = set(unique_pats[tr_pat_idx])
        test_pats = set(unique_pats[te_pat_idx])
        tr_mask = np.array([pid in train_pats for pid in patient_ids])
        te_mask = np.array([pid in test_pats for pid in patient_ids])

        P_tr, P_te = paths[tr_mask], paths[te_mask]
        y_tr, y_te = labels[tr_mask], labels[te_mask]
        d_tr = domains[tr_mask]

        train_loader, val_loader = make_loaders(
            P_tr, y_tr, d_tr, P_te, y_te, train_tf, test_tf)

        model = MyEfficientNetB2(n=len(CFG["classes"])).to(DEVICE)
        model.freeze_backbone()  # ← FIX: Phase 1 with frozen backbone
        criterion, _ = build_criterion(y_tr)
        opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                    lr=CFG["lr_phase1"])
        sch = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1, eta_min=1e-6)

        train_phase(model, train_loader, val_loader,
                    criterion, opt, CFG["epochs_phase1"], sch,
                    phase=1, prefix=f"kf{fold}_")

        model.load_state_dict(
            torch.load(f"kf{fold}_best_model_phase1.pth", map_location=DEVICE))

        y_true_f, prob_f = predict_probs(model, val_loader)
        y_pred_f = np.argmax(prob_f, axis=1)
        yb_bin = label_binarize(y_true_f, classes=list(range(len(CFG["classes"]))))

        r = {
            "fold": fold + 1,
            "accuracy": float((y_true_f == y_pred_f).mean()),
            "f1_macro": float(f1_score(y_true_f, y_pred_f, average="macro")),
            "auc_macro": float(roc_auc_score(yb_bin, prob_f, multi_class="ovr")),
            "mcc": float(matthews_corrcoef(y_true_f, y_pred_f)),
        }
        fold_results.append(r)
        print(f"  Fold {fold+1}: Acc={r['accuracy']:.4f}  F1={r['f1_macro']:.4f}  "
              f"AUC={r['auc_macro']:.4f}  MCC={r['mcc']:.4f}")

        # Cleanup fold checkpoints
        for f_ckpt in [f"kf{fold}_best_model_phase1.pth"]:
            if os.path.exists(f_ckpt):
                os.remove(f_ckpt)

    df = pd.DataFrame(fold_results)
    df.to_csv("kfold_results.csv", index=False)
    print("\nk-Fold Summary (patient-level):")
    for col in ["accuracy", "f1_macro", "auc_macro", "mcc"]:
        print(f"  {col:12s}: {df[col].mean():.4f} +/- {df[col].std():.4f}")
    return df


# ── Leave-One-Domain-Out (Revisor 2) ──
def run_lodo_evaluation(paths, labels, domains):
    """
    Train on one domain, test on the other.
    Only for classes present in both domains (SANO, GINGIVITIS).
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

    train_tf, test_tf = get_transforms(CFG["img_size"])
    results = []

    for train_dom, test_dom in [("MIO", "MENDELEY"), ("MENDELEY", "MIO")]:
        print(f"\n{'='*60}")
        print(f"  LODO: Train on {train_dom} -> Test on {test_dom}")
        print(f"{'='*60}")

        tr_mask = domains == train_dom
        te_mask = domains == test_dom

        P_tr, y_tr, d_tr = paths[tr_mask], labels[tr_mask], domains[tr_mask]
        P_te, y_te, d_te = paths[te_mask], labels[te_mask], domains[te_mask]

        # Filter to classes present in BOTH splits
        te_classes = set(np.unique(y_te))
        tr_classes = set(np.unique(y_tr))
        shared = tr_classes & te_classes

        if len(shared) < 2:
            print(f"  Skip: only {len(shared)} shared classes")
            continue

        shared_list = sorted(shared)
        tr_keep = np.isin(y_tr, shared_list)
        te_keep = np.isin(y_te, shared_list)
        P_tr, y_tr, d_tr = P_tr[tr_keep], y_tr[tr_keep], d_tr[tr_keep]
        P_te, y_te = P_te[te_keep], y_te[te_keep]

        print(f"  Train: {len(P_tr)} imgs, Test: {len(P_te)} imgs")
        print(f"  Shared classes: {[CFG['classes'][c] for c in shared_list]}")

        set_all_seeds(CFG["seed"])

        # Use 10% of train as val
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=CFG["seed"])
        tr_idx, val_idx = next(sss.split(P_tr, y_tr))
        P_val, y_val = P_tr[val_idx], y_tr[val_idx]
        d_val = d_tr[val_idx]
        P_tr2, y_tr2, d_tr2 = P_tr[tr_idx], y_tr[tr_idx], d_tr[tr_idx]

        train_loader, val_loader = make_loaders(
            P_tr2, y_tr2, d_tr2, P_val, y_val, train_tf, test_tf)

        model = MyEfficientNetB2(n=len(CFG["classes"])).to(DEVICE)
        model.freeze_backbone()
        criterion, _ = build_criterion(y_tr2)

        # Phase 1
        opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=CFG["lr_phase1"])
        sch1 = CosineAnnealingWarmRestarts(opt1, T_0=10, eta_min=1e-6)
        train_phase(model, train_loader, val_loader, criterion, opt1,
                    CFG["epochs_phase1"], sch1, phase=1,
                    prefix=f"lodo_{train_dom}_")

        model.load_state_dict(torch.load(
            f"lodo_{train_dom}_best_model_phase1.pth", map_location=DEVICE))

        # Phase 2
        model.unfreeze_all()
        opt2 = AdamW(model.parameters(), lr=CFG["lr_phase2"])
        sch2 = OneCycleLR(opt2, max_lr=6e-5,
                          epochs=CFG["epochs_phase2"],
                          steps_per_epoch=len(train_loader))
        train_phase(model, train_loader, val_loader, criterion, opt2,
                    CFG["epochs_phase2"], sch2, phase=2,
                    prefix=f"lodo_{train_dom}_")

        model.load_state_dict(torch.load(
            f"lodo_{train_dom}_best_model_phase2.pth", map_location=DEVICE))

        # Eval on unseen domain
        test_ds = CustomDentalDataset(P_te, y_te, transform=test_tf)
        test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"],
                                 shuffle=False, num_workers=CFG["num_workers"],
                                 pin_memory=True)
        y_true_l, prob_l = predict_probs(model, test_loader)
        y_pred_l = np.argmax(prob_l, axis=1)

        # Only compute metrics for shared classes
        n_classes = len(CFG["classes"])
        yb_bin = label_binarize(y_true_l, classes=list(range(n_classes)))

        row = {
            "Train Domain": train_dom,
            "Test Domain": test_dom,
            "N_test": len(P_te),
            "Accuracy": float((y_true_l == y_pred_l).mean()),
            "F1 Macro": float(f1_score(y_true_l, y_pred_l, average="macro", zero_division=0)),
            "MCC": float(matthews_corrcoef(y_true_l, y_pred_l)),
        }
        try:
            row["AUC Macro"] = float(roc_auc_score(yb_bin, prob_l, multi_class="ovr"))
        except Exception:
            row["AUC Macro"] = np.nan

        results.append(row)
        print(f"\n  LODO Result: Acc={row['Accuracy']:.4f}  "
              f"F1={row['F1 Macro']:.4f}  AUC={row.get('AUC Macro', 'N/A')}")

        # Cleanup
        for ckpt in [f"lodo_{train_dom}_best_model_phase1.pth",
                     f"lodo_{train_dom}_best_model_phase2.pth"]:
            if os.path.exists(ckpt): os.remove(ckpt)

    df = pd.DataFrame(results)
    df.to_csv("lodo_cross_domain.csv", index=False)
    print("\nLODO Cross-Domain Results:")
    print(df.to_string(index=False))
    return df


# ── Ablation study (REDISEÑADO — Revisor 2 & 4) ──
def run_ablation(P_train, y_train, d_train, P_val, y_val,
                 P_test, y_test, test_tf):
    """
    Ablation progresivo que demuestra la contribución de cada componente.
    Cada fila agrega UN componente al anterior.
    Con freeze corregido, el baseline debería tener más overfitting.
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    train_tf_base, _ = get_transforms(CFG["img_size"])
    configs = [
        # (label, freeze, cutmix_p, smoothing, use_tta, dropout, two_phase)
        ("A: CE + No freeze (vanilla)",    False, 0.0, 0.0, False, 0.2, False),
        ("B: + Backbone freeze (Phase 1)", True,  0.0, 0.0, False, 0.2, False),
        ("C: + Two-phase training",        True,  0.0, 0.0, False, 0.2, True),
        ("D: + Focal Loss + CutMix/MixUp", True,  0.5, 0.0, False, 0.2, True),
        ("E: + Label Smoothing",           True,  0.5, 0.1, False, 0.2, True),
        ("F: + Dropout 0.4",               True,  0.5, 0.1, False, 0.4, True),
        ("G: + TTA (proposed)",            True,  0.5, 0.1, True,  0.4, True),
    ]
    results = []
    baseline_preds = None

    for label, do_freeze, cp, sm, use_tta, drop, two_phase in configs:
        print(f"\n{'='*60}\n  Ablation: {label}\n{'='*60}")
        set_all_seeds(CFG["seed"])

        _orig_cp = CFG["cutmix_prob"]; _orig_sm = CFG["label_smoothing"]
        CFG["cutmix_prob"] = cp; CFG["label_smoothing"] = sm

        model = MyEfficientNetB2(n=len(CFG["classes"]), dropout=drop).to(DEVICE)

        # Use CE loss for vanilla, Focal for rest
        if cp == 0.0 and sm == 0.0 and not do_freeze:
            criterion_abl = nn.CrossEntropyLoss()
        else:
            criterion_abl, _ = build_criterion(y_train)

        if do_freeze:
            model.freeze_backbone()

        train_loader, val_loader = make_loaders(
            P_train, y_train, d_train, P_val, y_val, train_tf_base, test_tf)

        # Phase 1
        opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                    lr=CFG["lr_phase1"])
        sch = CosineAnnealingWarmRestarts(opt, T_0=10, eta_min=1e-6)
        train_phase(model, train_loader, val_loader,
                    criterion_abl, opt, CFG["epochs_phase1"], sch,
                    phase=1, prefix=f"abl_{label[:1]}_")

        model.load_state_dict(
            torch.load(f"abl_{label[:1]}_best_model_phase1.pth", map_location=DEVICE))

        # Phase 2 (if enabled)
        if two_phase:
            from torch.optim.lr_scheduler import OneCycleLR
            model.unfreeze_all()
            opt2 = AdamW(model.parameters(), lr=CFG["lr_phase2"])
            sch2 = OneCycleLR(opt2, max_lr=6e-5,
                              epochs=CFG["epochs_phase2"],
                              steps_per_epoch=len(train_loader))
            train_phase(model, train_loader, val_loader,
                        criterion_abl, opt2, min(CFG["epochs_phase2"], 30), sch2,
                        phase=2, prefix=f"abl_{label[:1]}_")
            best_p2 = f"abl_{label[:1]}_best_model_phase2.pth"
            if os.path.exists(best_p2):
                model.load_state_dict(torch.load(best_p2, map_location=DEVICE))

        CFG["cutmix_prob"] = _orig_cp; CFG["label_smoothing"] = _orig_sm

        # Inference
        test_ds = CustomDentalDataset(P_test, y_test, transform=test_tf)
        test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"],
                                 shuffle=False, num_workers=CFG["num_workers"],
                                 pin_memory=True)
        if use_tta:
            tta_tfs = get_tta_transforms(CFG["img_size"])
            y_true_a, prob_a = predict_probs_tta(
                model, P_test, y_test, test_tf, tta_tfs)
        else:
            y_true_a, prob_a = predict_probs(model, test_loader)

        y_pred_a = np.argmax(prob_a, axis=1)
        yb_bin = label_binarize(y_true_a, classes=[0, 1, 2])

        row = {
            "Configuration": label,
            "Accuracy": f"{(y_true_a==y_pred_a).mean():.4f}",
            "F1 Macro": f"{f1_score(y_true_a, y_pred_a, average='macro'):.4f}",
            "AUC Macro": f"{roc_auc_score(yb_bin, prob_a, multi_class='ovr'):.4f}",
            "MCC": f"{matthews_corrcoef(y_true_a, y_pred_a):.4f}",
        }
        results.append(row)
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))

        if baseline_preds is None:
            baseline_preds = y_pred_a.copy()

        # Cleanup ablation checkpoints
        for ckpt in [f"abl_{label[:1]}_best_model_phase1.pth",
                     f"abl_{label[:1]}_best_model_phase2.pth"]:
            if os.path.exists(ckpt): os.remove(ckpt)

    df = pd.DataFrame(results)
    df.to_csv("ablation_study.csv", index=False)
    print("\nAblation Study:")
    print(df.to_string(index=False))
    return df, baseline_preds


# ── Baseline comparison (ResNet-50 — Revisor 2 & 4) ──
def run_baseline_comparison(P_train, y_train, d_train,
                            P_val, y_val,
                            P_test, y_test, test_tf):
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

    train_tf, _ = get_transforms(CFG["img_size"])
    results = []

    models_to_compare = [
        ("EfficientNet-B2 (proposed)", MyEfficientNetB2),
        ("ResNet-50 (baseline)", MyResNet50),
    ]

    for model_name, ModelClass in models_to_compare:
        print(f"\n{'='*60}")
        print(f"  Baseline Comparison: {model_name}")
        print(f"{'='*60}")
        set_all_seeds(CFG["seed"])

        model = ModelClass(n=len(CFG["classes"])).to(DEVICE)
        model.freeze_backbone()
        criterion, _ = build_criterion(y_train)

        train_loader, val_loader = make_loaders(
            P_train, y_train, d_train, P_val, y_val, train_tf, test_tf)

        # Phase 1
        opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=CFG["lr_phase1"])
        sch1 = CosineAnnealingWarmRestarts(opt1, T_0=10, eta_min=1e-6)
        pfx = model_name.split()[0].lower() + "_"
        train_phase(model, train_loader, val_loader,
                    criterion, opt1, CFG["epochs_phase1"], sch1,
                    phase=1, prefix=pfx)

        model.load_state_dict(
            torch.load(f"{pfx}best_model_phase1.pth", map_location=DEVICE))

        # Phase 2
        model.unfreeze_all()
        opt2 = AdamW(model.parameters(), lr=CFG["lr_phase2"])
        sch2 = OneCycleLR(opt2, max_lr=6e-5,
                          epochs=CFG["epochs_phase2"],
                          steps_per_epoch=len(train_loader))
        train_phase(model, train_loader, val_loader,
                    criterion, opt2, CFG["epochs_phase2"], sch2,
                    phase=2, prefix=pfx)

        best_p2 = f"{pfx}best_model_phase2.pth"
        if os.path.exists(best_p2):
            model.load_state_dict(torch.load(best_p2, map_location=DEVICE))

        # Eval
        test_ds = CustomDentalDataset(P_test, y_test, transform=test_tf)
        test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"],
                                 shuffle=False, num_workers=CFG["num_workers"],
                                 pin_memory=True)

        if CFG["use_tta"]:
            tta_tfs = get_tta_transforms(CFG["img_size"])
            y_true_b, prob_b = predict_probs_tta(
                model, P_test, y_test, test_tf, tta_tfs)
        else:
            y_true_b, prob_b = predict_probs(model, test_loader)

        y_pred_b = np.argmax(prob_b, axis=1)
        yb_bin = label_binarize(y_true_b, classes=[0, 1, 2])

        # Bootstrap CI for this model
        boot = bootstrap_metrics(y_true_b, y_pred_b, prob_b,
                                 list(CFG["classes"]), n_boot=CFG["n_boot"])

        row = {
            "Model": model_name,
            "Accuracy": f"{(y_true_b==y_pred_b).mean():.4f}",
            "F1 Macro": boot["f1_macro"]["str"],
            "AUC Macro": boot["auc_macro"]["str"],
            "MCC": boot["mcc"]["str"],
            "Kappa": f"{cohen_kappa_score(y_true_b, y_pred_b):.4f}",
        }
        results.append(row)
        print(f"  {model_name}: F1={row['F1 Macro']}  AUC={row['AUC Macro']}")

        # Cleanup
        for ckpt in [f"{pfx}best_model_phase1.pth", f"{pfx}best_model_phase2.pth"]:
            if os.path.exists(ckpt): os.remove(ckpt)

    df = pd.DataFrame(results)
    df.to_csv("baseline_comparison.csv", index=False)
    print("\nBaseline Comparison:")
    print(df.to_string(index=False))
    return df


# =================== GRAD-CAM =========================
def make_cam(method, model, target_layer):
    return {"gradcam++": GradCAMPlusPlus, "layercam": LayerCAM}.get(
        method, GradCAMPlusPlus)(model=model, target_layers=[target_layer])


def save_cam_errors(model, paths, labels, probs, classes,
                    outdir="cams_errores", method="gradcam++", max_per_class=12):
    os.makedirs(outdir, exist_ok=True)
    preds = np.argmax(probs, axis=1)
    errs_idx = np.where(preds != labels)[0]
    if len(errs_idx) == 0:
        print("No errors for CAM."); return

    sel = errs_idx[np.argsort(-probs[errs_idx, preds[errs_idx]])]
    model.eval()
    cam_obj = make_cam(method, model, model.get_cam_target_layer())
    per_cls = {c: 0 for c in range(len(classes))}
    base_tf = A.Compose([A.Resize(CFG["img_size"], CFG["img_size"]),
                         A.Lambda(image=crop_band_central, p=1.0)])
    norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    for idx in sel:
        t = int(labels[idx])
        if per_cls[t] >= max_per_class: continue
        img = np.array(Image.open(paths[idx]).convert('RGB'))
        img_r = base_tf(image=img)['image']
        x = ToTensorV2()(image=norm(image=img_r)['image'])['image'] \
                .unsqueeze(0).to(DEVICE)
        x.requires_grad_(True)
        cam = cam_obj(input_tensor=x)[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        vis = show_cam_on_image(img_r.astype(np.float32)/255.0, cam, use_rgb=True)
        p_idx = int(preds[idx])
        Image.fromarray(vis).save(
            os.path.join(outdir,
                         f"err_{idx}_real_{classes[t]}_pred_{classes[p_idx]}.png"))
        per_cls[t] += 1
        if all(per_cls[c] >= max_per_class for c in per_cls): break
    print(f"CAMs saved: '{outdir}'")


# ========================= MAIN =======================
def main():
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

    set_all_seeds(CFG["seed"])
    os.makedirs("results", exist_ok=True)
    os.chdir("results")

    # ── Data loading ──
    paths, labels, domains = cargar_imagenes_por_dominio(
        CFG["base_dir"], CFG["classes"], CFG["domains"])
    print(f"Total images loaded: {len(paths)}")

    # ── Check cross-class patients (data quality) ──
    conflicts = check_cross_class_patients(paths, labels, domains, CFG["classes"])

    # ── Patient-level stratified split (FIX Revisor 2) ──
    (P_train, y_train, d_train,
     P_val,   y_val,   d_val,
     P_test,  y_test,  d_test,
     patient_ids) = patient_stratified_split(
        paths, labels, domains, CFG["classes"],
        test_size=0.20, val_size=0.10, seed=CFG["seed"])

    n_total = len(paths)
    for y, d, name in [(y_train, d_train, "TRAIN"),
                        (y_val, d_val, "VAL"),
                        (y_test, d_test, "TEST")]:
        c = Counter(zip(y, d))
        print(f"\n[{name}] n={len(y)}")
        for (cl, dm), v in sorted(c.items()):
            print(f"  {CFG['classes'][cl]:>13} | {dm:>9} -> {v}")

    with open("run_metadata.json", "w") as f:
        json.dump({
            "classes": CFG["classes"], "domains": CFG["domains"],
            "img_size": CFG["img_size"], "batch_size": CFG["batch_size"],
            "epochs": [CFG["epochs_phase1"], CFG["epochs_phase2"]],
            "splits": {"train": len(P_train), "val": len(P_val), "test": len(P_test)},
            "patient_level_split": True,
            "cross_class_conflicts": len(conflicts),
        }, f, indent=2)

    # ── Transforms & loaders ──
    train_tf, test_tf = get_transforms(CFG["img_size"])
    train_loader, val_loader = make_loaders(
        P_train, y_train, d_train, P_val, y_val, train_tf, test_tf)
    test_ds = CustomDentalDataset(P_test, y_test, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"],
                             shuffle=False, num_workers=CFG["num_workers"],
                             pin_memory=True)

    # ── Model & loss ──
    criterion, alpha_vec = build_criterion(y_train)
    print("Focal alphas:", alpha_vec)

    model = MyEfficientNetB2(n=len(CFG["classes"])).to(DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ── Phase 1: HEAD ONLY (backbone frozen) ──
    print("\n" + "="*60 + "\n  PHASE 1: Classifier head training (backbone frozen)\n" + "="*60)
    model.freeze_backbone()  # ← FIX CRÍTICO

    opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=CFG["lr_phase1"])
    sch1 = CosineAnnealingWarmRestarts(opt1, T_0=10, T_mult=1, eta_min=1e-6)
    _, trA1, vaA1, trL1, vaL1, vaR1, top3_p1 = train_phase(
        model, train_loader, val_loader,
        criterion, opt1, CFG["epochs_phase1"], sch1, phase=1)

    model.load_state_dict(torch.load("best_model_phase1.pth", map_location=DEVICE))

    # ── Phase 2: FULL fine-tune (all layers) ──
    print("\n" + "="*60 + "\n  PHASE 2: Full fine-tuning (all layers)\n" + "="*60)
    model.unfreeze_all()  # ← Ahora sí desbloquea

    opt2 = AdamW(model.parameters(), lr=CFG["lr_phase2"])
    sch2 = OneCycleLR(opt2, max_lr=6e-5,
                      epochs=CFG["epochs_phase2"],
                      steps_per_epoch=len(train_loader))
    _, trA2, vaA2, trL2, vaL2, vaR2, top3_p2 = train_phase(
        model, train_loader, val_loader,
        criterion, opt2, CFG["epochs_phase2"], sch2, phase=2)

    model.load_state_dict(torch.load("best_model_phase2.pth", map_location=DEVICE))

    # Unified training history
    hist_path = "historia_entrenamiento_unida.csv"
    pd.DataFrame({
        "Train_Accuracy": trA1 + trA2, "Val_Accuracy": vaA1 + vaA2,
        "Train_Loss": trL1 + trL2, "Val_Loss": vaL1 + vaL2,
        "Val_AUC": vaR1 + vaR2,
    }).to_csv(hist_path, index=False)

    # ── SWA ──
    model = run_swa_minicycle(model, train_loader, epochs=2, lr=3e-5)
    torch.save(model.state_dict(), "best_model_deploy.pth")

    # ── Temperature calibration ──
    T = temperature_scale(model, val_loader)
    print(f"Temperature T = {T:.3f}")

    # ── Inference (TTA) ──
    if CFG["use_tta"]:
        tta_tfs = get_tta_transforms(CFG["img_size"])
        y_true, prob_test = predict_probs_tta(
            model, P_test, y_test, test_tf, tta_tfs, temperature=T)
    else:
        y_true, prob_test = predict_probs(model, test_loader, temperature=T)

    y_true_a, y_pred_a = evaluate_argmax_from_probs(y_true, prob_test)

    # ── Standard evaluation ──
    print("\n=== Final Test Evaluation ===")
    print_metrics(y_true_a, y_pred_a, CFG["classes"])
    save_confmat(y_true_a, y_pred_a, CFG["classes"], "MC_TestMixto_argmax")

    # Per-class stats
    df_adv, cm_full = [], confusion_matrix(y_true, y_pred_a, labels=[0, 1, 2])
    for i, cname in enumerate(CFG["classes"]):
        tp = cm_full[i, i]; fn = cm_full[i, :].sum() - tp
        fp = cm_full[:, i].sum() - tp; tn = cm_full.sum() - (tp + fn + fp)
        yb_bin = (np.array(y_true) == i).astype(int)
        try:   rocA = roc_auc_score(yb_bin, prob_test[:, i])
        except: rocA = np.nan
        pr, rc, _ = precision_recall_curve(yb_bin, prob_test[:, i])
        df_adv.append({
            "Clase": cname,
            "Sensibilidad": tp/(tp+fn) if (tp+fn)>0 else 0,
            "Especificidad": tn/(tn+fp) if (tn+fp)>0 else 0,
            "ROC AUC": rocA, "PR AUC": auc(rc, pr),
        })
    pd.DataFrame(df_adv).to_csv("estadisticas_avanzadas.csv", index=False)
    np.savez("test_probs_final.npz", y_true=y_true, probas=prob_test)

    # Plots
    en_classes = ["Healthy", "Gingivitis", "Periodontitis"]
    plot_multiclass_roc(y_true, prob_test, en_classes)
    plot_multiclass_pr(y_true, prob_test, en_classes)
    plot_calibration_curves(y_true, prob_test, en_classes)
    plot_training_curves(hist_path, phase1_epochs=CFG["epochs_phase1"])

    # Grad-CAM
    if CFG["save_cams_errors"]:
        save_cam_errors(model, P_test, y_true, prob_test, CFG["classes"],
                        outdir="cams_errores", method="gradcam++",
                        max_per_class=CFG["cams_per_class"])

    # ══════════════════════════════════════════════════════
    #   AI-JOURNAL METRICS
    # ══════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  AI-JOURNAL METRICS")
    print("="*60)

    y_true_np = np.array(y_true_a)
    y_pred_np = np.array(y_pred_a)

    mcc_val = compute_mcc(y_true_np, y_pred_np)
    kappas = compute_kappa(y_true_np, y_pred_np)

    print(f"\nBootstrap CI 95% ({CFG['n_boot']} iterations) ...")
    boot = bootstrap_metrics(y_true_np, y_pred_np, prob_test,
                             list(CFG["classes"]), n_boot=CFG["n_boot"])
    print("\nBootstrap results:")
    for k, v in boot.items():
        print(f"  {k:20s}: {v['str']}")

    tabla_publicacion(y_true_np, y_pred_np, prob_test,
                      list(CFG["classes"]), boot, kappas, mcc_val)

    # ── t-SNE ──
    if CFG["run_tsne"]:
        plot_tsne_features(model, test_loader, list(CFG["classes"]))

    # ── Ablation study (REDISEÑADO) ──
    if CFG["run_ablation"]:
        print("\n" + "="*60 + "\n  ABLATION STUDY (Reviewer 2 & 4)\n" + "="*60)
        df_abl, baseline_preds = run_ablation(
            P_train, y_train, d_train, P_val, y_val,
            P_test, y_test, test_tf)
        mcnemar_test(y_true_np, y_pred_np, baseline_preds,
                     name_A="Proposed (G)", name_B="Baseline (A)")

    # ── Baseline comparison (ResNet-50) ──
    if CFG["run_baseline"]:
        print("\n" + "="*60 + "\n  BASELINE COMPARISON (Reviewer 2 & 4)\n" + "="*60)
        run_baseline_comparison(P_train, y_train, d_train,
                                P_val, y_val,
                                P_test, y_test, test_tf)

    # ── Leave-One-Domain-Out (Revisor 2) ──
    if CFG["run_lodo"]:
        print("\n" + "="*60 + "\n  LEAVE-ONE-DOMAIN-OUT (Reviewer 2)\n" + "="*60)
        run_lodo_evaluation(paths, labels, domains)

    # ── k-Fold cross-validation (patient-level) ──
    if CFG["run_kfold"]:
        print("\n" + "="*60 + "\n  k-FOLD CROSS-VALIDATION (patient-level)\n" + "="*60)
        run_kfold_evaluation(paths, labels, domains, k=CFG["kfold_k"])

    # ── Save trainable params count for paper (Revisor 3) ──
    model_info = {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_phase1": sum(p.numel() for n, p in model.named_parameters()
                                if 'classifier' in n or 'fc' in n),
        "trainable_phase2": sum(p.numel() for p in model.parameters()),
    }
    with open("model_params.json", "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"\nModel params: Total={model_info['total_params']:,}, "
          f"Phase1={model_info['trainable_phase1']:,}, "
          f"Phase2={model_info['trainable_phase2']:,}")

    print("\nPipeline complete. All results in 'results/'.")


if __name__ == "__main__":
    main()
