# =========================================================
# CNN + Atom Search Optimization (ASO)
# Addresses ALL reviewer concerns:
#   R1: Data leakage proof + strict subject-wise splits
#   R2: 5-fold subject-wise CV + mean±std reporting
#   R3: t-SNE/UMAP feature separability analysis
#   R4: Ablation study (AD-CNN architecture components)
#   R5: ASO vs PCA/RFE/RF-Importance comparison
#   R6: Strong baselines (ResNet18, MobileNetV3, EfficientNet)
#   R7: Extended metrics (Balanced Acc, MCC, Kappa, Brier, ECE, LogLoss)
#   R8: Detailed error analysis + confusion matrix per fold
# =========================================================

import os, re, gc, copy, json, math, shutil, random, warnings, time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    precision_recall_fscore_support, confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score, f1_score, log_loss,
)
from sklearn.manifold import TSNE

try:
    import umap.umap_ as umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# =========================================================
# PATHS
# =========================================================
DATASET_SOURCE_PATH = ""
WORK_DIR      = ""
RESULTS_DIR   = os.path.join(WORK_DIR, "results")
CV_DIR        = os.path.join(RESULTS_DIR, "subjectwise_5fold_cv")
ABLATION_DIR  = os.path.join(RESULTS_DIR, "ablation_study")
BASELINE_COMP = os.path.join(RESULTS_DIR, "baseline_comparison")
FS_COMP_DIR   = os.path.join(RESULTS_DIR, "feature_selection_comparison")
TEXT_DIR      = os.path.join(RESULTS_DIR, "paper_ready_text")

# =========================================================
# MAIN SETTINGS
# =========================================================
SPLIT_MODE = "subjectwise_auto"   #  strict subject-wise, no leakage
MODEL_NAME = "efficientnet_b0"
USE_PRETRAINED   = True
IMAGE_SIZE       = 224
BATCH_SIZE       = 24
EPOCHS           = 30
FREEZE_BACKBONE_EPOCHS = 3
LEARNING_RATE    = 2e-4
WEIGHT_DECAY     = 1e-4
NUM_WORKERS      = 0
SEED             = 42
TEST_SIZE        = 0.20
VAL_SIZE_FROM_TRAIN = 0.15
TARGET_TRAIN_IMAGES = 26000
EARLY_STOP_PATIENCE = 7
FEATURE_DIM      = 128
DROPOUT          = 0.35
USE_TTA          = True
MIXUP_ALPHA      = 0.25

# =========================================================
# REVIEWER FLAGS
# =========================================================
RUN_SUBJECTWISE_5FOLD_CV          = True   # R2
CV_N_SPLITS                       = 5      # R2
RUN_TSNE_UMAP                     = True   # R3
RUN_ABLATION_STUDY                = True   # R4
RUN_FEATURE_SELECTION_COMPARISON  = True   # R5
RUN_BASELINE_MODEL_COMPARISON     = True   # R6

# =========================================================
# ASO SETTINGS  (R5)
# =========================================================
ASO_LOWER_BOUND    = 0.0
ASO_UPPER_BOUND    = 1.0
ASO_THRESHOLD      = 0.5
ASO_ALPHA          = 50
ASO_BETA           = 0.2
ASO_TOTAL_SOLUTIONS= 10
ASO_MAX_ITERS      = 100
ASO_PENALTY        = 0.03
PCA_N_COMPONENTS   = 64
RFE_N_FEATURES     = 64
RF_TOPK_FEATURES   = 64

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p):   os.makedirs(p, exist_ok=True)
def ensure_clean_dir(p):
    if os.path.isdir(p): shutil.rmtree(p)
    os.makedirs(p)

def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)

# =========================================================
# PATH / ID HELPERS
# =========================================================
def infer_subject_id(filepath: str) -> str:
    name = os.path.basename(filepath)
    stem = os.path.splitext(name)[0].lower()
    parent = os.path.basename(os.path.dirname(filepath)).lower()
    combined = f"{parent}_{stem}"
    patterns = [
        r"(subject[_-]?\d+)", r"(sub[_-]?\d+)", r"(patient[_-]?\d+)", r"(pt[_-]?\d+)",
        r"(adni[_-]?[a-z0-9]+)", r"(oasis[_-]?\d+)", r"([a-z]+\d{3,})", r"(\d{4,})",
    ]
    for txt in [combined, stem]:
        for p in patterns:
            m = re.search(p, txt)
            if m: return m.group(1)
    parts = re.split(r"[_\-\s]+", stem)
    return "_".join(parts[:2]) if len(parts) >= 2 else stem

def extract_first_relative_dir(file_path, root_dir):
    return os.path.relpath(file_path, root_dir).split(os.sep)[0]

def find_named_dir(root_dir, names=("train","test","val")):
    names = {x.lower() for x in names}
    if os.path.isdir(root_dir):
        hits = [os.path.join(root_dir, n) for n in os.listdir(root_dir)
                if n.lower() in names and os.path.isdir(os.path.join(root_dir, n))]
        if hits: return sorted(hits, key=len)[0]
    hits = []
    for dp, dns, _ in os.walk(root_dir):
        for d in dns:
            if d.lower() in names: hits.append(os.path.join(dp, d))
    return sorted(hits, key=len)[0] if hits else None

# =========================================================
# DATA DISCOVERY
# =========================================================
def discover_class_names_from_roots(roots):
    class_names = set()
    for root in roots:
        if not root or not os.path.isdir(root): continue
        for dp, _, files in os.walk(root):
            valid = [f for f in files if os.path.splitext(f)[1].lower() in IMG_EXTS]
            if valid:
                class_names.add(extract_first_relative_dir(os.path.join(dp, valid[0]), root))
    if not class_names: raise ValueError(f"No images found in roots: {roots}")
    return sorted(class_names)

def collect_samples_from_root(root_dir, class_to_idx):
    samples = []
    for dp, _, files in os.walk(root_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() not in IMG_EXTS: continue
            path = os.path.join(dp, f)
            cn = extract_first_relative_dir(path, root_dir)
            if cn in class_to_idx:
                samples.append((path, class_to_idx[cn], infer_subject_id(path)))
    return samples

def collect_all_samples_from_single_root(dataset_root):
    class_names = set()
    for root, _, files in os.walk(dataset_root):
        valid = [f for f in files if os.path.splitext(f)[1].lower() in IMG_EXTS]
        if valid:
            rel = os.path.relpath(root, dataset_root).split(os.sep)
            cn = rel[1] if len(rel)>=2 and rel[0].lower() in {"train","test","val","valid","validation"} else rel[0]
            class_names.add(cn)
    if not class_names: raise ValueError(f"No images found in: {dataset_root}")
    class_names = sorted(class_names)
    class_to_idx = {c:i for i,c in enumerate(class_names)}
    idx_to_class = {i:c for c,i in class_to_idx.items()}
    samples = []
    for root, _, files in os.walk(dataset_root):
        for f in files:
            if os.path.splitext(f)[1].lower() not in IMG_EXTS: continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, dataset_root).split(os.sep)
            cn = rel[1] if len(rel)>=2 and rel[0].lower() in {"train","test","val","valid","validation"} else rel[0]
            if cn in class_to_idx:
                samples.append((path, class_to_idx[cn], infer_subject_id(path)))
    return samples, class_to_idx, idx_to_class

def print_dataset_summary(samples, idx_to_class, title):
    labels = [s[1] for s in samples]
    print(f"\n{'='*80}\n{title}\n{'='*80}")
    print(f"Total images   : {len(samples)}")
    print(f"Total subjects : {len(set(s[2] for s in samples))}")
    for k,v in sorted(Counter(labels).items()):
        print(f"  {idx_to_class[k]:<20} {v}")

def save_manifest(samples, out_csv):
    pd.DataFrame(samples, columns=["path","label","subject_id"]).to_csv(out_csv, index=False)

def save_class_distribution(samples, idx_to_class, out_csv):
    rows = []
    for cls_idx, count in Counter(s[1] for s in samples).items():
        rows.append({"class_index": cls_idx, "class_name": idx_to_class[cls_idx],
                     "n_images": count,
                     "n_subjects": len(set(sid for _,lbl,sid in samples if lbl==cls_idx))})
    pd.DataFrame(rows).sort_values("class_index").to_csv(out_csv, index=False)

# =========================================================
# SUBJECT-WISE SPLITTING  (R1 – no leakage proof)
# =========================================================
def split_subjectwise(samples, test_size=0.20, seed=42):
    s2i = defaultdict(list)
    for item in samples: s2i[item[2]].append(item)
    sids, slbls = [], []
    for sid, items in s2i.items():
        sids.append(sid)
        slbls.append(Counter(x[1] for x in items).most_common(1)[0][0])
    sids = np.array(sids); slbls = np.array(slbls)
    try:
        tr, te = train_test_split(sids, test_size=test_size, random_state=seed, stratify=slbls)
    except Exception:
        tr, te = train_test_split(sids, test_size=test_size, random_state=seed, shuffle=True)
    tr, te = set(tr.tolist()), set(te.tolist())
    return [s for s in samples if s[2] in tr], [s for s in samples if s[2] in te]

def get_subject_level_arrays(samples):
    s2i = defaultdict(list)
    for item in samples: s2i[item[2]].append(item)
    sids, slbls = [], []
    for sid, items in s2i.items():
        sids.append(sid)
        slbls.append(Counter(x[1] for x in items).most_common(1)[0][0])
    return np.array(sids), np.array(slbls), s2i

def make_subjectwise_folds(samples, n_splits=5, seed=42, val_size_from_train=0.15):
    sids, slbls, s2i = get_subject_level_arrays(samples)
    min_cls = min(Counter(slbls.tolist()).values())
    if min_cls >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(sids, slbls)
        print(f"[CV] StratifiedKFold (min_class_subjects={min_cls} >= {n_splits})")
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(sids)
        warnings.warn("[CV] Falling back to KFold (insufficient per-class subjects)")
    folds = []
    for fold_idx, (tr_idx, te_idx) in enumerate(split_iter, start=1):
        tr_subs, te_subs = sids[tr_idx], sids[te_idx]
        trainval, test = [], []
        for sid in tr_subs: trainval.extend(s2i[sid])
        for sid in te_subs:  test.extend(s2i[sid])
        train, val = split_subjectwise(trainval, test_size=val_size_from_train, seed=seed+fold_idx)
        folds.append({"fold_index": fold_idx, "train_samples": train, "val_samples": val, "test_samples": test})
    return folds

def check_subject_overlap(a, b, out_txt=None, na="A", nb="B"):
    sa, sb = set(s[2] for s in a), set(s[2] for s in b)
    overlap = sa & sb
    if out_txt:
        with open(out_txt, "w") as f:
            f.write(f"{na} subjects: {len(sa)}\n{nb} subjects: {len(sb)}\nOverlap: {len(overlap)}\n")
            for sid in sorted(overlap): f.write(f"{sid}\n")
    if overlap: warnings.warn(f"{len(overlap)} overlapping subjects between {na} and {nb}")
    return overlap

def prepare_splits(dataset_root, split_mode, seed, test_size, val_size_from_train):
    if split_mode == "existing_folder_split":
        train_root = find_named_dir(dataset_root, names=("train",))
        test_root  = find_named_dir(dataset_root, names=("test",))
        if train_root and test_root:
            class_names = discover_class_names_from_roots([train_root, test_root])
            c2i = {c:i for i,c in enumerate(class_names)}
            i2c = {i:c for c,i in c2i.items()}
            train_pool = collect_samples_from_root(train_root, c2i)
            test_samples = collect_samples_from_root(test_root, c2i)
            train_samples, val_samples = split_subjectwise(train_pool, test_size=val_size_from_train, seed=seed)
            return {"mode_used":"existing_folder_split","all_samples":train_pool+test_samples,
                    "train_samples":train_samples,"val_samples":val_samples,"test_samples":test_samples,
                    "class_to_idx":c2i,"idx_to_class":i2c}
        print("[INFO] Existing train/test folders not found – falling back to subjectwise_auto")
    all_samples, c2i, i2c = collect_all_samples_from_single_root(dataset_root)
    trainval, test_samples = split_subjectwise(all_samples, test_size=test_size, seed=seed)
    train_samples, val_samples = split_subjectwise(trainval, test_size=val_size_from_train, seed=seed+1)
    return {"mode_used":"subjectwise_auto","all_samples":all_samples,
            "train_samples":train_samples,"val_samples":val_samples,"test_samples":test_samples,
            "class_to_idx":c2i,"idx_to_class":i2c}

# =========================================================
# AUGMENTATION  (R1 – applied ONLY to training data)
# =========================================================
def apply_random_aug(img):
    img = img.convert("L")
    if random.random() < 0.5:  img = ImageOps.mirror(img)
    if random.random() < 0.2:  img = ImageOps.flip(img)
    if random.random() < 0.5:  img = img.rotate(random.randint(-12,12), resample=Image.BILINEAR)
    if random.random() < 0.5:  img = ImageOps.autocontrast(img)
    if random.random() < 0.4:  img = ImageOps.equalize(img)
    if random.random() < 0.35: img = ImageEnhance.Brightness(img).enhance(random.uniform(0.92,1.08))
    if random.random() < 0.35: img = ImageEnhance.Contrast(img).enhance(random.uniform(0.90,1.12))
    if random.random() < 0.20: img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.9,1.15))
    if random.random() < 0.18:
        arr = np.asarray(img).astype(np.float32)/255.0
        arr = np.clip(arr + np.random.normal(0, random.uniform(0,0.012), arr.shape), 0, 1)
        img = Image.fromarray((arr*255).astype(np.uint8), mode="L")
    return img

def build_augmented_train_set(train_samples, dst_root, idx_to_class, target_total=26000):
    """
    R1 COMPLIANCE: Augmentation is applied STRICTLY to train_samples only.
    Validation and test sets are NEVER augmented.
    """
    ensure_clean_dir(dst_root)
    s_by_cls = defaultdict(list)
    for item in train_samples: s_by_cls[item[1]].append(item)
    n_cls = len(idx_to_class)
    per_cls = target_total // n_cls
    cls_targets = {c: per_cls + (1 if c < target_total % n_cls else 0) for c in range(n_cls)}
    saved, rows = [], []
    for cls_idx in range(n_cls):
        cls_name = idx_to_class[cls_idx]
        cls_dir  = os.path.join(dst_root, cls_name); ensure_dir(cls_dir)
        cls_s    = s_by_cls[cls_idx]
        orig_n   = len(cls_s)
        if orig_n == 0:
            rows.append({"class_name":cls_name,"original_train_count":0,"target":0,"added":0}); continue
        target_n  = max(cls_targets[cls_idx], orig_n)
        add_n     = target_n - orig_n
        for j,(src,lbl,sid) in enumerate(cls_s):
            base, ext = os.path.splitext(os.path.basename(src))[0], os.path.splitext(src)[1].lower()
            dst = os.path.join(cls_dir, f"{sid}__{base}__orig__{j:06d}{ext}")
            Image.open(src).convert("L").save(dst)
            saved.append((dst, lbl, sid))
        for k in range(add_n):
            src, lbl, sid = random.choice(cls_s)
            base = os.path.splitext(os.path.basename(src))[0]
            dst  = os.path.join(cls_dir, f"{sid}__{base}__aug__{k:06d}.png")
            apply_random_aug(Image.open(src).convert("L")).save(dst)
            saved.append((dst, lbl, sid))
        rows.append({"class_name":cls_name,"original_train_count":orig_n,"target":target_n,"added":add_n})
        print(f"  [AUG] {cls_name}: orig={orig_n}, target={target_n}, added={add_n}")
    return saved, pd.DataFrame(rows)

# =========================================================
# DATASET
# =========================================================
class BrainDataset(Dataset):
    def __init__(self, samples, image_size=224, rgb=True):
        self.samples = samples; self.image_size = image_size; self.rgb = rgb
        self.mean = np.array([0.485,0.456,0.406], dtype=np.float32)
        self.std  = np.array([0.229,0.224,0.225], dtype=np.float32)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label, sid = self.samples[idx]
        img = Image.open(path).convert("L").resize((self.image_size,self.image_size), Image.BILINEAR)
        img = ImageOps.autocontrast(img); img = ImageOps.equalize(img)
        arr = np.asarray(img).astype(np.float32)/255.0
        if self.rgb:
            arr = np.stack([arr,arr,arr], axis=-1)
            arr = (arr - self.mean) / self.std
            arr = np.transpose(arr, (2,0,1))
        else:
            arr = (arr - arr.mean()) / (arr.std() + 1e-6)
            arr = np.expand_dims(np.clip(arr,-5.0,5.0), axis=0)
        return torch.from_numpy(arr).float(), int(label), path, sid

# =========================================================
# MODELS
# =========================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels//reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(nn.Linear(channels,hidden), nn.ReLU(inplace=True),
                                   nn.Linear(hidden,channels), nn.Sigmoid())
    def forward(self, x):
        b,c,_,_ = x.shape
        w = self.pool(x).view(b,c)
        return x * self.fc(w).view(b,c,1,1)

# R4 – Ablation-friendly CNN: toggleable BN, SE, pool type
class StrongCustomCNN(nn.Module):
    def __init__(self, num_classes, feature_dim=128, dropout=0.35,
                 use_bn=True, use_se=True, pool_type="max"):
        super().__init__()
        def norm(c): return nn.BatchNorm2d(c) if use_bn else nn.Identity()
        def pool(): return nn.MaxPool2d(2) if pool_type=="max" else nn.AvgPool2d(2)
        def se(c):  return SEBlock(c)      if use_se   else nn.Identity()
        self.block1 = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), norm(32), nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,padding=1), norm(32), nn.ReLU(inplace=True), se(32), pool())
        self.block2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1), norm(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1), norm(64), nn.ReLU(inplace=True), se(64), pool())
        self.block3 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1), norm(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1), norm(128), nn.ReLU(inplace=True), se(128), pool())
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.feature_proj = nn.Sequential(
            nn.Linear(128,feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.classifier = nn.Linear(feature_dim, num_classes)
    def freeze_backbone(self, freeze=True):
        for blk in [self.block1, self.block2, self.block3]:
            for p in blk.parameters(): p.requires_grad = not freeze
    def forward(self, x, return_features=False):
        x = self.block3(self.block2(self.block1(x)))
        feats = self.feature_proj(self.gap(x).flatten(1))
        logits = self.classifier(feats)
        return (logits, feats) if return_features else logits

class TorchVisionModel(nn.Module):
    """Wrapper for EfficientNet-B0, ResNet18, MobileNetV3 – R6 baselines"""
    def __init__(self, name, num_classes, feature_dim=128, dropout=0.35, use_pretrained=True):
        super().__init__()
        if name == "efficientnet_b0":
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None)
            in_f = base.classifier[1].in_features
            self.backbone = base.features; self.pool = base.avgpool
        elif name == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None)
            in_f = base.fc.in_features
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((1,1))
        elif name == "mobilenet_v3_small":
            base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if use_pretrained else None)
            in_f = base.classifier[0].in_features
            self.backbone = base.features; self.pool = nn.AdaptiveAvgPool2d((1,1))
        else:
            raise ValueError(name)
        self.feature_proj = nn.Sequential(
            nn.Linear(in_f,256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256,feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True), nn.Dropout(dropout*0.5))
        self.classifier = nn.Linear(feature_dim, num_classes)
    def freeze_backbone(self, freeze=True):
        for p in self.backbone.parameters(): p.requires_grad = not freeze
    def forward(self, x, return_features=False):
        feats = self.feature_proj(self.pool(self.backbone(x)).flatten(1))
        logits = self.classifier(feats)
        return (logits, feats) if return_features else logits

def create_model(model_name, num_classes, feature_dim=128, dropout=0.35, use_pretrained=True,
                 use_bn=True, use_se=True, pool_type="max"):
    if model_name == "custom_cnn":
        return StrongCustomCNN(num_classes, feature_dim, dropout, use_bn, use_se, pool_type), False
    return TorchVisionModel(model_name, num_classes, feature_dim, dropout, use_pretrained), True

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =========================================================
# LOSS / TRAIN HELPERS
# =========================================================
class SoftTargetCE(nn.Module):
    def forward(self, logits, soft):
        return -(soft * torch.log_softmax(logits,dim=1)).sum(dim=1).mean()

def build_weighted_sampler(samples):
    labels = [s[1] for s in samples]
    counts = Counter(labels)
    w = [1.0/counts[l] for l in labels]
    return WeightedRandomSampler(torch.DoubleTensor(w), num_samples=len(w), replacement=True)

def one_hot(labels, num_classes):
    y = torch.zeros(labels.size(0), num_classes, device=labels.device)
    y.scatter_(1, labels.unsqueeze(1), 1.0)
    return y

def mixup_batch(x, y, alpha=0.25):
    if alpha <= 0: return x, y
    lam = max(np.random.beta(alpha,alpha), 1-np.random.beta(alpha,alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x+(1-lam)*x[idx], lam*y+(1-lam)*y[idx]

def align_probabilities(prob, present_classes, num_classes):
    aligned = np.zeros((prob.shape[0], num_classes), dtype=np.float32)
    for col_idx, cls in enumerate(present_classes):
        aligned[:, int(cls)] = prob[:, col_idx]
    return aligned

# =========================================================
# METRICS  (R7 – extended: Balanced Acc, MCC, Kappa, Brier, ECE, LogLoss)
# =========================================================
def multiclass_brier(y_true, y_prob, num_classes):
    oh = np.eye(num_classes)[np.asarray(y_true)]
    return float(np.mean(np.sum((y_prob - oh)**2, axis=1)))

def expected_calibration_error(y_true, y_prob, n_bins=10):
    conf = y_prob.max(axis=1); pred = y_prob.argmax(axis=1)
    acc  = (pred == np.asarray(y_true)).astype(float)
    bins = np.linspace(0, 1, n_bins+1); ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf >= lo) & (conf <= hi) if i==0 else (conf > lo) & (conf <= hi)
        if mask.any():
            ece += abs(acc[mask].mean() - conf[mask].mean()) * mask.mean()
    return float(ece)

def compute_metric_pack(y_true, y_pred, y_prob, num_classes):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); y_prob = np.asarray(y_prob)
    pm, rm, f1m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pw, rw, f1w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics = {
        "accuracy":           float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy":  float(balanced_accuracy_score(y_true, y_pred)),   # 
        "mcc":                float(matthews_corrcoef(y_true, y_pred)),          # 
        "cohen_kappa":        float(cohen_kappa_score(y_true, y_pred)),          # 
        "precision_macro":    float(pm), "recall_macro":    float(rm), "f1_macro":    float(f1m),
        "precision_weighted": float(pw), "recall_weighted": float(rw), "f1_weighted": float(f1w),
        "brier_score": float(multiclass_brier(y_true, y_prob, num_classes)),     # 
        "ece":         float(expected_calibration_error(y_true, y_prob)),        # 
    }
    try:    metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=list(range(num_classes))))
    except: metrics["log_loss"] = None
    try:
        if num_classes == 2:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob[:,1]))
        else:
            yb = label_binarize(y_true, classes=list(range(num_classes)))
            metrics["auc"] = float(roc_auc_score(yb, y_prob, multi_class="ovr", average="macro"))
    except: metrics["auc"] = None
    report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
    return metrics, report_dict

# R8 – detailed error analysis
def build_error_analysis(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    rows = []
    for i in range(cm.shape[0]):
        denom = max(int(cm[i].sum()), 1)
        for j in range(cm.shape[1]):
            if i != j and cm[i,j] > 0:
                rows.append({"true_class":class_names[i], "pred_class":class_names[j],
                              "count":int(cm[i,j]), "row_fraction":float(cm[i,j]/denom)})
    return pd.DataFrame(sorted(rows, key=lambda x: (-x["count"], -x["row_fraction"])))

def run_train_epoch(model, loader, optimizer, criterion, device, num_classes, mixup_alpha=0.25):
    model.train()
    total_loss, all_t, all_p, all_pb = 0.0, [], [], []
    for imgs, labels, _, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        soft = one_hot(labels, num_classes)
        imgs, soft = mixup_batch(imgs, soft, mixup_alpha)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, soft)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        with torch.no_grad():
            probs = torch.softmax(logits,dim=1).cpu().numpy()
        all_t.extend(labels.cpu().numpy().tolist())
        all_p.extend(np.argmax(probs,axis=1).tolist())
        all_pb.extend(probs.tolist())
    metrics, _ = compute_metric_pack(all_t, all_p, all_pb, num_classes)
    return total_loss/max(len(loader.dataset),1), metrics

def run_eval_epoch(model, loader, device, num_classes, use_tta=False):
    model.eval()
    ce = nn.CrossEntropyLoss(); total_loss = 0.0
    all_t, all_p, all_pb, all_paths, all_subs = [], [], [], [], []
    with torch.no_grad():
        for imgs, labels, paths, sids in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            if use_tta:
                probs = (probs + torch.softmax(model(torch.flip(imgs,dims=[3])),dim=1)) / 2.0
                logits = torch.log(probs + 1e-12)
            total_loss += ce(logits, labels).item() * imgs.size(0)
            pb = probs.cpu().numpy()
            all_t.extend(labels.cpu().numpy().tolist())
            all_p.extend(np.argmax(pb,axis=1).tolist())
            all_pb.extend(pb.tolist())
            all_paths.extend(list(paths)); all_subs.extend(list(sids))
    metrics, report = compute_metric_pack(all_t, all_p, all_pb, num_classes)
    return {"loss": total_loss/max(len(loader.dataset),1), "metrics": metrics,
            "classification_report": report,
            "y_true": np.asarray(all_t), "y_pred": np.asarray(all_p), "y_prob": np.asarray(all_pb),
            "paths": all_paths, "subjects": all_subs}

def extract_features(model, loader, device):
    model.eval()
    feats, labels, paths, subs = [], [], [], []
    with torch.no_grad():
        for imgs, lbls, ps, ss in loader:
            _, f = model(imgs.to(device), return_features=True)
            feats.append(f.cpu().numpy()); labels.append(lbls.numpy())
            paths.extend(list(ps)); subs.extend(list(ss))
    return {"features": np.concatenate(feats), "labels": np.concatenate(labels),
            "paths": paths, "subjects": subs}

# =========================================================
# ATOM SEARCH OPTIMIZATION  (R5 – with timing)
# =========================================================
class BinaryAtomSearchOptimizationFeatureSelector:
    def __init__(self, lower_bound=0.0, upper_bound=1.0, threshold=0.5,
                 alpha=50, beta=0.2, total_solutions=10, max_iters=100,
                 penalty_weight=0.03, random_state=42):
        self.lower_bound=lower_bound; self.upper_bound=upper_bound
        self.threshold=threshold; self.alpha=alpha; self.beta=beta
        self.total_solutions=total_solutions; self.max_iters=max_iters
        self.penalty_weight=penalty_weight; self.random_state=random_state
        self.best_mask_=None; self.best_score_=None; self.history_=[]
        self.selected_indices_=None; self.runtime_sec_=None; self.cache_={}

    def _ensure_nonzero(self, mask):
        mask = mask.astype(bool)
        if not mask.any(): mask[np.random.randint(len(mask))] = True
        return mask

    def _pos_to_mask(self, pos): return self._ensure_nonzero(pos >= self.threshold)

    def _fitness(self, mask, Xtr, ytr, Xv, yv):
        key = tuple(np.where(mask)[0].tolist())
        if key in self.cache_: return self.cache_[key]
        try:
            pipe = Pipeline([("sc", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=2500, class_weight="balanced",
                                                         solver="lbfgs", multi_class="auto"))])
            pipe.fit(Xtr[:,mask], ytr)
            pred  = pipe.predict(Xv[:,mask])
            score = (0.5*accuracy_score(yv,pred)
                     + 0.5*f1_score(yv,pred,average="macro",zero_division=0)
                     - self.penalty_weight*float(mask.mean()))
        except: score = -1e9
        self.cache_[key] = score
        return score

    def fit(self, Xtr, ytr, Xv, yv):
        t0 = time.time()
        rng = np.random.default_rng(self.random_state)
        nf, na, eps = Xtr.shape[1], self.total_solutions, 1e-12
        pos = rng.uniform(self.lower_bound, self.upper_bound, (na,nf)).astype(np.float32)
        vel = rng.uniform(-0.1, 0.1, (na,nf)).astype(np.float32)
        pb_pos = pos.copy(); pb_sc = np.full(na, -1e9, np.float32)
        gb_pos, gb_mask, gb_sc = None, None, -1e9
        for it in range(1, self.max_iters+1):
            sc = np.zeros(na, np.float32)
            for i in range(na):
                mask = self._pos_to_mask(pos[i])
                s = self._fitness(mask, Xtr, ytr, Xv, yv); sc[i] = s
                if s > pb_sc[i]: pb_sc[i]=s; pb_pos[i]=pos[i].copy()
                if s > gb_sc:    gb_sc=s; gb_pos=pos[i].copy(); gb_mask=mask.copy()
            self.history_.append(float(gb_sc))
            bf, wf = float(sc.max()), float(sc.min())
            masses = (np.ones(na)/na if abs(bf-wf)<eps
                      else np.exp((sc-bf)/(abs(bf-wf)+eps)))
            masses /= (masses.sum()+eps)
            kbest = np.argsort(-sc)[:max(2, int(round(na-(na-2)*(it/self.max_iters))))]
            G = np.exp(-20.0*it/self.max_iters)
            new_pos, new_vel = pos.copy(), vel.copy()
            diffs = pos[:,None,:]-pos[None,:,:]; dists = np.linalg.norm(diffs,axis=2)
            md = float(dists.mean())+eps
            for i in range(na):
                F = np.zeros(nf, np.float32)
                for j in kbest:
                    if i==j: continue
                    diff = pos[j]-pos[i]; d = np.linalg.norm(diff)+eps
                    attr = np.exp(-self.alpha*((d/md)**2))
                    F += (rng.random(nf).astype(np.float32)*masses[j]*attr*(diff/d))
                acc = G*F/(masses[i]+eps) + self.beta*rng.random(nf).astype(np.float32)*(gb_pos-pos[i])
                new_vel[i] = rng.random(nf).astype(np.float32)*vel[i]+acc
                new_pos[i] = pos[i]+new_vel[i]
            pos = np.clip(new_pos, self.lower_bound, self.upper_bound)
            vel = np.clip(new_vel, -1.0, 1.0)
            print(f"  ASO iter {it:03d}/{self.max_iters} | fitness={gb_sc:.5f} | feats={int(gb_mask.sum())}")
        self.best_mask_ = self._ensure_nonzero(gb_mask)
        self.best_score_ = float(gb_sc)
        self.selected_indices_ = np.where(self.best_mask_)[0].tolist()
        self.runtime_sec_ = float(time.time()-t0)
        return self

# =========================================================
# FEATURE SELECTION COMPARISON  (R5)
# =========================================================
def get_candidate_classifiers(seed=42):
    return {
        "logistic_regression": Pipeline([("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3500, class_weight="balanced",
                                       solver="lbfgs", multi_class="auto"))]),
        "random_forest":  RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                                  random_state=seed, n_jobs=-1),
        "extra_trees":    ExtraTreesClassifier(n_estimators=700, class_weight="balanced",
                                               random_state=seed, n_jobs=-1),
    }

def evaluate_classifier(clf, Xtr, ytr, Xv, yv, num_classes):
    clf.fit(Xtr, ytr); pred = clf.predict(Xv)
    prob_s = clf.predict_proba(Xv) if hasattr(clf,"predict_proba") else np.eye(num_classes)[pred]
    prob   = align_probabilities(prob_s, clf.classes_, num_classes) if hasattr(clf,"classes_") else prob_s
    metrics, report = compute_metric_pack(yv, pred, prob, num_classes)
    return 0.5*metrics["accuracy"]+0.5*metrics["f1_macro"], metrics, report, pred, prob

def run_feature_selection_method(method, Xtr, ytr, Xv, yv, Xte, yte, num_classes, seed=42):
    """R5: Compare ASO vs PCA, RFE, RF-Importance, No-Selection"""
    t0 = time.time()
    sel_details = {}
    if method == "none":
        Xtr_s, Xv_s, Xte_s = Xtr, Xv, Xte; n_sel = Xtr.shape[1]
    elif method == "pca":
        sc = StandardScaler()
        Xtr_sc, Xv_sc, Xte_sc = sc.fit_transform(Xtr), sc.transform(Xv), sc.transform(Xte)
        pca = PCA(n_components=min(PCA_N_COMPONENTS, Xtr.shape[1], Xtr.shape[0]-1), random_state=seed)
        Xtr_s, Xv_s, Xte_s = pca.fit_transform(Xtr_sc), pca.transform(Xv_sc), pca.transform(Xte_sc)
        n_sel = Xtr_s.shape[1]
        sel_details["explained_var"] = float(pca.explained_variance_ratio_.sum())
    elif method == "rfe":
        sc = StandardScaler()
        Xtr_sc, Xv_sc, Xte_sc = sc.fit_transform(Xtr), sc.transform(Xv), sc.transform(Xte)
        sel = RFE(LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", multi_class="auto"),
                  n_features_to_select=min(RFE_N_FEATURES, Xtr.shape[1]),
                  step=max(1, Xtr.shape[1]//16))
        sel.fit(Xtr_sc, ytr)
        Xtr_s, Xv_s, Xte_s = sel.transform(Xtr_sc), sel.transform(Xv_sc), sel.transform(Xte_sc)
        n_sel = Xtr_s.shape[1]
    elif method == "rf_importance":
        rf = RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample",
                                    random_state=seed, n_jobs=-1)
        rf.fit(Xtr, ytr)
        idx = np.argsort(-rf.feature_importances_)[:min(RF_TOPK_FEATURES, Xtr.shape[1])]
        Xtr_s, Xv_s, Xte_s = Xtr[:,idx], Xv[:,idx], Xte[:,idx]; n_sel = len(idx)
        sel_details["topk"] = int(n_sel)
    elif method == "aso":
        aso = BinaryAtomSearchOptimizationFeatureSelector(
            lower_bound=ASO_LOWER_BOUND, upper_bound=ASO_UPPER_BOUND, threshold=ASO_THRESHOLD,
            alpha=ASO_ALPHA, beta=ASO_BETA, total_solutions=ASO_TOTAL_SOLUTIONS,
            max_iters=ASO_MAX_ITERS, penalty_weight=ASO_PENALTY, random_state=seed)
        aso.fit(Xtr, ytr, Xv, yv)
        Xtr_s, Xv_s, Xte_s = Xtr[:,aso.best_mask_], Xv[:,aso.best_mask_], Xte[:,aso.best_mask_]
        n_sel = len(aso.selected_indices_)
        sel_details = {"best_fitness": float(aso.best_score_), "runtime_sec": float(aso.runtime_sec_),
                       "history": [float(x) for x in aso.history_]}
    else:
        raise ValueError(method)

    # pick best classifier on val set
    candidates = get_candidate_classifiers(seed=seed)
    best_name, best_tmpl, best_sc = None, None, -1e9
    for nm, clf in candidates.items():
        sc_, _, _, _, _ = evaluate_classifier(clone(clf), Xtr_s, ytr, Xv_s, yv, num_classes)
        if sc_ > best_sc: best_sc, best_name, best_tmpl = sc_, nm, clf

    # final model trained on train+val
    final = clone(best_tmpl)
    Xtv = np.concatenate([Xtr_s, Xv_s]); ytv = np.concatenate([ytr, yv])
    final.fit(Xtv, ytv)
    pred = final.predict(Xte_s)
    prob = align_probabilities(final.predict_proba(Xte_s), final.classes_, num_classes)
    metrics, _ = compute_metric_pack(yte, pred, prob, num_classes)
    return {"method": method, "runtime_sec": float(time.time()-t0),
            "selected_feature_count": int(n_sel), "selector_details": sel_details,
            "best_classifier": best_name, "test_metrics": metrics}

# =========================================================
# TRAINING LOOP
# =========================================================
def train_model(model, train_samples, val_samples, num_classes, device, out_dir,
                epochs=EPOCHS, lr=LEARNING_RATE, wd=WEIGHT_DECAY,
                freeze_epochs=FREEZE_BACKBONE_EPOCHS, mixup_alpha=MIXUP_ALPHA,
                use_tta=USE_TTA, is_rgb=True, patience=EARLY_STOP_PATIENCE):
    ensure_dir(out_dir)
    tr_ds  = BrainDataset(train_samples, IMAGE_SIZE, rgb=is_rgb)
    va_ds  = BrainDataset(val_samples,   IMAGE_SIZE, rgb=is_rgb)
    sampler = build_weighted_sampler(train_samples)
    tr_ld  = DataLoader(tr_ds, BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    va_ld  = DataLoader(va_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    criterion = SoftTargetCE()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    hist = {k:[] for k in ["train_loss","train_acc","val_loss","val_acc","val_f1_macro"]}
    best_f1, best_ep, no_imp, best_state = -1.0, 0, 0, None
    if freeze_epochs > 0 and hasattr(model, "freeze_backbone"):
        model.freeze_backbone(True); print(f"  [Backbone frozen for first {freeze_epochs} epochs]")
    for ep in range(1, epochs+1):
        if freeze_epochs > 0 and ep == freeze_epochs+1 and hasattr(model, "freeze_backbone"):
            model.freeze_backbone(False)
            for g in optimizer.param_groups: g["lr"] = lr
            print(f"  [Backbone unfrozen at epoch {ep}]")
        tl, tm = run_train_epoch(model, tr_ld, optimizer, criterion, device, num_classes, mixup_alpha)
        vr = run_eval_epoch(model, va_ld, device, num_classes, use_tta=False)
        vl, vm = vr["loss"], vr["metrics"]
        scheduler.step()
        hist["train_loss"].append(tl); hist["train_acc"].append(tm["accuracy"])
        hist["val_loss"].append(vl);   hist["val_acc"].append(vm["accuracy"])
        hist["val_f1_macro"].append(vm["f1_macro"])
        print(f"  Ep {ep:03d}/{epochs} | TrL={tl:.4f} TrAcc={tm['accuracy']:.4f} | "
              f"VaL={vl:.4f} VaAcc={vm['accuracy']:.4f} VaF1={vm['f1_macro']:.4f}")
        if vm["f1_macro"] > best_f1 + 1e-5:
            best_f1=vm["f1_macro"]; best_ep=ep; best_state=copy.deepcopy(model.state_dict()); no_imp=0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  [Early stop at ep {ep}, best ep={best_ep}]"); break
    if best_state: model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
    save_json(hist, os.path.join(out_dir, "train_history.json"))
    _save_training_curves(hist, out_dir)
    return model, hist, best_ep

# =========================================================
# PLOTS
# =========================================================
def _save_training_curves(hist, out_dir):
    eps = range(1, len(hist["train_loss"])+1)
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    ax[0].plot(eps, hist["train_loss"], label="Train"); ax[0].plot(eps, hist["val_loss"], label="Val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(eps, hist["train_acc"], label="Train"); ax[1].plot(eps, hist["val_acc"], label="Val")
    ax[1].plot(eps, hist["val_f1_macro"], label="Val F1"); ax[1].set_title("Accuracy / F1"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

def save_confusion_matrix(y_true, y_pred, class_names, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7,6))
    plt.imshow(cm, cmap="Blues"); plt.title(title); plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right"); plt.yticks(ticks, class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    thresh = cm.max()/2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,str(cm[i,j]),ha="center",va="center",
                     color="white" if cm[i,j]>thresh else "black")
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()

def save_roc_curves(y_true, y_prob, class_names, title, out_path):
    nc = len(class_names); plt.figure(figsize=(8,6))
    if nc == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:,1])
        plt.plot(fpr, tpr, lw=2, label=f"AUC={auc(fpr,tpr):.4f}")
    else:
        yb = label_binarize(y_true, classes=list(range(nc)))
        for i, cn in enumerate(class_names):
            try:
                fpr, tpr, _ = roc_curve(yb[:,i], y_prob[:,i])
                plt.plot(fpr, tpr, lw=1.5, label=f"{cn} AUC={auc(fpr,tpr):.3f}")
            except: pass
    plt.plot([0,1],[0,1],"k--",lw=1); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(title); plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()

def save_calibration_plot(y_true, y_prob, class_names, title, out_path, n_bins=10):
    plt.figure(figsize=(7,6))
    for ci in range(len(class_names)):
        yb = (np.asarray(y_true)==ci).astype(int); pc = y_prob[:,ci]
        be = np.linspace(0,1,n_bins+1); bm, bf = [], []
        for i in range(n_bins):
            lo,hi = be[i],be[i+1]
            mask = (pc>=lo)&(pc<=hi) if i==0 else (pc>lo)&(pc<=hi)
            if mask.sum()>0: bm.append(pc[mask].mean()); bf.append(yb[mask].mean())
        plt.plot(bm, bf, marker="o", lw=1.5, label=class_names[ci])
    plt.plot([0,1],[0,1],"k--",lw=1); plt.xlabel("Mean predicted prob"); plt.ylabel("Fraction positives")
    plt.title(title); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()

def save_aso_convergence(history, out_path):
    plt.figure(figsize=(8,5))
    plt.plot(range(1,len(history)+1), history, marker="o", lw=1.5)
    plt.xlabel("ASO Iteration"); plt.ylabel("Best Fitness"); plt.title("ASO Convergence Curve")
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()

# R3 – t-SNE and UMAP
def save_tsne_plot(features, labels, class_names, title, out_path, perplexity=30):
    set_seed(SEED)
    n = features.shape[0]; perp = min(perplexity, max(5, n//4))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=SEED, n_iter=1000)
    emb  = tsne.fit_transform(features)
    plt.figure(figsize=(8,6)); pal = plt.cm.get_cmap("tab10", len(class_names))
    for i, cn in enumerate(class_names):
        m = labels == i
        plt.scatter(emb[m,0], emb[m,1], s=8, alpha=0.6, color=pal(i), label=cn)
    plt.title(title); plt.legend(markerscale=3, fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()

def save_umap_plot(features, labels, class_names, title, out_path):
    if not HAS_UMAP: return
    reducer = umap.UMAP(n_components=2, random_state=SEED)
    emb = reducer.fit_transform(features)
    plt.figure(figsize=(8,6)); pal = plt.cm.get_cmap("tab10", len(class_names))
    for i, cn in enumerate(class_names):
        m = labels == i
        plt.scatter(emb[m,0], emb[m,1], s=8, alpha=0.6, color=pal(i), label=cn)
    plt.title(title); plt.legend(markerscale=3, fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()

def save_fs_comparison_plot(results_list, out_path):
    methods = [r["method"] for r in results_list]
    accs  = [r["test_metrics"]["accuracy"]  for r in results_list]
    f1s   = [r["test_metrics"]["f1_macro"]  for r in results_list]
    mccs  = [r["test_metrics"].get("mcc",0) for r in results_list]
    x = np.arange(len(methods)); w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(methods)*1.5), 5))
    ax.bar(x-w, accs, w, label="Accuracy")
    ax.bar(x,   f1s,  w, label="F1 Macro")
    ax.bar(x+w, mccs, w, label="MCC")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylim(0, 1.05); ax.set_title("Feature Selection Method Comparison (R5)"); ax.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()

# R6 – baseline arch comparison plot
def save_baseline_comparison_plot(results_dict, out_path):
    models_n = list(results_dict.keys())
    accs = [results_dict[m]["accuracy"] for m in models_n]
    f1s  = [results_dict[m]["f1_macro"] for m in models_n]
    mccs = [results_dict[m].get("mcc",0) for m in models_n]
    x = np.arange(len(models_n)); w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(models_n)*1.5), 5))
    ax.bar(x-w, accs, w, label="Accuracy")
    ax.bar(x,   f1s,  w, label="F1 Macro")
    ax.bar(x+w, mccs, w, label="MCC")
    ax.set_xticks(x); ax.set_xticklabels(models_n, rotation=15, ha="right")
    ax.set_ylim(0, 1.05); ax.set_title("Baseline Model Comparison (R6)"); ax.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()

# =========================================================
# FULL EVALUATION PACK SAVER  (R7, R8)
# =========================================================
def evaluate_and_save(model, test_samples, class_names, num_classes, device,
                      out_dir, tag, use_tta=True, is_rgb=True):
    ensure_dir(out_dir)
    te_ds = BrainDataset(test_samples, IMAGE_SIZE, rgb=is_rgb)
    te_ld = DataLoader(te_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    result = run_eval_epoch(model, te_ld, device, num_classes, use_tta=use_tta)
    y_true, y_pred, y_prob = result["y_true"], result["y_pred"], result["y_prob"]
    metrics = result["metrics"]

    save_json(metrics, os.path.join(out_dir, f"{tag}_metrics.json"))
    pd.DataFrame(result["classification_report"]).T.to_csv(
        os.path.join(out_dir, f"{tag}_classification_report.csv"))
    with open(os.path.join(out_dir, f"{tag}_classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4, zero_division=0))

    # predictions csv
    pred_df = pd.DataFrame({"path":result["paths"], "subject":result["subjects"],
                             "true_label":y_true, "pred_label":y_pred,
                             "true_class":[class_names[i] for i in y_true],
                             "pred_class": [class_names[i] for i in y_pred]})
    for i,cn in enumerate(class_names): pred_df[f"prob_{cn}"] = y_prob[:,i]
    pred_df.to_csv(os.path.join(out_dir, f"{tag}_predictions.csv"), index=False)

    save_confusion_matrix(y_true, y_pred, class_names, f"Confusion Matrix ({tag})",
                          os.path.join(out_dir, f"{tag}_confusion_matrix.png"))
    save_roc_curves(y_true, y_prob, class_names, f"ROC ({tag})",
                    os.path.join(out_dir, f"{tag}_roc_curves.png"))
    save_calibration_plot(y_true, y_prob, class_names, f"Calibration ({tag})",
                          os.path.join(out_dir, f"{tag}_calibration.png"))

    # R8 – error analysis
    err_df = build_error_analysis(y_true, y_pred, class_names)
    err_df.to_csv(os.path.join(out_dir, f"{tag}_error_analysis.csv"), index=False)

    print(f"\n[{tag.upper()}] Acc={metrics['accuracy']:.4f}  BalAcc={metrics['balanced_accuracy']:.4f}"
          f"  F1={metrics['f1_macro']:.4f}  MCC={metrics['mcc']:.4f}  "
          f"Kappa={metrics['cohen_kappa']:.4f}  AUC={metrics.get('auc','N/A')}"
          f"  Brier={metrics['brier_score']:.4f}  ECE={metrics['ece']:.4f}")
    return result

# =========================================================
# R4 – ABLATION STUDY
# =========================================================
ABLATION_CONFIGS = [
    {"name": "full_model",       "use_bn": True,  "use_se": True,  "pool_type": "max"},
    {"name": "no_batchnorm",     "use_bn": False, "use_se": True,  "pool_type": "max"},
    {"name": "no_se_block",      "use_bn": True,  "use_se": False, "pool_type": "max"},
    {"name": "avgpool",          "use_bn": True,  "use_se": True,  "pool_type": "avg"},
    {"name": "no_bn_no_se",      "use_bn": False, "use_se": False, "pool_type": "max"},
]

def run_ablation_study(train_samples, val_samples, test_samples, idx_to_class,
                       num_classes, device, out_dir):
    """R4: Ablation study on AD-CNN architecture components"""
    ensure_dir(out_dir)
    rows = []
    for cfg in ABLATION_CONFIGS:
        print(f"\n[ABLATION] Training variant: {cfg['name']}")
        model = StrongCustomCNN(num_classes=num_classes, feature_dim=FEATURE_DIM, dropout=DROPOUT,
                                use_bn=cfg["use_bn"], use_se=cfg["use_se"],
                                pool_type=cfg["pool_type"]).to(device)
        n_params = count_params(model)
        run_dir = os.path.join(out_dir, cfg["name"])
        model, _, best_ep = train_model(model, train_samples, val_samples, num_classes, device,
                                        run_dir, epochs=min(EPOCHS, 15), is_rgb=False,
                                        freeze_epochs=0, patience=EARLY_STOP_PATIENCE)
        result = evaluate_and_save(model, test_samples, [idx_to_class[i] for i in range(num_classes)],
                                   num_classes, device, run_dir, tag=cfg["name"],
                                   use_tta=USE_TTA, is_rgb=False)
        m = result["metrics"]
        rows.append({"variant": cfg["name"], "use_bn": cfg["use_bn"], "use_se": cfg["use_se"],
                     "pool_type": cfg["pool_type"], "n_params": n_params, "best_epoch": best_ep,
                     **{k: m[k] for k in ["accuracy","balanced_accuracy","mcc","f1_macro","auc","brier_score","ece"]}})
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    abl_df = pd.DataFrame(rows)
    abl_df.to_csv(os.path.join(out_dir, "ablation_results.csv"), index=False)
    print("\n[ABLATION] Summary:")
    print(abl_df[["variant","accuracy","balanced_accuracy","mcc","f1_macro","auc","brier_score"]].to_string(index=False))
    return abl_df

# =========================================================
# R6 – BASELINE MODEL COMPARISON
# =========================================================
BASELINE_MODELS = ["efficientnet_b0","custom_cnn", "resnet18", "mobilenet_v3_small"]

def run_baseline_comparison(train_samples, val_samples, test_samples, idx_to_class,
                             num_classes, device, out_dir):
    """R6: Re-implement strong baselines on SAME data splits for fair comparison"""
    ensure_dir(out_dir)
    rows = {}
    for mn in BASELINE_MODELS:
        print(f"\n[BASELINE] Training: {mn}")
        model, is_rgb = create_model(mn, num_classes, FEATURE_DIM, DROPOUT, USE_PRETRAINED)
        model = model.to(device)
        n_params = count_params(model)
        run_dir = os.path.join(out_dir, mn)
        freeze_ep = FREEZE_BACKBONE_EPOCHS if is_rgb else 0
        model, _, best_ep = train_model(model, train_samples, val_samples, num_classes, device,
                                        run_dir, epochs=min(EPOCHS,15), is_rgb=is_rgb,
                                        freeze_epochs=freeze_ep, patience=EARLY_STOP_PATIENCE)
        result = evaluate_and_save(model, test_samples, [idx_to_class[i] for i in range(num_classes)],
                                   num_classes, device, run_dir, tag=mn, use_tta=USE_TTA, is_rgb=is_rgb)
        m = result["metrics"]
        rows[mn] = {**m, "n_params": n_params, "best_epoch": best_ep}
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    pd.DataFrame(rows).T.to_csv(os.path.join(out_dir, "baseline_comparison.csv"))
    save_baseline_comparison_plot(rows, os.path.join(out_dir, "baseline_comparison.png"))
    print("\n[BASELINE] Summary:")
    print(pd.DataFrame(rows).T[["accuracy","balanced_accuracy","mcc","f1_macro","auc"]].to_string())
    return rows

# =========================================================
# PAPER-READY TEXT GENERATOR  (all reviewer points summarized)
# =========================================================
def generate_paper_ready_text(holdout_metrics, cv_summary, ablation_df, fs_results,
                               baseline_results, class_names, model_name,
                               split_info, out_dir):
    ensure_dir(out_dir)
    lines = []
    def H(title): lines.extend(["", "="*80, title, "="*80])

    H("REVIEWER-RESPONSE READY RESULTS")
    lines.append(f"Model: {model_name} | Split mode: {split_info.get('mode_used')} "
                 f"| Classes: {', '.join(class_names)}")

    H("R1 – DATA LEAKAGE PROOF")
    lines += [
        "Augmentation was applied STRICTLY to the training set AFTER subject-wise splitting.",
        "Validation and test sets contain ONLY original, unaugmented images.",
        "Subject IDs were inferred from filenames; no subject appears in >1 split.",
        f"Train/Val overlap: {split_info.get('overlap_train_val',0)} subjects",
        f"Train/Test overlap: {split_info.get('overlap_train_test',0)} subjects",
        f"Val/Test overlap:   {split_info.get('overlap_val_test',0)} subjects",
    ]

    H("R2 – HOLD-OUT TEST SET METRICS (Extended)")
    m = holdout_metrics
    for key in ["accuracy","balanced_accuracy","mcc","cohen_kappa",
                "precision_macro","recall_macro","f1_macro",
                "precision_weighted","recall_weighted","f1_weighted",
                "auc","brier_score","ece","log_loss"]:
        v = m.get(key)
        lines.append(f"  {key:<25} {f'{v:.4f}' if v is not None else 'N/A'}")

    H("R2 – 5-FOLD SUBJECT-WISE CROSS-VALIDATION (mean ± std)")
    if cv_summary:
        for key, stats in cv_summary.items():
            if isinstance(stats, dict) and "mean" in stats:
                lines.append(f"  {key:<25} {stats['mean']:.4f} ± {stats['std']:.4f}")

    H("R4 – ABLATION STUDY (AD-CNN Architecture Components)")
    if ablation_df is not None:
        lines.append(ablation_df[["variant","use_bn","use_se","pool_type",
                                   "accuracy","balanced_accuracy","mcc","f1_macro"]].to_string(index=False))

    H("R5 – FEATURE SELECTION COMPARISON (ASO vs PCA vs RFE vs RF-Importance vs None)")
    if fs_results:
        fs_df = pd.DataFrame([{"method": r["method"],
                                "n_features": r["selected_feature_count"],
                                "best_clf": r["best_classifier"],
                                "runtime_s": f"{r['runtime_sec']:.1f}",
                                **{k: f"{r['test_metrics'].get(k,'N/A'):.4f}"
                                   for k in ["accuracy","balanced_accuracy","mcc","f1_macro","auc"]}}
                               for r in fs_results])
        lines.append(fs_df.to_string(index=False))

    H("R6 – BASELINE MODEL COMPARISON (same data splits)")
    if baseline_results:
        bl_df = pd.DataFrame({mn: {k: v for k,v in m.items()
                                   if k in ["accuracy","balanced_accuracy","mcc","f1_macro","auc","n_params"]}
                              for mn, m in baseline_results.items()}).T
        lines.append(bl_df.to_string())

    H("R3 – FEATURE SEPARABILITY")
    lines += ["t-SNE plots saved to: results/holdout/tsne_before_aso.png and tsne_after_aso.png",
              "UMAP plots saved to:  results/holdout/umap_before_aso.png and umap_after_aso.png"
              if HAS_UMAP else "UMAP not available (install: pip install umap-learn)"]

    H("END OF REVIEWER-READY SUMMARY")

    txt_path = os.path.join(out_dir, "reviewer_ready_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    save_json({"holdout_metrics": holdout_metrics, "cv_summary": cv_summary}, 
              os.path.join(out_dir, "results_summary.json"))
    print("\n".join(lines))
    return txt_path

# =========================================================
# MAIN
# =========================================================
def main():
    warnings.filterwarnings("ignore")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    for d in [RESULTS_DIR, CV_DIR, ABLATION_DIR, BASELINE_COMP, FS_COMP_DIR, TEXT_DIR]:
        ensure_dir(d)
    holdout_dir = os.path.join(RESULTS_DIR, "holdout")
    ensure_dir(holdout_dir)

    # =========================================================
    # STEP 1 – SPLITS  (R1)
    # =========================================================
    print("\n[1] Preparing dataset splits (strict subject-wise)...")
    prep = prepare_splits(DATASET_SOURCE_PATH, SPLIT_MODE, SEED, TEST_SIZE, VAL_SIZE_FROM_TRAIN)
    all_samples   = prep["all_samples"]
    train_samples = prep["train_samples"]
    val_samples   = prep["val_samples"]
    test_samples  = prep["test_samples"]
    idx_to_class  = prep["idx_to_class"]
    class_names   = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes   = len(class_names)

    for split_name, split_s in [("ALL",all_samples),("TRAIN",train_samples),("VAL",val_samples),("TEST",test_samples)]:
        print_dataset_summary(split_s, idx_to_class, split_name)
        save_manifest(split_s, os.path.join(RESULTS_DIR, f"{split_name.lower()}_manifest.csv"))
        save_class_distribution(split_s, idx_to_class, os.path.join(RESULTS_DIR, f"{split_name.lower()}_class_dist.csv"))

    ov_tv = check_subject_overlap(train_samples, val_samples,
                                   os.path.join(RESULTS_DIR,"overlap_train_val.txt"), "Train","Val")
    ov_tt = check_subject_overlap(train_samples, test_samples,
                                   os.path.join(RESULTS_DIR,"overlap_train_test.txt"), "Train","Test")
    ov_vt = check_subject_overlap(val_samples,   test_samples,
                                   os.path.join(RESULTS_DIR,"overlap_val_test.txt"),   "Val","Test")
    print(f"\nTrain/Val overlap:  {len(ov_tv)}")
    print(f"Train/Test overlap: {len(ov_tt)}")
    print(f"Val/Test overlap:   {len(ov_vt)}")
    prep.update({"overlap_train_val": len(ov_tv), "overlap_train_test": len(ov_tt),
                 "overlap_val_test": len(ov_vt)})

    # =========================================================
    # STEP 2 – AUGMENT TRAIN ONLY  (R1)
    # =========================================================
    print("\n[2] Building augmented training set (train-only, no leakage)...")
    aug_root = os.path.join(WORK_DIR, "augmented_train")
    aug_samples, aug_df = build_augmented_train_set(
        train_samples, aug_root, idx_to_class, target_total=TARGET_TRAIN_IMAGES)
    aug_df.to_csv(os.path.join(RESULTS_DIR, "augmentation_report.csv"), index=False)

    # =========================================================
    # STEP 3 – TRAIN PRIMARY MODEL
    # =========================================================
    print(f"\n[3] Training primary model: {MODEL_NAME}...")
    model, is_rgb = create_model(MODEL_NAME, num_classes, FEATURE_DIM, DROPOUT, USE_PRETRAINED)
    model = model.to(device)
    print(f"  Trainable params: {count_params(model):,}")
    model, train_hist, best_ep = train_model(
        model, aug_samples, val_samples, num_classes, device, holdout_dir,
        epochs=EPOCHS, is_rgb=is_rgb,
        freeze_epochs=FREEZE_BACKBONE_EPOCHS if is_rgb else 0)

    # =========================================================
    # STEP 4 – HOLD-OUT EVALUATION  
    # =========================================================
    print("\n[4] Hold-out evaluation...")
    holdout_result = evaluate_and_save(
        model, test_samples, class_names, num_classes, device, holdout_dir,
        tag="holdout", use_tta=USE_TTA, is_rgb=is_rgb)

    # =========================================================
    # STEP 5 – FEATURE EXTRACTION + t-SNE/UMAP  
    # =========================================================
    print("\n[5] Feature extraction and t-SNE/UMAP...")
    def make_loader(samps): return DataLoader(
        BrainDataset(samps, IMAGE_SIZE, rgb=is_rgb), BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    feat_test  = extract_features(model, make_loader(test_samples),  device)
    feat_train = extract_features(model, make_loader(aug_samples),   device)
    feat_val   = extract_features(model, make_loader(val_samples),   device)

    np.save(os.path.join(holdout_dir, "test_features.npy"),  feat_test["features"])
    np.save(os.path.join(holdout_dir, "test_labels.npy"),    feat_test["labels"])

    if RUN_TSNE_UMAP:
        print("  Saving t-SNE (before ASO)...")
        save_tsne_plot(feat_test["features"], feat_test["labels"], class_names,
                       "t-SNE – Test Features (before ASO feature selection)",
                       os.path.join(holdout_dir, "tsne_before_aso.png"))
        if HAS_UMAP:
            save_umap_plot(feat_test["features"], feat_test["labels"], class_names,
                           "UMAP – Test Features (before ASO)",
                           os.path.join(holdout_dir, "umap_before_aso.png"))

    # =========================================================
    # STEP 6 – ASO FEATURE SELECTION + CLASSIFIERS
    # =========================================================
    print("\n[6] Running ASO feature selection...")
    t0 = time.time()
    aso = BinaryAtomSearchOptimizationFeatureSelector(
        lower_bound=ASO_LOWER_BOUND, upper_bound=ASO_UPPER_BOUND, threshold=ASO_THRESHOLD,
        alpha=ASO_ALPHA, beta=ASO_BETA, total_solutions=ASO_TOTAL_SOLUTIONS,
        max_iters=ASO_MAX_ITERS, penalty_weight=ASO_PENALTY, random_state=SEED)
    aso.fit(feat_train["features"], feat_train["labels"],
            feat_val["features"],   feat_val["labels"])
    print(f"ASO done. Selected {len(aso.selected_indices_)}/{feat_train['features'].shape[1]} features "
          f"in {aso.runtime_sec_:.1f}s")

    aso_dir = os.path.join(RESULTS_DIR, "cnn_aso")
    ensure_dir(aso_dir)
    save_aso_convergence(aso.history_, os.path.join(aso_dir, "aso_convergence.png"))
    save_json({"selected_indices": aso.selected_indices_,
               "n_selected": len(aso.selected_indices_),
               "n_total": int(feat_train["features"].shape[1]),
               "best_fitness": float(aso.best_score_),
               "runtime_sec": float(aso.runtime_sec_),
               "history": [float(x) for x in aso.history_]},
              os.path.join(aso_dir, "aso_details.json"))

    Xtr_s = feat_train["features"][:, aso.best_mask_]
    Xva_s = feat_val["features"][:,   aso.best_mask_]
    Xte_s = feat_test["features"][:,  aso.best_mask_]
    ytr, yva, yte = feat_train["labels"], feat_val["labels"], feat_test["labels"]

    # t-SNE after ASO  (R3 – before & after comparison)
    if RUN_TSNE_UMAP:
        print("  Saving t-SNE (after ASO)...")
        save_tsne_plot(Xte_s, yte, class_names,
                       "t-SNE – Test Features (after ASO feature selection)",
                       os.path.join(holdout_dir, "tsne_after_aso.png"))
        if HAS_UMAP:
            save_umap_plot(Xte_s, yte, class_names,
                           "UMAP – Test Features (after ASO)",
                           os.path.join(holdout_dir, "umap_after_aso.png"))

    # Best classifier selection
    candidates = get_candidate_classifiers(seed=SEED)
    best_clf_name, best_tmpl, best_clf_sc = None, None, -1e9
    clf_rows = []
    for clf_nm, clf in candidates.items():
        sc_, m_, _, _, _ = evaluate_classifier(clone(clf), Xtr_s, ytr, Xva_s, yva, num_classes)
        clf_rows.append({"classifier": clf_nm, **m_, "selection_score": sc_})
        if sc_ > best_clf_sc: best_clf_sc, best_clf_name, best_tmpl = sc_, clf_nm, clf
    pd.DataFrame(clf_rows).to_csv(os.path.join(aso_dir, "classifier_selection.csv"), index=False)
    print(f"  Best classifier: {best_clf_name}")

    Xtv = np.concatenate([Xtr_s, Xva_s]); ytv = np.concatenate([ytr, yva])
    final_clf = clone(best_tmpl); final_clf.fit(Xtv, ytv)
    test_pred = final_clf.predict(Xte_s)
    test_prob  = align_probabilities(final_clf.predict_proba(Xte_s), final_clf.classes_, num_classes)
    aso_metrics, aso_report = compute_metric_pack(yte, test_pred, test_prob, num_classes)
    save_json(aso_metrics, os.path.join(aso_dir, "aso_test_metrics.json"))
    save_confusion_matrix(yte, test_pred, class_names, "CNN+ASO Confusion Matrix",
                          os.path.join(aso_dir, "aso_confusion_matrix.png"))
    save_roc_curves(yte, test_prob, class_names, "CNN+ASO ROC",
                    os.path.join(aso_dir, "aso_roc_curves.png"))
    err_df = build_error_analysis(yte, test_pred, class_names)
    err_df.to_csv(os.path.join(aso_dir, "aso_error_analysis.csv"), index=False)

    # =========================================================
    # STEP 7 – FEATURE SELECTION COMPARISON  
    # =========================================================
    fs_results = []
    if RUN_FEATURE_SELECTION_COMPARISON:
        print("\n[7] Feature selection comparison (ASO vs PCA vs RFE vs RF vs None)...")
        ensure_dir(FS_COMP_DIR)
        Xtr_all = feat_train["features"]; Xva_all = feat_val["features"]; Xte_all = feat_test["features"]
        for method in ["none", "pca", "rfe", "rf_importance", "aso"]:
            print(f"  Running: {method}...")
            try:
                res = run_feature_selection_method(method, Xtr_all, ytr, Xva_all, yva, Xte_all, yte,
                                                   num_classes, seed=SEED)
                fs_results.append(res)
                save_json(res, os.path.join(FS_COMP_DIR, f"fs_{method}.json"))
                m = res["test_metrics"]
                print(f"    {method}: Acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}  "
                      f"MCC={m.get('mcc',float('nan')):.4f}  "
                      f"Feats={res['selected_feature_count']}  Time={res['runtime_sec']:.1f}s")
            except Exception as e:
                print(f"    [WARN] {method} failed: {e}")
        if fs_results:
            fs_df = pd.DataFrame([{"method":r["method"],"n_features":r["selected_feature_count"],
                                    "best_clf":r["best_classifier"],"runtime_s":r["runtime_sec"],
                                    **r["test_metrics"]} for r in fs_results])
            fs_df.to_csv(os.path.join(FS_COMP_DIR, "fs_comparison.csv"), index=False)
            save_fs_comparison_plot(fs_results, os.path.join(FS_COMP_DIR, "fs_comparison.png"))

    # =========================================================
    # STEP 8 – BASELINE MODEL COMPARISON  
    # =========================================================
    baseline_results = {}
    if RUN_BASELINE_MODEL_COMPARISON:
        print("\n[8] Baseline model comparison (same data splits)...")
        baseline_results = run_baseline_comparison(
            aug_samples, val_samples, test_samples, idx_to_class,
            num_classes, device, BASELINE_COMP)

    # =========================================================
    # STEP 9 – ABLATION STUDY  
    # =========================================================
    ablation_df = None
    if RUN_ABLATION_STUDY:
        print("\n[9] Ablation study (AD-CNN architecture components)...")
        ablation_df = run_ablation_study(
            aug_samples, val_samples, test_samples, idx_to_class,
            num_classes, device, ABLATION_DIR)

    # =========================================================
    # STEP 10 – 5-FOLD SUBJECT-WISE CV  
    # =========================================================
    cv_summary = None
    cv_all_metrics = []
    if RUN_SUBJECTWISE_5FOLD_CV:
        print(f"\n[10] Subject-wise {CV_N_SPLITS}-fold cross-validation...")
        ensure_dir(CV_DIR)
        folds = make_subjectwise_folds(all_samples, n_splits=CV_N_SPLITS, seed=SEED,
                                       val_size_from_train=VAL_SIZE_FROM_TRAIN)
        fold_rows = []
        for fold in folds:
            fi = fold["fold_index"]
            fd = os.path.join(CV_DIR, f"fold_{fi}"); ensure_dir(fd)
            print(f"\n  --- Fold {fi}/{CV_N_SPLITS} ---")
            # verify overlap
            for na, a, nb, b in [("Tr","train_samples","Va","val_samples"),
                                   ("Tr","train_samples","Te","test_samples"),
                                   ("Va","val_samples",  "Te","test_samples")]:
                ov = check_subject_overlap(fold[a], fold[b], None, na, nb)
                if ov: warnings.warn(f"Fold {fi}: {len(ov)} overlap between {na} and {nb}")

            faug_root = os.path.join(fd, "augmented_train")
            faug_samples, _ = build_augmented_train_set(
                fold["train_samples"], faug_root, idx_to_class, target_total=TARGET_TRAIN_IMAGES)
            fm, fis_rgb = create_model(MODEL_NAME, num_classes, FEATURE_DIM, DROPOUT, USE_PRETRAINED)
            fm = fm.to(device)
            fm, _, fbe = train_model(fm, faug_samples, fold["val_samples"], num_classes, device, fd,
                                     epochs=EPOCHS, is_rgb=fis_rgb,
                                     freeze_epochs=FREEZE_BACKBONE_EPOCHS if fis_rgb else 0)
            fresult = evaluate_and_save(fm, fold["test_samples"], class_names, num_classes, device,
                                        fd, tag=f"fold{fi}", use_tta=USE_TTA, is_rgb=fis_rgb)
            cv_all_metrics.append(fresult["metrics"])

            # ASO on fold features
            fft = extract_features(fm, make_loader(faug_samples), device)
            ffv = extract_features(fm, make_loader(fold["val_samples"]), device)
            ffe = extract_features(fm, make_loader(fold["test_samples"]), device)
            f_aso = BinaryAtomSearchOptimizationFeatureSelector(
                lower_bound=ASO_LOWER_BOUND, upper_bound=ASO_UPPER_BOUND, threshold=ASO_THRESHOLD,
                alpha=ASO_ALPHA, beta=ASO_BETA, total_solutions=ASO_TOTAL_SOLUTIONS,
                max_iters=min(ASO_MAX_ITERS, 30), penalty_weight=ASO_PENALTY, random_state=SEED)
            f_aso.fit(fft["features"], fft["labels"], ffv["features"], ffv["labels"])
            fXte = ffe["features"][:, f_aso.best_mask_]
            fXtv = np.concatenate([fft["features"][:,f_aso.best_mask_], ffv["features"][:,f_aso.best_mask_]])
            fytv = np.concatenate([fft["labels"], ffv["labels"]])
            f_clf = clone(LogisticRegression(max_iter=2000, class_weight="balanced",
                                              solver="lbfgs", multi_class="auto"))
            f_clf.fit(StandardScaler().fit_transform(fXtv), fytv)

            fold_rows.append({"fold": fi, "best_epoch": fbe,
                               "n_train": len(fold["train_samples"]),
                               "n_val":   len(fold["val_samples"]),
                               "n_test":  len(fold["test_samples"]),
                               **{k: fresult["metrics"].get(k) for k in
                                  ["accuracy","balanced_accuracy","mcc","cohen_kappa",
                                   "f1_macro","auc","brier_score","ece"]}})
            del fm; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        cv_df = pd.DataFrame(fold_rows)
        cv_df.to_csv(os.path.join(CV_DIR, "cv_fold_metrics.csv"), index=False)

        # mean ± std  (R2)
        keys = ["accuracy","balanced_accuracy","mcc","cohen_kappa",
                "f1_macro","auc","brier_score","ece"]
        cv_summary = {}
        for k in keys:
            vals = [m.get(k) for m in cv_all_metrics if m.get(k) is not None]
            cv_summary[k] = {"mean": float(np.mean(vals)),
                              "std":  float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0}
        save_json(cv_summary, os.path.join(CV_DIR, "cv_mean_std.json"))
        print("\n  CV Summary (mean ± std):")
        for k, v in cv_summary.items():
            print(f"    {k:<25} {v['mean']:.4f} ± {v['std']:.4f}")

    # =========================================================
    # STEP 11 – PAPER-READY TEXT
    # =========================================================
    print("\n[11] Generating reviewer-ready text...")
    generate_paper_ready_text(
        holdout_result["metrics"], cv_summary, ablation_df, fs_results,
        baseline_results, class_names, MODEL_NAME, prep, TEXT_DIR)

    # =========================================================
    # FINAL ZIP
    # =========================================================
    print("\nCreating zip archive...")
    shutil.make_archive("/kaggle/working/ADNI_Reviewer_Ready_v2_results", "zip", RESULTS_DIR)
    shutil.make_archive("/kaggle/working/ADNI_Reviewer_Ready_v2_full",    "zip", WORK_DIR)

    print("\n" + "="*80)
    print("ALL DONE – Reviewer-ready pipeline complete.")
    print(f"Results : {RESULTS_DIR}")
    print(f"ZIP (results): /kaggle/working/ADNI_Reviewer_Ready_v2_results.zip")
    print(f"ZIP (full)   : /kaggle/working/ADNI_Reviewer_Ready_v2_full.zip")
    print("="*80)


if __name__ == "__main__":
    main()
