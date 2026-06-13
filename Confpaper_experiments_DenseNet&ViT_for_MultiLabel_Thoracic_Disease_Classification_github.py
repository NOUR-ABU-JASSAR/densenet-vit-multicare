"""
paper_experiments.py
Reproducible training and evaluation of DenseNet-201 & ViT-B/16 on the MultiCaRe thorax subset.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, hamming_loss
from tf_keras.preprocessing.image import load_img, img_to_array
from tf_keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tf_keras.models import Model
from transformers import TFViTForImageClassification
import warnings
warnings.filterwarnings('ignore')

# ========================== CONFIGURATION ==========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data", "xray_thorax_filtered")

IMAGE_DIR   = os.path.join(DATA_ROOT, "thorax_images")          
LABELS_FILE = os.path.join(DATA_ROOT, "merged_all_cleaned_scanned_per_row_labels.csv")
OUT_DIR     = os.path.join(SCRIPT_DIR, "paper_results")

os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 16
EPOCHS_MAX    = 100
PATIENCE      = 15
LR            = 1e-4
SIGMA         = 0.05
THRESHOLD     = 0.5
SEEDS         = [42, 123, 256]
TEST_SIZE     = 0.2
VAL_SIZE      = 0.1
# ====================================================================


# ---------------------------
# 1. LOAD DATASET
# ---------------------------
df = pd.read_csv(LABELS_FILE)

meta_cols   = ['file', 'patient_id', 'case_text']
label_cols  = [c for c in df.columns if c not in meta_cols]
class_names = label_cols
num_classes = len(class_names)

df['full_path'] = df['file'].apply(lambda f: os.path.join(IMAGE_DIR, f))

print(f"Dataset loaded: {len(df)} images, {num_classes} classes")
print(f"Classes: {class_names}")


# ---------------------------
# 2. PATIENT-LEVEL SPLIT
# ---------------------------
np.random.seed(0)

gss_test = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=0)
train_val_idx, test_idx = next(gss_test.split(df, groups=df['patient_id']))
df_train_val = df.iloc[train_val_idx].copy()
df_test      = df.iloc[test_idx].copy()

gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=0)
train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val['patient_id']))
df_train = df_train_val.iloc[train_idx].copy()
df_val   = df_train_val.iloc[val_idx].copy()

print(f"Patients – Train: {df_train['patient_id'].nunique()}, "
      f"Val: {df_val['patient_id'].nunique()}, "
      f"Test: {df_test['patient_id'].nunique()}")


# ---------------------------
# 3. IMAGE LOADING HELPERS
# ---------------------------
def load_batch_channels_last(paths, target_size=IMG_SIZE):
    """Load images in (N, H, W, C) format — for DenseNet."""
    batch = []
    for p in paths:
        img = load_img(p, target_size=target_size, color_mode='rgb')
        batch.append(img_to_array(img) / 255.0)
    return np.array(batch, dtype=np.float32)


def load_batch_channels_first(paths, target_size=IMG_SIZE):
    """Load images in (N, C, H, W) format — for ViT."""
    batch = []
    for p in paths:
        img = load_img(p, target_size=target_size, color_mode='rgb')
        arr = img_to_array(img) / 255.0          # (H, W, C)
        arr = np.transpose(arr, (2, 0, 1))        # (C, H, W)
        batch.append(arr)
    return np.array(batch, dtype=np.float32)


# ---------------------------
# 4. OVERSAMPLING WITH GAUSSIAN NOISE
# ---------------------------
def augment_train_set(df_train, channels_first=False):
    """
    Returns augmented (images, labels).
    channels_first=False  → shape (N, H, W, C)  for DenseNet
    channels_first=True   → shape (N, C, H, W)  for ViT
    """
    paths = df_train['full_path'].values
    y     = df_train[label_cols].values.astype(np.float32)

    label_counts = y.sum(axis=0)
    max_count    = int(label_counts.max())

    print("  Loading original training images...")
    all_images = []
    for p in paths:
        img = load_img(p, target_size=IMG_SIZE, color_mode='rgb')
        arr = img_to_array(img) / 255.0   # always (H, W, C) first
        all_images.append(arr)

    all_y = list(y)

    print("  Generating augmented samples...")
    for cls_idx, cnt in enumerate(label_counts):
        needed = max_count - int(cnt)
        if needed > 0:
            pos_indices = np.where(y[:, cls_idx] == 1)[0]
            for i in range(needed):
                src_idx = pos_indices[i % len(pos_indices)]
                img_arr = all_images[src_idx] * 255.0
                noise   = SIGMA * np.random.randn(*img_arr.shape) * 255.0
                noisy   = np.clip(img_arr + noise, 0, 255) / 255.0
                all_images.append(noisy.astype(np.float32))
                all_y.append(y[src_idx])

    images_arr = np.array(all_images, dtype=np.float32)  # (N, H, W, C)

    if channels_first:
        # Transpose to (N, C, H, W) for ViT
        images_arr = np.transpose(images_arr, (0, 3, 1, 2))

    print(f"  Augmented dataset: {len(images_arr)} samples, shape {images_arr.shape}")
    return images_arr, np.array(all_y, dtype=np.float32)


# ---------------------------
# 5. BUILD MODELS
# ---------------------------
def build_densenet201(num_classes):
    base = tf.keras.applications.DenseNet201(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)       # channels-last
    )
    x   = GlobalAveragePooling2D()(base.output)
    x   = Dropout(0.3)(x)
    out = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=tf_keras.optimizers.Adam(LR),
        loss=tf_keras.losses.BinaryCrossentropy()
    )
    return model


def build_vitb16(num_classes):
    # ViT expects pixel_values in (N, C, H, W) — channels-first
    model = TFViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=num_classes,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True
    )
    model.compile(
        optimizer=tf_keras.optimizers.Adam(LR),
        loss=tf_keras.losses.BinaryCrossentropy(from_logits=True)
    )
    return model


# ---------------------------
# HELPER: evaluate one model and store results
# ---------------------------
def _evaluate_and_store(probs, y_test, name, seed,
                        results, per_class_data, class_names):
    preds_bin = (probs >= THRESHOLD).astype(int)
    auc_list, acc_list = [], []

    for i, cls in enumerate(class_names):
        try:
            auc = roc_auc_score(y_test[:, i], probs[:, i])
        except ValueError:
            auc = np.nan
        auc_list.append(auc)
        acc_list.append(accuracy_score(y_test[:, i], preds_bin[:, i]))

    mean_auc = np.nanmean(auc_list)
    mean_acc = np.mean(acc_list)
    hloss    = hamming_loss(y_test, preds_bin)

    results['seed'].append(seed)
    results['model'].append(name)
    results['mean_auc'].append(mean_auc)
    results['hamming'].append(hloss)
    results['mean_acc'].append(mean_acc)

    for ci, cls in enumerate(class_names):
        per_class_data.append({
            'seed':  seed,
            'model': name,
            'class': cls,
            'auc':   auc_list[ci],
            'acc':   acc_list[ci]
        })

    print(f"{name} – Seed {seed}: "
          f"Mean AUC={mean_auc:.4f}, "
          f"Hamming={hloss:.4f}, "
          f"Acc={mean_acc:.4f}")


# ---------------------------
# 6. TRAINING LOOP OVER SEEDS (ViT first, then DenseNet)
# ---------------------------
results        = {'seed': [], 'model': [], 'mean_auc': [], 'hamming': [], 'mean_acc': []}
per_class_data = []

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    np.random.seed(seed)
    tf.random.set_seed(seed)

    # ---- ViT-B/16 (channels-first) ----
    print("\nPreparing data for ViT-B/16...")
    X_train_vit, y_train_aug = augment_train_set(df_train, channels_first=True)
    X_val_vit       = load_batch_channels_first(df_val['full_path'].values)
    X_test_vit      = load_batch_channels_first(df_test['full_path'].values)
    y_val           = df_val[label_cols].values.astype(np.float32)
    y_test          = df_test[label_cols].values.astype(np.float32)

    print("\nTraining ViT-B/16...")
    model_vit = build_vitb16(num_classes)
    es_vit    = tf_keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=PATIENCE,
        restore_best_weights=True, verbose=1
    )
    model_vit.fit(
        X_train_vit, y_train_aug,
        validation_data=(X_val_vit, y_val),
        epochs=EPOCHS_MAX, batch_size=BATCH_SIZE,
        callbacks=[es_vit], verbose=1
    )

    # Evaluate ViT
    print("\nEvaluating ViT-B/16...")
    output    = model_vit.predict(X_test_vit, batch_size=BATCH_SIZE)
    logits    = output.logits if hasattr(output, 'logits') else output
    probs_vit = tf.sigmoid(logits).numpy()
    _evaluate_and_store(probs_vit, y_test, 'ViT-B/16', seed,
                        results, per_class_data, class_names)

    # Free ViT memory
    del model_vit, X_train_vit, X_val_vit, X_test_vit
    tf_keras.backend.clear_session()

    # ---- DenseNet-201 (channels-last) ----
    print("\nPreparing data for DenseNet-201...")
    X_train_dn, _ = augment_train_set(df_train, channels_first=False)  
    X_val_dn  = load_batch_channels_last(df_val['full_path'].values)
    X_test_dn = load_batch_channels_last(df_test['full_path'].values)
    # y_val, y_test are already defined

    print("\nTraining DenseNet-201...")
    model_dn = build_densenet201(num_classes)
    es_dn    = tf_keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=PATIENCE,
        restore_best_weights=True, verbose=1
    )
    model_dn.fit(
        X_train_dn, y_train_aug,
        validation_data=(X_val_dn, y_val),
        epochs=EPOCHS_MAX, batch_size=BATCH_SIZE,
        callbacks=[es_dn], verbose=1
    )

    # Evaluate DenseNet
    print("\nEvaluating DenseNet-201...")
    probs_dn  = model_dn.predict(X_test_dn, batch_size=BATCH_SIZE)
    _evaluate_and_store(probs_dn, y_test, 'DenseNet-201', seed,
                        results, per_class_data, class_names)

    # Free DenseNet memory
    del model_dn, X_train_dn, X_val_dn, X_test_dn
    tf_keras.backend.clear_session()


# ---------------------------
# 7. AGGREGATE & SAVE
# ---------------------------
res_df = pd.DataFrame(results)

per_class_df  = pd.DataFrame(per_class_data)
per_class_agg = per_class_df.groupby(['model', 'class']).agg(
    auc_mean=('auc', 'mean'),
    auc_std=('auc',  'std'),
    acc_mean=('acc', 'mean'),
    acc_std=('acc',  'std')
).reset_index()

table_rows = []
for cls in class_names:
    dn  = per_class_agg[(per_class_agg.model == 'DenseNet-201') & (per_class_agg['class'] == cls)]
    vit = per_class_agg[(per_class_agg.model == 'ViT-B/16')     & (per_class_agg['class'] == cls)]
    if len(dn) == 0 or len(vit) == 0:
        continue
    delta = vit.auc_mean.values[0] - dn.auc_mean.values[0]
    table_rows.append({
        'Pathology': cls,
        'DN_AUC':    f"{dn.auc_mean.values[0]:.3f} ± {dn.auc_std.values[0]:.3f}",
        'DN_Acc':    f"{dn.acc_mean.values[0]:.3f} ± {dn.acc_std.values[0]:.3f}",
        'ViT_AUC':   f"{vit.auc_mean.values[0]:.3f} ± {vit.auc_std.values[0]:.3f}",
        'ViT_Acc':   f"{vit.acc_mean.values[0]:.3f} ± {vit.acc_std.values[0]:.3f}",
        'DeltaAUC':  round(delta, 4),
        'ViT_leads': delta > 0
    })

table_df = pd.DataFrame(table_rows)

agg = res_df.groupby('model').agg(
    mean_auc_mean=('mean_auc', 'mean'),
    mean_auc_std=('mean_auc',  'std'),
    hamming_mean=('hamming',   'mean'),
    hamming_std=('hamming',    'std'),
    mean_acc_mean=('mean_acc', 'mean'),
    mean_acc_std=('mean_acc',  'std')
).reset_index()

print("\n" + "="*60)
print("AGGREGATE RESULTS (mean ± std over 3 seeds)")
print("="*60)
print(agg.to_string(index=False))

table_df.to_csv(os.path.join(OUT_DIR, 'per_class_metrics.csv'),  index=False)
agg.to_csv(     os.path.join(OUT_DIR, 'aggregate_metrics.csv'),  index=False)
res_df.to_csv(  os.path.join(OUT_DIR, 'raw_seed_results.csv'),   index=False)

print(f"\nAll results saved to: {OUT_DIR}/")
print("per_class_metrics.csv")
print("aggregate_metrics.csv")
print("raw_seed_results.csv")