# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HAFSSL v2 — HYBRID ATTENTION FEDERATED SEMI-SUPERVISED LEARNING       ║
# ║  Ablation   : MHA → SE (proposed ordering)                             ║
# ║  Environment : Kaggle                                                   ║
# ║  Dataset     : ISCX (5-class)                                          ║
# ║  Clients     : 10                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

!pip install GPUtil psutil kaggle -q

import os, json, shutil, subprocess, datetime, csv, time
import numpy as np
import pandas as pd
from threading import Thread

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import (
    Input, Dense, Reshape, Flatten,
    Convolution1D, MaxPooling1D, UpSampling1D,
    Add, LayerNormalization, GlobalAveragePooling1D,
    MultiHeadAttention, Dropout
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import backend as K
import tensorflow as tf

import psutil
import GPUtil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 12, 'font.family': 'serif',
    'axes.titlesize': 14, 'axes.labelsize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 10, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'axes.grid': True,
    'grid.alpha': 0.3, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2,
})

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

LABELS = [
    'Browsing', 'Email', 'Chat', 'Streaming', 'VoIP',
]

KAGGLE_USERNAME    = 'xcxxxxxx'

DATASET_PATH = (
    '/kaggle/input/datasets/xxxxxxxx/iscx-benchmark/ISCX_5class_each_normalized_cuttedfloefeature.csv'
)

CHECKPOINT_DATASET = 'mha-se-check-hafssl-ae-cnn'


CHECKPOINT_INPUT   = (
    f'/kaggle/input/datasets/{KAGGLE_USERNAME}/{CHECKPOINT_DATASET}/'
)


# Training parameters
labelnum        = 500
latent_dim      = 39
verbose         = 2
batch_size      = 256
epochs          = 5
numOfIterations = 50
numOfClients    = 10

# Non-IID
non_iid_alpha = 0.5

# Attention hyper-parameters
SE_RATIO     = 16
NUM_HEADS    = 4
KEY_DIM      = 32
DROPOUT_RATE = 0.1

FORCE_FRESH_START = False
MODEL_NAME        = "MHA+SE-HAFSSL_10clientsEn+Cn_ISCX5class"

_alpha_str   = str(non_iid_alpha).replace('.', '')
_folder_name = (
    f'MHA+SE-HAFSSL_10clientsEn+Cn_ISCX5class_e{epochs}_b{batch_size}'
    f'_r{numOfIterations}_a{_alpha_str}'
)

WORKING_BASE     = f'/kaggle/working/{_folder_name}'
FIGURE_DIR       = f'{WORKING_BASE}/result/figures'
AEmodelLocation  = f'{WORKING_BASE}/Models/{MODEL_NAME}_AE_{numOfClients}_nodes.keras'
CNNmodelLocation = f'{WORKING_BASE}/Models/{MODEL_NAME}_CNN_{numOfClients}_nodes.keras'
CLIENT_MODEL_DIR = f'{WORKING_BASE}/Models/AECNNmodel'
CHECKPOINT_FILE  = f'{WORKING_BASE}/checkpoint.json'
CONVERGENCE_CSV  = f'{WORKING_BASE}/result/{MODEL_NAME}_convergence.csv'
monitoring_filename  = f'{WORKING_BASE}/result/{MODEL_NAME}_monitoring.csv'
performance_filename = f'{WORKING_BASE}/result/{MODEL_NAME}_performance.csv'

for d in [
    f'{WORKING_BASE}/Models/AECNNmodel',
    f'{WORKING_BASE}/result/figures',
]:
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print(f"  {MODEL_NAME} — ISCX (5 Classes)  [Kaggle]")
print(f"  ✅ HYBRID ATTENTION v2 — 10 CLIENTS")
print(f"  Labels: {LABELS}")
print(f"  α={non_iid_alpha} | epochs={epochs} | "
      f"rounds={numOfIterations} | batch={batch_size} | K={numOfClients}")
print(f"  Output folder : {_folder_name}")
print(f"  FORCE_FRESH_START: {FORCE_FRESH_START}")
print("=" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# KAGGLE API SETUP
# ═════════════════════════════════════════════════════════════════════════════

def setup_kaggle_api():
    try:
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("KAGGLE_KEY")
        os.environ['KAGGLE_API_TOKEN'] = key
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
        with open(kaggle_json, 'w') as f:
            json.dump({"username": KAGGLE_USERNAME, "key": key}, f)
        os.chmod(kaggle_json, 0o600)
        print(f"  ✅ Kaggle API ready — user: {KAGGLE_USERNAME}")
        return True
    except Exception as e:
        print(f"  ⚠️  Kaggle API setup failed: {e}")
        return False


# ═════════════════════════════════════════════════════════════════════════════
# KAGGLE CHECKPOINT STORAGE
# ═════════════════════════════════════════════════════════════════════════════

def push_checkpoint_to_kaggle(round_num):
    try:
        metadata = {
            "title": CHECKPOINT_DATASET,
            "id":    f"{KAGGLE_USERNAME}/{CHECKPOINT_DATASET}",
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(f'{WORKING_BASE}/dataset-metadata.json', 'w') as f:
            json.dump(metadata, f)
        result = subprocess.run(
            ['kaggle', 'datasets', 'version',
             '-p', WORKING_BASE,
             '-m', f'[HAFSSLv2-10K-ISCX5] Round {round_num}/{numOfIterations} '
                   f'e={epochs} b={batch_size} α={non_iid_alpha}',
             '--dir-mode', 'zip'],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            pushed = len(os.listdir(CLIENT_MODEL_DIR)) \
                if os.path.exists(CLIENT_MODEL_DIR) else 0
            print(f"  ✅ Saved to Kaggle Dataset — Round {round_num} "
                  f"({pushed} client model files)")
        else:
            print(f"  ⚠️  Push warning: {result.stderr[:150]}")
    except Exception as e:
        print(f"  ⚠️  Push failed (local copy safe): {str(e)[:150]}")


def pull_checkpoint_from_kaggle():
    if FORCE_FRESH_START:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("  🔄 FORCE_FRESH_START=True → deleted local checkpoint")
        models_dir = f'{WORKING_BASE}/Models'
        if os.path.exists(models_dir):
            shutil.rmtree(models_dir)
            os.makedirs(f'{models_dir}/HAFSSLmodel', exist_ok=True)
        print("  ℹ️  Fresh start — skipping pull")
        return False

    if os.path.exists(CHECKPOINT_FILE):
        print("  ℹ️  Checkpoint found locally — skipping pull")
        return True

    src = os.path.join(CHECKPOINT_INPUT, _folder_name)
    if not os.path.exists(src):
        src = CHECKPOINT_INPUT
        if not os.path.exists(os.path.join(src, 'checkpoint.json')):
            print("  ℹ️  No previous checkpoint — starting fresh")
            return False

    print("  📥 Pulling checkpoint from Kaggle Dataset...")
    try:
        for item in os.listdir(src):
            if item in ('dataset-metadata.json',) or item.endswith('.rtf'):
                continue
            s = os.path.join(src, item)
            d = os.path.join(WORKING_BASE, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        if os.path.exists(CHECKPOINT_FILE):
            print("  ✅ Checkpoint pulled — ready to resume")
            return True
    except Exception as e:
        print(f"  ⚠️  Pull error: {e}")
    print("  ℹ️  Starting fresh")
    return False


# ═════════════════════════════════════════════════════════════════════════════
# CONVERGENCE TRACKER
# ═════════════════════════════════════════════════════════════════════════════

class ConvergenceTracker:
    def __init__(self):
        self.rounds = []; self.global_acc = []; self.global_prec = []
        self.global_recall = []; self.global_f1 = []
        self.per_client_acc = {}

    def record_round(self, round_num, acc, prec, recall, f1, client_accs):
        self.rounds.append(round_num)
        self.global_acc.append(acc)
        self.global_prec.append(prec)
        self.global_recall.append(recall)
        self.global_f1.append(f1)
        self.per_client_acc[round_num] = client_accs

    def save_csv(self, filepath):
        pd.DataFrame({
            'round':     self.rounds,
            'accuracy':  self.global_acc,
            'precision': self.global_prec,
            'recall':    self.global_recall,
            'f1_score':  self.global_f1,
        }).to_csv(filepath, index=False)


# ═════════════════════════════════════════════════════════════════════════════
# CHECKPOINT SAVE / LOAD
# ═════════════════════════════════════════════════════════════════════════════

def save_checkpoint(round_num, tracker):
    checkpoint = {
        'last_completed_round': round_num,
        'model':      MODEL_NAME,
        'epochs':     epochs,
        'batch_size': batch_size,
        'rounds':     numOfIterations,
        'alpha':      non_iid_alpha,
        'tracker': {
            'rounds':         tracker.rounds,
            'global_acc':     tracker.global_acc,
            'global_prec':    tracker.global_prec,
            'global_recall':  tracker.global_recall,
            'global_f1':      tracker.global_f1,
            'per_client_acc': {
                str(k): v for k, v in tracker.per_client_acc.items()
            },
        },
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f"  ✅ Checkpoint saved → Round {round_num}/{numOfIterations}")


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        print("  ℹ️  No checkpoint → starting fresh from Round 1")
        return 1, ConvergenceTracker()

    with open(CHECKPOINT_FILE, 'r') as f:
        cp = json.load(f)
    last = cp['last_completed_round']

    tracker = ConvergenceTracker()
    t = cp['tracker']
    tracker.rounds         = t['rounds']
    tracker.global_acc     = t['global_acc']
    tracker.global_prec    = t['global_prec']
    tracker.global_recall  = t['global_recall']
    tracker.global_f1      = t['global_f1']
    tracker.per_client_acc = {
        int(k): v for k, v in t['per_client_acc'].items()
    }

    if last >= numOfIterations:
        if (not os.path.exists(AEmodelLocation)
                or not os.path.exists(CNNmodelLocation)):
            print("\n  ⚠️  Checkpoint says COMPLETE but model files missing!")
            print("  → Restarting from Round 1...\n")
            os.remove(CHECKPOINT_FILE)
            return 1, ConvergenceTracker()
        print(f"  ✅ Training COMPLETE ({last}/{numOfIterations} rounds)")
        return 'done', tracker

    last_acc = tracker.global_acc[-1] if tracker.global_acc else 0.0
    print(f"\n{'='*60}")
    print(f"  🔄 RESUMING FROM CHECKPOINT")
    print(f"  Last completed round : {last}")
    print(f"  Resuming from round  : {last + 1}")
    print(f"  Remaining rounds     : {numOfIterations - last}")
    print(f"  Last global acc      : {last_acc:.4f}")
    print(f"{'='*60}\n")
    return last + 1, tracker


def check_models_on_disk():
    ok = True
    for path in [AEmodelLocation, CNNmodelLocation]:
        if not os.path.exists(path):
            print(f"  ❌ Missing: {path}")
            ok = False
    return ok


# ═════════════════════════════════════════════════════════════════════════════
# ATTRIBUTE RE-ATTACHMENT
# ═════════════════════════════════════════════════════════════════════════════

def reattach_attributes(ae_model, cnn_model):
    ae_model.encoder  = ae_model.get_layer("encoder")
    cnn_model.encoder = cnn_model.get_layer("encoder")
    cnn_model.cnn     = cnn_model.get_layer("AEcnn")


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_dataset():
    """Load the ISCX 5-class dataset from CSV."""
    print(f"  Loading dataset from: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    dfDS = pd.read_csv(DATASET_PATH)
    X_full = dfDS.iloc[:, 1:len(dfDS.columns)].values
    Y_full = dfDS["label"].values
    num_classes = len(set(Y_full))
    Y_full = tf.keras.utils.to_categorical(Y_full, num_classes)
    print(f"  X_full: {X_full.shape}   Classes: {num_classes}  "
          f"Labels: {LABELS}")
    assert num_classes == len(LABELS), (
        f"CSV has {num_classes} classes but LABELS list has {len(LABELS)}")
    return X_full, Y_full, num_classes


# ═════════════════════════════════════════════════════════════════════════════
# NON-IID PARTITION
# ═════════════════════════════════════════════════════════════════════════════

def split_non_iid(data, labels, num_clients, alpha=0.5, min_samples=5):
    n_classes = len(np.unique(labels))
    N = len(labels)
    min_size = 0
    attempts = 0
    while min_size < min_samples:
        attempts += 1
        if attempts > 100:
            break
        client_indices = [[] for _ in range(num_clients)]
        for c in range(n_classes):
            idx_k = np.where(labels == c)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([
                p * (len(idx_j) < N / num_clients)
                for p, idx_j in zip(proportions, client_indices)
            ])
            proportions = proportions / (proportions.sum() + 1e-12)
            proportions = (
                np.cumsum(proportions) * len(idx_k)
            ).astype(int)[:-1]
            splits = np.split(idx_k, proportions)
            for cid in range(num_clients):
                if cid < len(splits):
                    client_indices[cid].extend(splits[cid].tolist())
        min_size = min(len(idx) for idx in client_indices)
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    return client_indices


# ═════════════════════════════════════════════════════════════════════════════
# SEBlock1D + hybrid_attention_block
# ═════════════════════════════════════════════════════════════════════════════

@register_keras_serializable()
class SEBlock1D(Layer):
    def __init__(self, ratio=SE_RATIO, **kwargs):
        super(SEBlock1D, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.filters      = input_shape[-1]
        self.squeeze      = GlobalAveragePooling1D()
        self.excitation   = Dense(self.filters // self.ratio, activation='relu')
        self.excitation_2 = Dense(self.filters, activation='sigmoid')

    def call(self, inputs):
        squeeze    = self.squeeze(inputs)
        excitation = self.excitation(squeeze)
        excitation = self.excitation_2(excitation)
        excitation = tf.reshape(excitation, [-1, 1, self.filters])
        return inputs * excitation

    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio})
        return config


def hybrid_attention_block(x, num_heads=NUM_HEADS, key_dim=KEY_DIM):
    """Hybrid attention block: multi-head attention followed by squeeze-and-excitation"""
    input_dim = x.shape[-1]
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=DROPOUT_RATE,
    )(x, x)
    attention_output = SEBlock1D()(attention_output)
    x   = Add()([x, attention_output])
    x   = LayerNormalization(epsilon=1e-6)(x)
    ffn = Dense(key_dim * 4, activation="relu")(x)
    ffn = Dropout(DROPOUT_RATE)(ffn)
    ffn = Dense(input_dim)(ffn)
    x   = Add()([x, ffn])
    x   = LayerNormalization(epsilon=1e-6)(x)
    return x


# ═════════════════════════════════════════════════════════════════════════════
# MODEL CREATION
# ═════════════════════════════════════════════════════════════════════════════

def createHAFSSLModel(inp_size, n_classes):
    # ── ENCODER ──────────────────────────────────────────────────────────
    input_e = Input(shape=(inp_size, 1))
    x = Convolution1D(64, 3, padding="same", activation="relu",
                      name='conv_1')(input_e)
    x = Convolution1D(64, 3, padding="same", activation="relu",
                      name='conv_2')(x)
    x = MaxPooling1D(name='maxpool_1')(x)
    x = hybrid_attention_block(x, num_heads=NUM_HEADS, key_dim=KEY_DIM)
    x = Convolution1D(32, 3, padding="same", activation="relu",
                      name='conv_3')(x)
    x = Convolution1D(32, 3, padding="same", activation="relu",
                      name='conv_4')(x)
    s_shape = x.shape[1:]
    latent  = Dense(latent_dim, activation="relu")(Flatten()(x))
    encoder = Model(input_e, latent, name="encoder")
    encoder.summary()

    # ── DECODER ──────────────────────────────────────────────────────────
    input_d = Input(shape=(latent_dim,))
    x = Reshape((int(s_shape[0]), int(s_shape[1])))(
        Dense(int(np.prod(s_shape)))(input_d))
    x = Convolution1D(32, 3, padding="same", activation="relu",
                      name='conv_5')(x)
    x = Convolution1D(32, 3, padding="same", activation="relu",
                      name='conv_6')(x)
    x = UpSampling1D(2, name='upsampling_1d_2')(x)
    x = Convolution1D(64, 3, padding="same", activation="relu",
                      name='conv_7')(x)
    output = Convolution1D(1, 3, padding="same", activation="relu",
                           name='conv_8')(x)
    decoder = Model(input_d, output, name="decoder")
    decoder.summary()

    # ── CNN CLASSIFIER ────────────────────────────────────────────────────
    input_c = Input(shape=(latent_dim,))
    x = Reshape((int(s_shape[0]), int(s_shape[1])))(
        Dense(int(np.prod(s_shape)))(input_c))
    x = Convolution1D(64, 3, padding="same", activation="relu")(x)
    x = Convolution1D(64, 3, padding="same", activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = hybrid_attention_block(x, num_heads=NUM_HEADS, key_dim=KEY_DIM // 2)
    x = Convolution1D(128, 3, padding="same", activation="relu")(x)
    x = Convolution1D(128, 3, padding="same", activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(n_classes, activation="softmax")(x)  # ← 5 output classes
    cnn = Model(input_c, x, name="AEcnn")
    cnn.summary()

    autoencoder = Model(input_e, decoder(encoder(input_e)))
    autoencoder.encoder = encoder

    classificationmodel = Model(input_e, cnn(encoder(input_e)))
    classificationmodel.encoder = encoder
    classificationmodel.cnn = cnn

    return autoencoder, classificationmodel


# ═════════════════════════════════════════════════════════════════════════════
# LABELED / UNLABELED SPLIT
# ═════════════════════════════════════════════════════════════════════════════

def splitLabel(x_train, y_train, labels=LABELS):
    if len(x_train) == 0:
        empty_x = np.empty((0, x_train.shape[1] if x_train.ndim > 1 else 1))
        empty_y = np.empty((0, len(labels)))
        return empty_x, empty_y, empty_x

    label_indices = {
        label: np.where(y_train.argmax(axis=1) == label)[0]
        for label in range(len(labels))
    }
    samples_per_class = labelnum // len(labels)
    x_labeled, y_labeled, all_idx = [], [], []

    for label in range(len(labels)):
        if len(label_indices[label]) == 0:
            continue
        sel = (
            label_indices[label]
            if len(label_indices[label]) < samples_per_class
            else np.random.choice(
                label_indices[label], samples_per_class, replace=False)
        )
        x_labeled.append(x_train[sel])
        y_labeled.append(y_train[sel])
        all_idx.extend(sel.tolist())

    if len(x_labeled) == 0:
        print("  ⚠️  Client has no labeled samples — using all as labeled")
        x_unlabeled = np.empty((0,) + x_train.shape[1:])
        return x_train, y_train, x_unlabeled

    x_unlabeled = x_train[
        np.setdiff1d(np.arange(len(x_train)), all_idx)
    ]
    return (
        np.concatenate(x_labeled),
        np.concatenate(y_labeled),
        x_unlabeled,
    )


# ═════════════════════════════════════════════════════════════════════════════
# CLIENT MODEL CLONING
# ═════════════════════════════════════════════════════════════════════════════

def updateClientsModels():
    global clientsAEModelList, clientsCNNModelist
    clientsAEModelList = []
    clientsCNNModelist = []
    for _ in range(numOfClients):
        ae = tf.keras.models.clone_model(originautoencoder)
        ae.set_weights(originautoencoder.get_weights())
        ae.encoder = ae.get_layer("encoder")
        clientsAEModelList.append(ae)

        cnn = tf.keras.models.clone_model(originclassificationmodel)
        cnn.set_weights(originclassificationmodel.get_weights())
        cnn.encoder = cnn.get_layer("encoder")
        cnn.cnn     = cnn.get_layer("AEcnn")
        clientsCNNModelist.append(cnn)


# ═════════════════════════════════════════════════════════════════════════════
# WEIGHTED FEDAVG AGGREGATOR
# ═════════════════════════════════════════════════════════════════════════════

class WeightedFedAvgAggregator:
    def __init__(self):
        self.weighted_sum = None
        self.total_weight = 0

    def clear(self):
        self.weighted_sum = None
        self.total_weight = 0

    def add_client_weights(self, w, n):
        self.weighted_sum = (
            [x.copy() * n for x in w]
            if self.weighted_sum is None
            else [s + x * n for s, x in zip(self.weighted_sum, w)]
        )
        self.total_weight += n

    def aggregate(self):
        if self.weighted_sum is None:
            raise ValueError("No weights added yet.")
        return [w / self.total_weight for w in self.weighted_sum]


# ═════════════════════════════════════════════════════════════════════════════
# MODEL PROFILING
# ═════════════════════════════════════════════════════════════════════════════

def profile_model(model, input_shape, name="Model"):
    total_params = model.count_params()
    trainable = sum(
        tf.keras.backend.count_params(w)
        for w in model.trainable_weights
    )
    dummy = np.random.randn(1, *input_shape).astype(np.float32)
    _ = model.predict(dummy, verbose=0)
    times = []
    for _ in range(50):
        t = time.time()
        _ = model.predict(dummy, verbose=0)
        times.append(time.time() - t)
    avg_time = np.mean(times) * 1000
    model.save("/tmp/_temp_model.keras")
    size_mb = os.path.getsize("/tmp/_temp_model.keras") / (1024 * 1024)
    print(f"\n  {name}: {total_params:,} params | "
          f"{avg_time:.1f} ms | {size_mb:.2f} MB")
    return {
        'name': name,
        'total_params': total_params,
        'trainable_params': trainable,
        'inference_ms': avg_time,
        'model_size_mb': size_mb,
    }


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════

def plot_non_iid_distribution(yClientsList, labels, save_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    dist = np.zeros((len(yClientsList), len(labels)))
    for c in range(len(yClientsList)):
        cl = np.argmax(yClientsList[c], axis=1)
        for cls in range(len(labels)):
            dist[c, cls] = np.sum(cl == cls)
    dist_norm = dist / (dist.sum(axis=1, keepdims=True) + 1e-12)
    sns.heatmap(
        dist_norm, annot=True, fmt='.2f', cmap='YlOrRd',
        xticklabels=labels,
        yticklabels=[f'Client {i}' for i in range(len(yClientsList))],
        ax=ax, annot_kws={"fontsize": 10})
    ax.set_title(
        f'Non-IID Distribution — {MODEL_NAME} (α={non_iid_alpha})',
        fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")


def plot_non_iid_bar(yClientsList, labels, save_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(yClientsList))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for cls in range(len(labels)):
        counts = [np.sum(np.argmax(yClientsList[c], axis=1) == cls)
                  for c in range(len(yClientsList))]
        ax.bar([f'Client {i}' for i in range(len(yClientsList))],
               counts, bottom=bottom, label=labels[cls],
               color=colors[cls], edgecolor='white')
        bottom += np.array(counts)
    ax.set_ylabel('Samples')
    ax.set_title(
        f'Non-IID Sample Distribution (α={non_iid_alpha}, K={numOfClients})',
        fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=1)
    ax.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")


def plot_convergence(tracker, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tracker.rounds, tracker.global_acc,    'o-',
            color='#1976D2', markersize=5, label='Accuracy')
    ax.plot(tracker.rounds, tracker.global_f1,     's-',
            color='#E53935', markersize=5, label='F1')
    ax.plot(tracker.rounds, tracker.global_prec,   '^-',
            color='#43A047', markersize=4, alpha=0.7, label='Precision')
    ax.plot(tracker.rounds, tracker.global_recall, 'v-',
            color='#FF9800', markersize=4, alpha=0.7, label='Recall')
    ax.fill_between(tracker.rounds, tracker.global_acc,
                    alpha=0.1, color='#1976D2')
    ax.set_xlabel('FL Round'); ax.set_ylabel('Score')
    ax.set_title(
        f'{MODEL_NAME} — Convergence '
        f'(α={non_iid_alpha}, e={epochs}, K={numOfClients})',
        fontweight='bold')
    ax.set_ylim([0, 1.05]); ax.legend(loc='lower right')
    if tracker.global_acc:
        ax.annotate(
            f'{tracker.global_acc[-1]:.3f}',
            xy=(tracker.rounds[-1], tracker.global_acc[-1]),
            xytext=(-60, 10), textcoords='offset points',
            fontsize=11, fontweight='bold', color='#1976D2',
            arrowprops=dict(arrowstyle='->', color='#1976D2'))
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")


def plot_per_client(tracker, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, numOfClients))
    for cid in range(numOfClients):
        accs = [tracker.per_client_acc[r][cid]
                for r in tracker.rounds
                if cid < len(tracker.per_client_acc[r])]
        rounds_for_client = tracker.rounds[:len(accs)]
        ax.plot(rounds_for_client, accs, '-',
                color=colors[cid], linewidth=1.5, alpha=0.8,
                label=f'Client {cid}')
    ax.plot(tracker.rounds, tracker.global_acc, 'k--',
            linewidth=2.5, label='Global acc', zorder=10)
    ax.set_xlabel('FL Round'); ax.set_ylabel('Accuracy')
    ax.set_title(
        f'{MODEL_NAME} — Per-Client Accuracy '
        f'(K={numOfClients})', fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")


def plot_final_bars(acc_l, prec_l, rec_l, f1_l, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    client_labels = [f'Client {i}' for i in range(len(acc_l))]

    for ax, values, title in zip(
        axes.flat,
        [acc_l, prec_l, rec_l, f1_l],
        ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    ):
        bar_colors = ['#E53935' if v < 0.70 else '#FFA726'
                      if v < 0.85 else '#43A047' for v in values]
        bars = ax.bar(client_labels, values,
                      color=bar_colors, edgecolor='white', width=0.6)
        ax.axhline(y=np.mean(values), color='navy', linestyle='--',
                   linewidth=1.5, alpha=0.8,
                   label=f'Mean = {np.mean(values):.3f}')
        ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)
        ax.set_ylim([0, 1.15])
        ax.set_title(title, fontweight='bold')
        ax.tick_params(axis='x', rotation=25, labelsize=9)
        ax.legend(fontsize=9)

    plt.suptitle(
        f'{MODEL_NAME} — Final Metrics per Client '
        f'(α={non_iid_alpha}, K={numOfClients})',
        fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")


def plot_confusion(y_true, y_pred, labels, save_path):
    cm   = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm,   annot=True, fmt='d',    square=True, cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=a1,
                annot_kws={"fontsize": 10})
    sns.heatmap(cm_n, annot=True, fmt='.0%', square=True, cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels, ax=a2,
                annot_kws={"fontsize": 10})
    a1.set_title('Counts', fontweight='bold')
    a1.tick_params(axis='x', rotation=20)
    a2.set_title('Normalized', fontweight='bold')
    a2.tick_params(axis='x', rotation=20)
    plt.suptitle(
        f'{MODEL_NAME} — Confusion Matrix (α={non_iid_alpha}, K={numOfClients})',
        fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")


def plot_complexity(profiles, save_path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    names = [p['name'] for p in profiles]
    cols  = plt.cm.Set2(np.linspace(0, 1, len(profiles)))
    b1 = a1.bar(names, [p['total_params']/1000 for p in profiles], color=cols)
    a1.set_ylabel('Params (K)'); a1.set_title('Model Complexity', fontweight='bold')
    a1.bar_label(b1, fmt='%.0fK', fontsize=9, padding=2)
    b2 = a2.bar(names, [p['inference_ms'] for p in profiles], color=cols)
    a2.set_ylabel('Inference (ms)'); a2.set_title('Inference Latency', fontweight='bold')
    a2.bar_label(b2, fmt='%.1fms', fontsize=9, padding=2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")


def plot_dashboard(tracker, acc_list, recon_err, profiles, save_path):
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(tracker.rounds, tracker.global_acc, 'o-',
             color='#1976D2', markersize=4, label='Acc')
    ax1.plot(tracker.rounds, tracker.global_f1,  's-',
             color='#E53935', markersize=4, label='F1')
    ax1.set_xlabel('Round'); ax1.set_ylabel('Score')
    ax1.set_title('Convergence', fontweight='bold')
    ax1.set_ylim([0, 1.05]); ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    acc_arr = np.array(acc_list)
    bar_colors = ['#E53935' if a < 0.70 else '#FFA726'
                  if a < 0.85 else '#43A047' for a in acc_arr]
    ax2.bar([f'C{i}' for i in range(len(acc_arr))], acc_arr,
            color=bar_colors, edgecolor='white')
    ax2.axhline(y=np.mean(acc_arr), color='navy', linestyle='--',
                alpha=0.8, label=f'Mean={np.mean(acc_arr):.3f}')
    ax2.set_xlabel('Client'); ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Final Accuracy per Client (K={numOfClients})',
                  fontweight='bold')
    ax2.set_ylim([0, 1.15]); ax2.legend(fontsize=9)
    ax2.tick_params(axis='x', rotation=20)

    ax3 = fig.add_subplot(gs[0, 2]); ax3.axis('off')
    r90 = next(
        (r for r, a in zip(tracker.rounds, tracker.global_acc) if a >= 0.90),
        'N/A')
    txt = (
        f"{MODEL_NAME}\n"
        f"ISCX | 5 Classes | Kaggle\n"
        f"α={non_iid_alpha} | K={numOfClients}\n"
        f"Labels={labelnum} ({labelnum//len(LABELS)}/class)\n"
        f"Rounds={numOfIterations} | epochs={epochs}\n"
        f"batch={batch_size} | dropout={DROPOUT_RATE}\n\n"
        f"Final Acc: {tracker.global_acc[-1]:.4f}\n"
        f"Final F1:  {tracker.global_f1[-1]:.4f}\n"
        f"Recon Err: {recon_err:.4f}\n\n"
        f"Max: {max(acc_list):.4f}  Min: {min(acc_list):.4f}\n"
        f"Std: {np.std(acc_list):.4f}\n\n"
        f"Params: {profiles[0]['total_params']:,}\n"
        f"Rounds→90%: {r90}"
    )
    ax3.text(0.5, 0.5, txt, ha='center', va='center', fontsize=9,
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='#E3F2FD',
                       edgecolor='#1976D2', linewidth=2))
    ax3.set_title('Summary', fontweight='bold')

    ax4 = fig.add_subplot(gs[1, 0])
    vars_ = [np.std(tracker.per_client_acc[r]) for r in tracker.rounds]
    ax4.plot(tracker.rounds, vars_, '-', color='#9C27B0')
    ax4.fill_between(tracker.rounds, vars_, alpha=0.2, color='#9C27B0')
    ax4.axhline(y=np.mean(vars_), color='red', linestyle='--',
                alpha=0.7, label=f'Mean={np.mean(vars_):.3f}')
    ax4.legend(fontsize=9)
    ax4.set_xlabel('Round'); ax4.set_ylabel('Std Dev')
    ax4.set_title('Client Variance per Round', fontweight='bold')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar([p['name'] for p in profiles],
            [p['model_size_mb'] for p in profiles],
            color='#FF9800', edgecolor='gray')
    ax5.set_ylabel('Size (MB)'); ax5.set_title('Model Size', fontweight='bold')

    ax6 = fig.add_subplot(gs[1, 2]); ax6.axis('off')
    bpr = sum(p['model_size_mb'] for p in profiles) * numOfClients * 2
    ctxt = (
        f"Communication Cost\n\n"
        f"Per round: {bpr:.1f} MB\n"
        f"({numOfClients} clients × up+down)\n\n"
        f"Rounds to 90%: {r90}\n"
        f"Total rounds: {numOfIterations}\n"
        f"Total comm: {bpr*numOfIterations:.0f} MB\n\n"
        f"Dataset: ISCX\n"
        f"Classes: {len(LABELS)} | α={non_iid_alpha}\n"
        f"K={numOfClients}"
    )
    ax6.text(0.5, 0.5, ctxt, ha='center', va='center', fontsize=10,
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='#E8F5E9',
                       edgecolor='#43A047', linewidth=2))
    ax6.set_title('Communication', fontweight='bold')

    plt.suptitle(
        f'{MODEL_NAME} — Dashboard '
        f'(α={non_iid_alpha}, K={numOfClients}, e={epochs})',
        fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MONITORING THREAD
# ═════════════════════════════════════════════════════════════════════════════

monitorheaders = [
    'stage', 'iterationNo', 'clientID',
    'avg_GPU_mem', 'avg_GPU_load',
    'avg_Memory_used', 'avg_cpu_used', 'used_time(us)',
]
Globalmonitordirct = {h: 0 for h in monitorheaders}
Globalmonitordirct['stage'] = ''
Globalmonitordirctrows = []

performanceheaders = [
    'stage', 'iterationNo', 'clientID',
    'train_acc', 'val_acc', 'test_acc', 'classification_report',
]
performancerdirct = {h: 0 for h in performanceheaders}
performancerdirct['stage'] = ''
performancerdirct['classification_report'] = ''
performancerdirctros = []


class Monitor(Thread):
    def __init__(self, delay, stage, iterationNo, clientID, process):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay; self.stage = stage
        self.iterationNo = iterationNo; self.clientID = clientID
        self.process = process
        self.gpu_mem_list = []; self.gpu_load_list = []
        self.used_mem_list = []; self.cpu_load_list = []
        self.start()

    def run(self):
        st = datetime.datetime.now()
        while not self.stopped:
            try:
                Gpus = GPUtil.getGPUs()
                if Gpus:
                    self.gpu_mem_list.append(Gpus[0].memoryUtil * 100)
                    self.gpu_load_list.append(Gpus[0].load * 100)
                else:
                    self.gpu_mem_list.append(0)
                    self.gpu_load_list.append(0)
            except Exception:
                self.gpu_mem_list.append(0)
                self.gpu_load_list.append(0)
            self.used_mem_list.append(
                self.process.memory_percent(memtype="uss"))
            self.cpu_load_list.append(
                self.process.cpu_percent(interval=1))
            time.sleep(self.delay)
        et = datetime.datetime.now()
        Globalmonitordirct.update({
            'stage':           self.stage,
            'iterationNo':     self.iterationNo,
            'clientID':        self.clientID,
            'avg_GPU_mem':     np.mean(self.gpu_mem_list),
            'avg_GPU_load':    np.mean(self.gpu_load_list),
            'avg_Memory_used': np.mean(self.used_mem_list),
            'avg_cpu_used':    np.mean(self.cpu_load_list),
            'used_time(us)':   (et - st).microseconds * 23,
        })
        Globalmonitordirctrows.append(Globalmonitordirct.copy())

    def stop(self):
        self.stopped = True


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    setup_kaggle_api()
    pull_checkpoint_from_kaggle()
    start_round, tracker = load_checkpoint()

    # ═══════════════════════════════════════════════════════════════════════
    # DONE PATH
    # ═══════════════════════════════════════════════════════════════════════
    if start_round == 'done':
        print("  Training already complete — running final evaluation + plots.")
        X_full, Y_full, num_classes = load_dataset()
        xServer, xClients, yServer, yClients = train_test_split(
            X_full, Y_full, test_size=0.90, random_state=523)
        xServer = np.expand_dims(xServer, axis=2)
        originautoencoder         = load_model(AEmodelLocation)
        originclassificationmodel = load_model(CNNmodelLocation)
        reattach_attributes(originautoencoder, originclassificationmodel)
        profiles = [
            profile_model(originautoencoder,
                          (xServer.shape[1], 1), "Autoencoder"),
            profile_model(originclassificationmodel,
                          (xServer.shape[1], 1), "Classifier"),
        ]
        start_time = time.time()

    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING PATH
    # ═══════════════════════════════════════════════════════════════════════
    else:
        X_full, Y_full, num_classes = load_dataset()
        assert num_classes == len(LABELS), (
            f"Expected {len(LABELS)} classes, got {num_classes}")

        xServer, xClients, yServer, yClients = train_test_split(
            X_full, Y_full, test_size=0.90, random_state=523)
        xServer = np.expand_dims(xServer, axis=2)
        inp_size = xServer.shape[1]
        print(f"  xServer:{xServer.shape}  "
              f"xClients:{xClients.shape}  Classes:{num_classes}")

        if start_round == 1:
            print(f"\n  Creating new model "
                  f"(K={numOfClients}, dropout={DROPOUT_RATE}, "
                  f"classes={num_classes})...")
            originautoencoder, originclassificationmodel = \
                createHAFSSLModel(inp_size, num_classes)
            profiles = [
                profile_model(originautoencoder,
                              (inp_size, 1), "Autoencoder"),
                profile_model(originclassificationmodel,
                              (inp_size, 1), "Classifier"),
            ]
            clientsAEModelList = []
            clientsCNNModelist = []
            updateClientsModels()
            originautoencoder.save(AEmodelLocation)
            originclassificationmodel.save(CNNmodelLocation)
            print("  ✅ New models saved")
        else:
            if not check_models_on_disk():
                print("  ⚠️  Model files missing — restarting from Round 1...")
                os.remove(CHECKPOINT_FILE)
                originautoencoder, originclassificationmodel = \
                    createHAFSSLModel(inp_size, num_classes)
                profiles = [
                    profile_model(originautoencoder,
                                  (inp_size, 1), "Autoencoder"),
                    profile_model(originclassificationmodel,
                                  (inp_size, 1), "Classifier"),
                ]
                clientsAEModelList = []
                clientsCNNModelist = []
                updateClientsModels()
                originautoencoder.save(AEmodelLocation)
                originclassificationmodel.save(CNNmodelLocation)
                tracker = ConvergenceTracker()
                start_round = 1
            else:
                print(f"  Loading models (resuming from Round {start_round})...")
                originautoencoder         = load_model(AEmodelLocation)
                originclassificationmodel = load_model(CNNmodelLocation)
                reattach_attributes(originautoencoder,
                                    originclassificationmodel)
                profiles = [
                    profile_model(originautoencoder,
                                  (inp_size, 1), "Autoencoder"),
                    profile_model(originclassificationmodel,
                                  (inp_size, 1), "Classifier"),
                ]
                clientsAEModelList = []
                clientsCNNModelist = []
                updateClientsModels()
                print("  ✅ Models loaded + attributes re-attached")

        print(f"\n  Partitioning data (seed=42, α={non_iid_alpha})...")
        np.random.seed(42)
        yClients_int = np.argmax(yClients, axis=1)
        client_indices = split_non_iid(
            xClients, yClients_int,
            numOfClients, alpha=non_iid_alpha, min_samples=5)
        xClientsList = [xClients[client_indices[c]]
                        for c in range(numOfClients)]
        yClientsList = [yClients[client_indices[c]]
                        for c in range(numOfClients)]

        if start_round > 1:
            print("  Loading per-client models...")
            clientsAEModelList = []
            clientsCNNModelist = []
            for cid in range(numOfClients):
                ae_path  = f"{CLIENT_MODEL_DIR}/AE_node_{cid}.keras"
                cnn_path = f"{CLIENT_MODEL_DIR}/CNN_node_{cid}.keras"
                if os.path.exists(ae_path) and os.path.exists(cnn_path):
                    ae  = load_model(ae_path)
                    cnn = load_model(cnn_path)
                    ae.encoder  = ae.get_layer("encoder")
                    cnn.encoder = cnn.get_layer("encoder")
                    cnn.cnn     = cnn.get_layer("AEcnn")
                    clientsAEModelList.append(ae)
                    clientsCNNModelist.append(cnn)
                    print(f"    ✅ Client {cid} loaded")
                else:
                    print(f"    ⚠️  Client {cid} missing — using global")
                    ae = tf.keras.models.clone_model(originautoencoder)
                    ae.set_weights(originautoencoder.get_weights())
                    ae.encoder = ae.get_layer("encoder")
                    clientsAEModelList.append(ae)
                    cnn = tf.keras.models.clone_model(
                        originclassificationmodel)
                    cnn.set_weights(
                        originclassificationmodel.get_weights())
                    cnn.encoder = cnn.get_layer("encoder")
                    cnn.cnn     = cnn.get_layer("AEcnn")
                    clientsCNNModelist.append(cnn)
        else:
            clientsAEModelList = []
            clientsCNNModelist = []
            for cid in range(numOfClients):
                ae  = load_model(AEmodelLocation)
                cnn = load_model(CNNmodelLocation)
                ae.encoder  = ae.get_layer("encoder")
                cnn.encoder = cnn.get_layer("encoder")
                cnn.cnn     = cnn.get_layer("AEcnn")
                clientsAEModelList.append(ae)
                clientsCNNModelist.append(cnn)

        if start_round == 1:
            for cid in range(numOfClients):
                cl = np.argmax(yClientsList[cid], axis=1)
                n  = len(client_indices[cid])
                print(f"\n  Client {cid}: {n} samples")
                for i, label in enumerate(LABELS):
                    c   = np.sum(cl == i)
                    pct = 100 * c / n if n > 0 else 0
                    if c > 0:
                        print(f"    {label}: {c} ({pct:.1f}%)")
            plot_non_iid_distribution(
                yClientsList, LABELS,
                f"{FIGURE_DIR}/{MODEL_NAME}_01_nonIID_heatmap.png")
            plot_non_iid_bar(
                yClientsList, LABELS,
                f"{FIGURE_DIR}/{MODEL_NAME}_02_nonIID_bars.png")

        xClientsListLabel   = []
        xClientsListUnLabel = []
        yClientsListLabel   = []
        client_sample_sizes = []
        for cid in range(numOfClients):
            xl, yl, xu = splitLabel(xClientsList[cid], yClientsList[cid])
            xl = np.expand_dims(xl, axis=2)
            xu = np.expand_dims(xu, axis=2)
            xClientsListLabel.append(xl)
            xClientsListUnLabel.append(xu)
            yClientsListLabel.append(yl)
            client_sample_sizes.append(len(xl) + len(xu))
        print(f"\n  Client sample sizes: {client_sample_sizes}")
        print(f"  Total: {sum(client_sample_sizes)}")

        agg_ae  = WeightedFedAvgAggregator()
        agg_cnn = WeightedFedAvgAggregator()
        start_time = time.time()
        process    = psutil.Process(os.getpid())

        print(f"\n{'='*60}")
        print(f"  {MODEL_NAME}  |  Round {start_round}/{numOfIterations}")
        print(f"  α={non_iid_alpha} | K={numOfClients} | "
              f"epochs={epochs} | batch={batch_size} | dropout={DROPOUT_RATE}")
        print(f"  Dataset: ISCX ({len(LABELS)} classes)")
        print(f"  Checkpoint → {KAGGLE_USERNAME}/{CHECKPOINT_DATASET}")
        print(f"{'='*60}")

        # ══════════════════════════════════════════════════════════════════
        # TRAINING LOOP
        # ══════════════════════════════════════════════════════════════════
        for iterationNo in range(start_round, numOfIterations + 1):
            print(f"\n{'='*60}\n  Round {iterationNo}/{numOfIterations}"
                  f"\n{'='*60}")
            round_client_accs = []
            agg_ae.clear()
            agg_cnn.clear()

            for clientID in range(numOfClients):
                print(f"\n  ── Client {clientID} ──")

                if client_sample_sizes[clientID] < 50:
                    print(f"  ⚠️  Client {clientID} skipped — "
                          f"only {client_sample_sizes[clientID]} samples")
                    round_client_accs.append(0.0)
                    continue

                monitor     = Monitor(1, "HAFSSLv2-ISCX5 training",
                                      iterationNo, clientID, process)
                subAEmodel  = originautoencoder
                subCNNmodel = originclassificationmodel
                subAEmodel.set_weights(
                    clientsAEModelList[clientID].get_weights())
                subCNNmodel.set_weights(
                    clientsCNNModelist[clientID].get_weights())

                subAEmodel.encoder  = subAEmodel.get_layer("encoder")
                subCNNmodel.encoder = subCNNmodel.get_layer("encoder")
                subCNNmodel.cnn     = subCNNmodel.get_layer("AEcnn")

                # ── Stage 1: AE unsupervised ──────────────────────────────
                if len(xClientsListUnLabel[clientID]) >= 50:
                    subAEmodel.compile(
                        loss='mse', optimizer='adam', metrics=['mse'])
                    subAEmodel.fit(
                        xClientsListUnLabel[clientID],
                        xClientsListUnLabel[clientID],
                        epochs=epochs, shuffle=True,
                        validation_data=(xServer, xServer),
                        verbose=verbose)
                    subCNNmodel.encoder.set_weights(
                        subAEmodel.encoder.get_weights())
                else:
                    print(f"  ⚠️  Client {clientID} — no unlabeled data, "
                          f"skipping AE stage")

                # ── Stage 1.5: supervised encoder fine-tune ───────────────
                for layer in subCNNmodel.cnn.layers:
                    layer.trainable = False
                subCNNmodel.compile(
                    loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    metrics=['accuracy'])
                subCNNmodel.fit(
                    xClientsListLabel[clientID],
                    yClientsListLabel[clientID],
                    epochs=2, batch_size=batch_size,
                    shuffle=True, verbose=0)
                for layer in subCNNmodel.cnn.layers:
                    layer.trainable = True

                # ── Stage 2: CNN supervised full training ─────────────────
                subCNNmodel.compile(
                    loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
                history = subCNNmodel.fit(
                    xClientsListLabel[clientID],
                    yClientsListLabel[clientID],
                    epochs=epochs, batch_size=batch_size,
                    shuffle=True,
                    validation_data=(xServer, yServer),
                    verbose=verbose)
                monitor.stop()

                agg_ae.add_client_weights(
                    subAEmodel.get_weights(),
                    client_sample_sizes[clientID])
                agg_cnn.add_client_weights(
                    subCNNmodel.get_weights(),
                    client_sample_sizes[clientID])

                y_pr   = subCNNmodel.predict(xServer, batch_size=300)
                report = classification_report(
                    yServer.argmax(1), y_pr.argmax(1),
                    target_names=LABELS,
                    zero_division=1, output_dict=True)
                round_client_accs.append(report['accuracy'])
                print(f"  Client {clientID} — "
                      f"Acc:{report['accuracy']:.4f} | "
                      f"F1:{report['weighted avg']['f1-score']:.4f}")

                performancerdirct.update({
                    'stage':       'HAFSSLv2-ISCX5 training',
                    'iterationNo': iterationNo,
                    'clientID':    clientID,
                    'train_acc':   history.history["accuracy"][-1],
                    'val_acc':     history.history["val_accuracy"][-1],
                    'test_acc':    report['accuracy'],
                    'classification_report': report,
                })
                performancerdirctros.append(performancerdirct.copy())

                subCNNmodel.save(
                    f"{CLIENT_MODEL_DIR}/CNN_node_{clientID}.keras")
                subAEmodel.save(
                    f"{CLIENT_MODEL_DIR}/AE_node_{clientID}.keras")
                print(f"  ✅ Client {clientID} models saved")

            # ── Aggregation ───────────────────────────────────────────────
            print("\n  ── Aggregation ──")
            originautoencoder.set_weights(agg_ae.aggregate())
            originautoencoder.save(AEmodelLocation)
            originclassificationmodel.set_weights(agg_cnn.aggregate())
            originclassificationmodel.save(CNNmodelLocation)
            reattach_attributes(originautoencoder, originclassificationmodel)
            print(f"  ✅ Global models saved (Round {iterationNo})")

            y_gpr = originclassificationmodel.predict(
                xServer, batch_size=300)
            gr = classification_report(
                yServer.argmax(1), y_gpr.argmax(1),
                target_names=LABELS, zero_division=1, output_dict=True)
            tracker.record_round(
                iterationNo,
                gr['accuracy'],
                gr['weighted avg']['precision'],
                gr['weighted avg']['recall'],
                gr['weighted avg']['f1-score'],
                round_client_accs)
            print(f"\n  Round {iterationNo} Global — "
                  f"Acc:{gr['accuracy']:.4f} | "
                  f"F1:{gr['weighted avg']['f1-score']:.4f}")

            save_checkpoint(iterationNo, tracker)
            tracker.save_csv(CONVERGENCE_CSV)
            push_checkpoint_to_kaggle(iterationNo)
            updateClientsModels()

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL EVALUATION
    # ═══════════════════════════════════════════════════════════════════════
    total_time = time.time() - start_time
    print(f"\n{'='*60}\n  FINAL EVALUATION — {MODEL_NAME}"
          f"\n  Time: {total_time/60:.1f} min\n{'='*60}")

    fa, fp, fr, ff = [], [], [], []
    for cid in range(numOfClients):
        nm = originclassificationmodel
        cnn_path = f"{CLIENT_MODEL_DIR}/CNN_node_{cid}.keras"
        if os.path.exists(cnn_path):
            nm.set_weights(load_model(cnn_path).get_weights())
        else:
            print(f"  ⚠️  Client {cid} — file missing, using global weights")
        yp = nm.predict(xServer, batch_size=100)
        r  = classification_report(
            yServer.argmax(1), yp.argmax(1),
            target_names=LABELS, zero_division=1, output_dict=True)
        fa.append(r['accuracy'])
        fp.append(r['weighted avg']['precision'])
        fr.append(r['weighted avg']['recall'])
        ff.append(r['weighted avg']['f1-score'])
        print(f"  Client {cid} — Acc:{r['accuracy']:.4f} | "
              f"F1:{r['weighted avg']['f1-score']:.4f}")
        performancerdirct.update({
            'stage':    'Global validation after training',
            'clientID': cid,
            'test_acc': r['accuracy'],
            'classification_report': r,
        })
        performancerdirctros.append(performancerdirct.copy())

    recon_err = np.mean(
        np.square(xServer - originautoencoder.predict(xServer, verbose=0)))
    print(f"\n  Avg Acc:{np.mean(fa):.4f} | "
          f"Avg F1:{np.mean(ff):.4f} | "
          f"Recon Err:{recon_err:.4f}")

    print(f"\n{'='*60}\n  GENERATING VISUALIZATIONS\n{'='*60}")

    plot_convergence(tracker,
        f"{FIGURE_DIR}/{MODEL_NAME}_03_convergence.png")
    plot_per_client(tracker,
        f"{FIGURE_DIR}/{MODEL_NAME}_04_per_client.png")
    plot_final_bars(fa, fp, fr, ff,
        f"{FIGURE_DIR}/{MODEL_NAME}_05_final_bars.png")
    yf = originclassificationmodel.predict(xServer, batch_size=300)
    plot_confusion(yServer.argmax(1), yf.argmax(1), LABELS,
        f"{FIGURE_DIR}/{MODEL_NAME}_06_confusion.png")
    plot_complexity(profiles,
        f"{FIGURE_DIR}/{MODEL_NAME}_07_complexity.png")
    plot_dashboard(tracker, fa, recon_err, profiles,
        f"{FIGURE_DIR}/{MODEL_NAME}_08_dashboard.png")

    tracker.save_csv(CONVERGENCE_CSV)

    pd.DataFrame([{
        'model':            MODEL_NAME,
        'dataset':          'ISCX_5class',
        'setting':          f'non-IID α={non_iid_alpha}',
        'aggregation':      'weighted_fedavg',
        'labels':           labelnum,
        'labels_per_class': labelnum // len(LABELS),
        'clients':          numOfClients,
        'rounds':           numOfIterations,
        'epochs':           epochs,
        'batch_size':       batch_size,
        'dropout':          DROPOUT_RATE,
        'avg_acc':          np.mean(fa),
        'avg_f1':           np.mean(ff),
        'min_acc':          min(fa),
        'max_acc':          max(fa),
        'std_acc':          np.std(fa),
        'recon_err':        recon_err,
        'time_min':         total_time / 60,
        'params':           profiles[0]['total_params'],
        'inference_ms':     profiles[1]['inference_ms'],
    }]).to_csv(
        f"{WORKING_BASE}/result/{MODEL_NAME}_final_summary.csv",
        index=False)

    with open(monitoring_filename, 'a+', newline='') as f:
        w = csv.DictWriter(f, monitorheaders)
        w.writeheader(); w.writerows(Globalmonitordirctrows)
    with open(performance_filename, 'a+', newline='') as f:
        w = csv.DictWriter(f, performanceheaders)
        w.writeheader(); w.writerows(performancerdirctros)

    push_checkpoint_to_kaggle(numOfIterations)

    print(f"\n{'='*60}\n  COMPLETE — {MODEL_NAME}")
    print(f"  Acc:{np.mean(fa):.4f} | F1:{np.mean(ff):.4f} | "
          f"Time:{total_time/60:.1f} min")
    print(f"  K={numOfClients} | dropout={DROPOUT_RATE} | α={non_iid_alpha}")
    print(f"  Dataset: ISCX ({len(LABELS)} classes)")
    print(f"  Results saved in: {WORKING_BASE}/result/")
    print(f"  Checkpoint → {KAGGLE_USERNAME}/{CHECKPOINT_DATASET}")
    print(f"{'='*60}")
