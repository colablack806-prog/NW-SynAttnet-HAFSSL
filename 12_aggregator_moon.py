# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HAFSSL v4 — HYBRID ATTENTION FEDERATED SEMI-SUPERVISED LEARNING       ║
# ║  Aggregation : MOON (Model-Contrastive Federated Learning)             ║
# ║  Environment : Kaggle                                                   ║
# ║  Dataset     : ISCX 5-Class (chat / email / file / streaming / voip)   ║
# ║  Clients     : 10                                                       ║
# ║                                                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────
# MOON — Key idea (Li et al., CVPR 2021):
#
#   During LOCAL TRAINING each client minimises:
#
#       L_total = L_CE  +  μ * L_con
#
#   where the contrastive loss is:
#
#       L_con = -log(
#           exp(sim(z, z_global) / τ)
#           ─────────────────────────────────────────────────
#           exp(sim(z, z_global) / τ) + exp(sim(z, z_prev) / τ)
#       )
#
#   z        = encoder representation of the CURRENT local model
#   z_global = encoder representation of the GLOBAL model (positive)
#   z_prev   = encoder representation of the PREVIOUS local model (negative)
#   sim      = cosine similarity
#   τ        = temperature (0.5)
#   μ        = contrastive weight (5.0)
#
#   AGGREGATION remains standard weighted FedAvg.
#
#   Reference: Li et al., "Model-Contrastive Federated Learning",
#              CVPR 2021. https://arxiv.org/abs/2103.16257
# ─────────────────────────────────────────────────────────────────────────────

!pip install GPUtil psutil kaggle -q   # uncomment on Kaggle

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
LABELS = ['chat', 'email', 'file', 'streaming', 'voip']

KAGGLE_USERNAME    = 'xxxxxx'
CHECKPOINT_DATASET = 'moon-hafssl-cheeckpoint'
DATASET_PATH       = (
    '/kaggle/input/datasets/xxxxxx/iscx-benchmark/ISCX_5class_each_normalized_cuttedfloefeature.csv'
)

CHECKPOINT_INPUT   = (
    f'/kaggle/input/datasets/{KAGGLE_USERNAME}/{CHECKPOINT_DATASET}/')



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

# ── MOON-specific hyper-parameters ─────────────────────────────────────────
MOON_MU          = 5.0   # contrastive loss weight  (μ)
MOON_TEMPERATURE = 0.5   # softmax temperature       (τ)

FORCE_FRESH_START = False
MODEL_NAME        = "HAFSSL_MOON_10client"

_alpha_str   = str(non_iid_alpha).replace('.', '')
_folder_name = (
    f'HAFSSL_MOON_10client_e{epochs}_b{batch_size}'
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

print("=" * 65)
print(f"  {MODEL_NAME}")
print(f"  Aggregation : MOON  (model-contrastive federated learning)")
print(f"  Dataset     : ISCX 5-Class")
print(f"  μ={MOON_MU} | τ={MOON_TEMPERATURE} | α={non_iid_alpha}")
print(f"  epochs={epochs} | rounds={numOfIterations} | "
      f"batch={batch_size} | K={numOfClients}")
print(f"  Output folder : {_folder_name}")
print(f"  FORCE_FRESH_START: {FORCE_FRESH_START}")
print("=" * 65)


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
             '-m', f'[HAFSSLv4-MOON-10K] Round {round_num}/{numOfIterations} '
                   f'e={epochs} b={batch_size} α={non_iid_alpha} '
                   f'μ={MOON_MU} τ={MOON_TEMPERATURE}',
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
            os.makedirs(f'{models_dir}/AECNNmodel', exist_ok=True)
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
# CONVERGENCE TRACKER  (extended for MOON contrastive losses)
# ═════════════════════════════════════════════════════════════════════════════

class ConvergenceTracker:
    def __init__(self):
        self.rounds        = []
        self.global_acc    = []
        self.global_prec   = []
        self.global_recall = []
        self.global_f1     = []
        self.per_client_acc = {}
        # MOON-specific: contrastive loss per round (mean over clients)
        self.mean_ce_loss  = []
        self.mean_con_loss = []

    def record_round(self, round_num, acc, prec, recall, f1,
                     client_accs, mean_ce=0.0, mean_con=0.0):
        self.rounds.append(round_num)
        self.global_acc.append(acc)
        self.global_prec.append(prec)
        self.global_recall.append(recall)
        self.global_f1.append(f1)
        self.per_client_acc[round_num] = client_accs
        self.mean_ce_loss.append(mean_ce)
        self.mean_con_loss.append(mean_con)

    def save_csv(self, filepath):
        pd.DataFrame({
            'round':        self.rounds,
            'accuracy':     self.global_acc,
            'precision':    self.global_prec,
            'recall':       self.global_recall,
            'f1_score':     self.global_f1,
            'mean_ce_loss': self.mean_ce_loss,
            'mean_con_loss':self.mean_con_loss,
        }).to_csv(filepath, index=False)


# ═════════════════════════════════════════════════════════════════════════════
# CHECKPOINT SAVE / LOAD
# ═════════════════════════════════════════════════════════════════════════════

def _to_python(obj):
    """
    Recursively convert numpy scalars / arrays to plain Python types so
    json.dump never encounters a non-serialisable numpy dtype.
    Handles: np.float32/64, np.int32/64, np.ndarray, list, dict.
    """
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # covers np.float32, np.float64, np.int32, np.int64, etc.
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_checkpoint(round_num, tracker):
    checkpoint = _to_python({
        'last_completed_round': round_num,
        'model':       MODEL_NAME,
        'epochs':      epochs,
        'batch_size':  batch_size,
        'rounds':      numOfIterations,
        'alpha':       non_iid_alpha,
        'aggregation': 'MOON',
        'moon_mu':     MOON_MU,
        'moon_tau':    MOON_TEMPERATURE,
        'tracker': {
            'rounds':         tracker.rounds,
            'global_acc':     tracker.global_acc,
            'global_prec':    tracker.global_prec,
            'global_recall':  tracker.global_recall,
            'global_f1':      tracker.global_f1,
            'mean_ce_loss':   tracker.mean_ce_loss,
            'mean_con_loss':  tracker.mean_con_loss,
            'per_client_acc': {
                str(k): v for k, v in tracker.per_client_acc.items()
            },
        },
    })
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
    tracker.mean_ce_loss   = t.get('mean_ce_loss',  [0.0]*len(t['rounds']))
    tracker.mean_con_loss  = t.get('mean_con_loss', [0.0]*len(t['rounds']))
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
    print(f"\n{'='*65}")
    print(f"  🔄 RESUMING FROM CHECKPOINT")
    print(f"  Last completed round : {last}")
    print(f"  Resuming from round  : {last + 1}")
    print(f"  Remaining rounds     : {numOfIterations - last}")
    print(f"  Last global acc      : {last_acc:.4f}")
    print(f"{'='*65}\n")
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

def load_iscx_dataset():
    print(f"  Loading dataset from: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    dfDS = pd.read_csv(DATASET_PATH)
    X_full = dfDS.iloc[:, 1:len(dfDS.columns)].values
    Y_full = dfDS["label"].values
    num_classes = len(set(Y_full))
    Y_full = tf.keras.utils.to_categorical(Y_full, num_classes)
    print(f"  X_full: {X_full.shape}   Classes: {num_classes}")
    return X_full, Y_full, num_classes


# ═════════════════════════════════════════════════════════════════════════════
# NON-IID PARTITION
# ═════════════════════════════════════════════════════════════════════════════

def split_non_iid(data, labels, num_clients, alpha=0.5, min_samples=5):
    n_classes  = len(np.unique(labels))
    N          = len(labels)
    min_size   = 0
    attempts   = 0
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
    x = Dense(n_classes, activation="softmax")(x)
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

    x_unlabeled = x_train[np.setdiff1d(np.arange(len(x_train)), all_idx)]
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
# ███╗   ███╗ ██████╗  ██████╗ ███╗   ██╗
# ████╗ ████║██╔═══██╗██╔═══██╗████╗  ██║
# ██╔████╔██║██║   ██║██║   ██║██╔██╗ ██║
# ██║╚██╔╝██║██║   ██║██║   ██║██║╚██╗██║
# ██║ ╚═╝ ██║╚██████╔╝╚██████╔╝██║ ╚████║
# ╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝
#
# MOON: Model-Contrastive Federated Learning
# ═════════════════════════════════════════════════════════════════════════════

def moon_contrastive_loss(z_local, z_global, z_prev, temperature=MOON_TEMPERATURE):
    """
    Compute the MOON contrastive loss.

    Parameters
    ----------
    z_local  : tf.Tensor (B, D) — encoder output of the current local model
    z_global : tf.Tensor (B, D) — encoder output of the global model (positive)
    z_prev   : tf.Tensor (B, D) — encoder output of the previous local model (negative)
    temperature : float         — softmax temperature τ

    Returns
    -------
    Scalar tensor: mean contrastive loss over the batch.
    """
    # L2-normalise all representations
    z_local_n  = tf.nn.l2_normalize(z_local,  axis=1)
    z_global_n = tf.nn.l2_normalize(z_global, axis=1)
    z_prev_n   = tf.nn.l2_normalize(z_prev,   axis=1)

    # Cosine similarities scaled by temperature — shape (B, 1)
    sim_pos = tf.reduce_sum(z_local_n * z_global_n, axis=1, keepdims=True) / temperature
    sim_neg = tf.reduce_sum(z_local_n * z_prev_n,   axis=1, keepdims=True) / temperature

    # Concatenate: [positive | negative] → shape (B, 2)
    logits = tf.concat([sim_pos, sim_neg], axis=1)

    # Labels: 0 → positive (global) is always at index 0
    labels = tf.zeros(tf.shape(z_local)[0], dtype=tf.int32)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return tf.reduce_mean(loss)


def moon_train_epoch(local_cnn, global_cnn, prev_cnn,
                     x_labeled, y_labeled, optimizer,
                     mu=MOON_MU, temperature=MOON_TEMPERATURE,
                     b_size=batch_size):
    """
    Run ONE epoch of MOON-augmented supervised training on a client.

    The total loss for each mini-batch is:
        L = L_CE + μ * L_con

    Parameters
    ----------
    local_cnn   : Keras model being trained (current local model)
    global_cnn  : Keras model with GLOBAL weights (frozen, positive reference)
    prev_cnn    : Keras model with PREVIOUS local weights (frozen, negative)
    x_labeled   : np.ndarray (N, seq_len, 1)
    y_labeled   : np.ndarray (N, n_classes)
    optimizer   : tf.keras.Optimizer
    mu          : contrastive loss weight
    temperature : contrastive temperature τ
    b_size      : mini-batch size

    Returns
    -------
    mean_ce_loss, mean_con_loss, mean_accuracy  (float, float, float)
    """
    dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_labeled, y_labeled))
        .shuffle(len(x_labeled))
        .batch(b_size)
    )

    ce_losses, con_losses, accs = [], [], []

    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            # ── Forward pass: classification ───────────────────────────
            y_pred = local_cnn(x_batch, training=True)
            ce_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(y_batch, y_pred))

            # ── Representations from all three models ──────────────────
            z_local  = local_cnn.encoder(x_batch,  training=True)
            z_global = global_cnn.encoder(x_batch, training=False)
            z_prev   = prev_cnn.encoder(x_batch,   training=False)

            # ── MOON contrastive loss ──────────────────────────────────
            con_loss = moon_contrastive_loss(
                z_local, z_global, z_prev, temperature)

            total_loss = ce_loss + mu * con_loss

        # ── Gradient update ────────────────────────────────────────────
        grads = tape.gradient(total_loss, local_cnn.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, local_cnn.trainable_variables))

        acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_batch, 1)),
            tf.float32))

        ce_losses.append(ce_loss.numpy())
        con_losses.append(con_loss.numpy())
        accs.append(acc.numpy())

    return np.mean(ce_losses), np.mean(con_losses), np.mean(accs)


# ═════════════════════════════════════════════════════════════════════════════
# MOON AGGREGATOR  — standard weighted FedAvg
# (MOON's novelty is in LOCAL TRAINING, not aggregation)
# ═════════════════════════════════════════════════════════════════════════════

class MoonAggregator:
    """
    Standard weighted FedAvg aggregation used by MOON.

    w_global = Σ_i (n_i / N) * w_i

    Reference: Li et al., CVPR 2021 use vanilla FedAvg for the server step;
    the contrastive loss is applied exclusively during local training.
    """

    def __init__(self):
        self.client_weights = []
        self.client_samples = []
        self.total_samples  = 0

    def reset(self):
        self.client_weights = []
        self.client_samples = []
        self.total_samples  = 0

    def add_client_update(self, client_weights, n_samples):
        self.client_weights.append([w.copy() for w in client_weights])
        self.client_samples.append(n_samples)
        self.total_samples += n_samples

    def aggregate(self):
        if not self.client_weights:
            raise ValueError("MoonAggregator: no client updates added.")

        N        = self.total_samples
        n_params = len(self.client_weights[0])
        new_weights = [
            np.zeros_like(self.client_weights[0][p])
            for p in range(n_params)
        ]
        for weights, n_i in zip(self.client_weights, self.client_samples):
            frac = n_i / N
            for p in range(n_params):
                new_weights[p] += frac * weights[p]
        return new_weights


# ═════════════════════════════════════════════════════════════════════════════
# HELPER: clone a frozen reference model for MOON
# ═════════════════════════════════════════════════════════════════════════════

def clone_frozen_cnn(source_model):
    """
    Return a non-trainable copy of source_model with the encoder sub-model
    attribute re-attached. Used for global and previous-local references
    in MOON contrastive loss.
    """
    ref = tf.keras.models.clone_model(source_model)
    ref.set_weights(source_model.get_weights())
    ref.encoder = ref.get_layer("encoder")
    for layer in ref.layers:
        layer.trainable = False
    return ref


# ═════════════════════════════════════════════════════════════════════════════
# MODEL PROFILING
# ═════════════════════════════════════════════════════════════════════════════

def profile_model(model, input_shape, name="Model"):
    total_params = model.count_params()
    trainable    = sum(
        tf.keras.backend.count_params(w)
        for w in model.trainable_weights)
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
        'name':            name,
        'total_params':    total_params,
        'trainable_params':trainable,
        'inference_ms':    avg_time,
        'model_size_mb':   size_mb,
    }


# ═════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════

def plot_non_iid_distribution(yClientsList, labels, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
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
    ax.tick_params(axis='x', rotation=30)
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
        f'Non-IID Sample Distribution (α={non_iid_alpha}) — MOON',
        fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.tick_params(axis='x', rotation=15, labelsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")


def plot_convergence(tracker, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tracker.rounds, tracker.global_acc,    'o-',
            color='#1976D2', markersize=4, label='Accuracy')
    ax.plot(tracker.rounds, tracker.global_f1,     's-',
            color='#E53935', markersize=4, label='F1')
    ax.plot(tracker.rounds, tracker.global_prec,   '^-',
            color='#43A047', markersize=3, alpha=0.7, label='Precision')
    ax.plot(tracker.rounds, tracker.global_recall, 'v-',
            color='#FF9800', markersize=3, alpha=0.7, label='Recall')
    ax.fill_between(tracker.rounds, tracker.global_acc,
                    alpha=0.1, color='#1976D2')
    ax.set_xlabel('FL Round'); ax.set_ylabel('Score')
    ax.set_title(
        f'{MODEL_NAME} — Convergence '
        f'(MOON | μ={MOON_MU}, τ={MOON_TEMPERATURE}, '
        f'α={non_iid_alpha}, K={numOfClients})',
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
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, numOfClients))
    for cid in range(numOfClients):
        accs = [tracker.per_client_acc[r][cid]
                for r in tracker.rounds
                if cid < len(tracker.per_client_acc[r])]
        rounds_valid = [r for r in tracker.rounds
                        if cid < len(tracker.per_client_acc[r])]
        ax.plot(rounds_valid, accs, '-', color=colors[cid],
                alpha=0.65, linewidth=1.3, label=f'Client {cid}')
    ax.plot(tracker.rounds, tracker.global_acc, 'k--',
            linewidth=2.5, label='Global acc', zorder=10)
    ax.set_xlabel('FL Round'); ax.set_ylabel('Accuracy')
    ax.set_title(
        f'{MODEL_NAME} — Per-Client Accuracy (MOON | K={numOfClients})',
        fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")


def plot_contrastive_loss(tracker, save_path):
    """
    MOON-specific: plot CE loss vs contrastive loss evolution over rounds.
    Shows how the contrastive regularisation behaves relative to classification.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: absolute losses ──────────────────────────────────────────────
    ax1.plot(tracker.rounds, tracker.mean_ce_loss,  'o-',
             color='#1976D2', markersize=4, label='CE Loss')
    ax1.plot(tracker.rounds, tracker.mean_con_loss, 's-',
             color='#E53935', markersize=4, label='Contrastive Loss')
    ax1.set_xlabel('FL Round'); ax1.set_ylabel('Loss')
    ax1.set_title('MOON Loss Components per Round', fontweight='bold')
    ax1.legend()

    # ── Right: contrastive-to-CE ratio ────────────────────────────────────
    ratio = [
        c / (e + 1e-12)
        for c, e in zip(tracker.mean_con_loss, tracker.mean_ce_loss)
    ]
    ax2.plot(tracker.rounds, ratio, '-', color='#9C27B0', linewidth=2)
    ax2.fill_between(tracker.rounds, ratio, alpha=0.15, color='#9C27B0')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.6,
                label='CE = Contrastive')
    ax2.set_xlabel('FL Round')
    ax2.set_ylabel('L_con / L_CE Ratio')
    ax2.set_title('Contrastive-to-CE Ratio', fontweight='bold')
    ax2.legend()

    plt.suptitle(
        f'{MODEL_NAME} — MOON Contrastive Loss Dynamics\n'
        f'μ={MOON_MU}, τ={MOON_TEMPERATURE}, α={non_iid_alpha}, K={numOfClients}',
        fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")


def plot_final_bars(acc_l, prec_l, rec_l, f1_l, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        (acc_l,  'Accuracy',  '#1976D2', axes[0, 0]),
        (prec_l, 'Precision', '#43A047', axes[0, 1]),
        (rec_l,  'Recall',    '#FF9800', axes[1, 0]),
        (f1_l,   'F1 Score',  '#E53935', axes[1, 1]),
    ]
    for vals, title, color, ax in metrics:
        x = [f'C{i}' for i in range(len(vals))]
        bar_colors = [
            '#E53935' if v < 0.70 else '#FFA726' if v < 0.85 else color
            for v in vals
        ]
        ax.bar(x, vals, color=bar_colors, edgecolor='white')
        ax.axhline(y=np.mean(vals), color='navy', linestyle='--',
                   alpha=0.7, label=f'Mean={np.mean(vals):.3f}')
        ax.set_ylim([0, 1.15]); ax.set_title(title, fontweight='bold')
        ax.set_ylabel(title); ax.legend(fontsize=9)
        ax.tick_params(axis='x', rotation=15)
    plt.suptitle(
        f'{MODEL_NAME} — Final Metrics per Client (MOON)',
        fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")


def plot_confusion(y_true, y_pred, labels, save_path):
    cm   = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm,   annot=True, fmt='d',    square=True, cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=a1,
                annot_kws={"fontsize": 9})
    sns.heatmap(cm_n, annot=True, fmt='.0%', square=True, cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels, ax=a2,
                annot_kws={"fontsize": 9})
    a1.set_title('Counts', fontweight='bold')
    a1.tick_params(axis='x', rotation=30)
    a2.set_title('Normalized', fontweight='bold')
    a2.tick_params(axis='x', rotation=30)
    plt.suptitle(
        f'{MODEL_NAME} — Confusion Matrix '
        f'(MOON | μ={MOON_MU}, α={non_iid_alpha}, K={numOfClients})',
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
    a1.set_ylabel('Params (K)')
    a1.set_title('Model Complexity', fontweight='bold')
    a1.bar_label(b1, fmt='%.0fK', fontsize=9, padding=2)
    b2 = a2.bar(names, [p['inference_ms'] for p in profiles], color=cols)
    a2.set_ylabel('Inference (ms)')
    a2.set_title('Inference Latency', fontweight='bold')
    a2.bar_label(b2, fmt='%.1fms', fontsize=9, padding=2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")


def plot_dashboard(tracker, acc_list, recon_err, profiles, save_path):
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # ── Convergence ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(tracker.rounds, tracker.global_acc, 'o-',
             color='#1976D2', markersize=3, label='Acc')
    ax1.plot(tracker.rounds, tracker.global_f1,  's-',
             color='#E53935', markersize=3, label='F1')
    ax1.set_xlabel('Round'); ax1.set_ylabel('Score')
    ax1.set_title('Convergence (MOON)', fontweight='bold')
    ax1.set_ylim([0, 1.05]); ax1.legend(fontsize=9)

    # ── Per-client final acc bars ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    acc_arr    = np.array(acc_list)
    bar_colors = ['#E53935' if a < 0.70 else '#FFA726'
                  if a < 0.85 else '#43A047' for a in acc_arr]
    ax2.bar([f'C{i}' for i in range(len(acc_arr))],
            acc_arr, color=bar_colors, edgecolor='white')
    ax2.axhline(y=np.mean(acc_arr), color='navy', linestyle='--',
                alpha=0.6, label=f'Mean={np.mean(acc_arr):.3f}')
    ax2.set_xlabel('Client'); ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Final Client Accuracy (K={numOfClients})',
                  fontweight='bold')
    ax2.set_ylim([0, 1.2]); ax2.legend(fontsize=8)
    ax2.tick_params(axis='x', rotation=15)

    # ── Summary text box ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2]); ax3.axis('off')
    r90 = next(
        (r for r, a in zip(tracker.rounds, tracker.global_acc) if a >= 0.90),
        'N/A')
    mean_con = np.mean(tracker.mean_con_loss) if tracker.mean_con_loss else 0
    mean_ce  = np.mean(tracker.mean_ce_loss)  if tracker.mean_ce_loss  else 0
    txt = (
        f"{MODEL_NAME}\n"
        f"Aggregation : MOON (FedAvg + contrastive)\n"
        f"ISCX 5 Classes | Kaggle\n"
        f"α={non_iid_alpha} | K={numOfClients}\n"
        f"Labels={labelnum} ({labelnum//len(LABELS)}/class)\n"
        f"Rounds={numOfIterations} | epochs={epochs}\n"
        f"batch={batch_size} | dropout={DROPOUT_RATE}\n"
        f"μ={MOON_MU}  |  τ={MOON_TEMPERATURE}\n\n"
        f"Final Acc  : {tracker.global_acc[-1]:.4f}\n"
        f"Final F1   : {tracker.global_f1[-1]:.4f}\n"
        f"Recon Err  : {recon_err:.4f}\n\n"
        f"Max : {max(acc_list):.4f}  Min : {min(acc_list):.4f}\n"
        f"Std : {np.std(acc_list):.4f}\n\n"
        f"Avg L_con  : {mean_con:.4f}\n"
        f"Avg L_CE   : {mean_ce:.4f}\n"
        f"Params     : {profiles[0]['total_params']:,}\n"
        f"Rounds→90% : {r90}"
    )
    ax3.text(0.5, 0.5, txt, ha='center', va='center', fontsize=9,
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='#E3F2FD',
                       edgecolor='#1976D2', linewidth=2))
    ax3.set_title('Summary', fontweight='bold')

    # ── Client variance ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    vars_ = [np.std(tracker.per_client_acc[r]) for r in tracker.rounds]
    ax4.plot(tracker.rounds, vars_, '-', color='#9C27B0')
    ax4.fill_between(tracker.rounds, vars_, alpha=0.2, color='#9C27B0')
    ax4.axhline(y=np.mean(vars_), color='red', linestyle='--',
                alpha=0.7, label=f'Mean={np.mean(vars_):.3f}')
    ax4.legend(fontsize=9)
    ax4.set_xlabel('Round'); ax4.set_ylabel('Std Dev')
    ax4.set_title('Client Variance', fontweight='bold')

    # ── MOON: contrastive vs CE loss per round ────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(tracker.rounds, tracker.mean_ce_loss,  'o-',
             color='#1976D2', markersize=3, label='CE Loss')
    ax5.plot(tracker.rounds, tracker.mean_con_loss, 's-',
             color='#E53935', markersize=3, label='Contrastive Loss')
    ax5.set_xlabel('Round'); ax5.set_ylabel('Loss')
    ax5.set_title('MOON Loss Components', fontweight='bold')
    ax5.legend(fontsize=8)

    # ── Communication cost ────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2]); ax6.axis('off')
    bpr  = sum(p['model_size_mb'] for p in profiles) * numOfClients * 2
    ctxt = (
        f"MOON Communication\n\n"
        f"Per round : {bpr:.1f} MB\n"
        f"({numOfClients} clients × up+down)\n\n"
        f"Rounds to 90% : {r90}\n"
        f"Total rounds  : {numOfIterations}\n"
        f"Total comm    : {bpr*numOfIterations:.0f} MB\n\n"
        f"Contrastive weight μ = {MOON_MU}\n"
        f"Temperature       τ = {MOON_TEMPERATURE}\n\n"
        f"Dataset  : ISCX | Classes : {len(LABELS)}\n"
        f"α={non_iid_alpha} | K={numOfClients}"
    )
    ax6.text(0.5, 0.5, ctxt, ha='center', va='center', fontsize=10,
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='#E8F5E9',
                       edgecolor='#43A047', linewidth=2))
    ax6.set_title('Communication (MOON)', fontweight='bold')

    plt.suptitle(
        f'{MODEL_NAME} — Dashboard '
        f'(MOON | μ={MOON_MU}, τ={MOON_TEMPERATURE}, '
        f'α={non_iid_alpha}, K={numOfClients}, e={epochs})',
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
Globalmonitordirct     = {h: 0 for h in monitorheaders}
Globalmonitordirct['stage'] = ''
Globalmonitordirctrows = []

performanceheaders = [
    'stage', 'iterationNo', 'clientID',
    'train_acc', 'val_acc', 'test_acc', 'classification_report',
    'ce_loss',   'con_loss',   # MOON-specific
]
performancerdirct  = {h: 0 for h in performanceheaders}
performancerdirct['stage']                 = ''
performancerdirct['classification_report'] = ''
performancerdirct['ce_loss']               = 0.0
performancerdirct['con_loss']              = 0.0
performancerdirctros = []


class Monitor(Thread):
    def __init__(self, delay, stage, iterationNo, clientID, process):
        super(Monitor, self).__init__()
        self.stopped      = False
        self.delay        = delay
        self.stage        = stage
        self.iterationNo  = iterationNo
        self.clientID     = clientID
        self.process      = process
        self.gpu_mem_list = []; self.gpu_load_list = []
        self.used_mem_list= []; self.cpu_load_list = []
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
        X_full, Y_full, num_classes = load_iscx_dataset()
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
        X_full, Y_full, num_classes = load_iscx_dataset()
        assert num_classes == len(LABELS), (
            f"Expected {len(LABELS)} classes, got {num_classes}")

        xServer, xClients, yServer, yClients = train_test_split(
            X_full, Y_full, test_size=0.90, random_state=523)
        xServer = np.expand_dims(xServer, axis=2)
        inp_size = xServer.shape[1]
        print(f"  xServer:{xServer.shape}  "
              f"xClients:{xClients.shape}  Classes:{num_classes}")

        # ── Model initialisation ───────────────────────────────────────────
        if start_round == 1:
            print(f"\n  Creating new model "
                  f"(MOON, {numOfClients} clients, "
                  f"μ={MOON_MU}, τ={MOON_TEMPERATURE})...")
            originautoencoder, originclassificationmodel = \
                createHAFSSLModel(inp_size, num_classes)
            profiles = [
                profile_model(originautoencoder,
                              (inp_size, 1), "Autoencoder"),
                profile_model(originclassificationmodel,
                              (inp_size, 1), "Classifier"),
            ]
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
                updateClientsModels()
                originautoencoder.save(AEmodelLocation)
                originclassificationmodel.save(CNNmodelLocation)
                tracker    = ConvergenceTracker()
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
                updateClientsModels()
                print("  ✅ Models loaded + attributes re-attached")

        # ── Data partition ─────────────────────────────────────────────────
        print(f"\n  Partitioning data (seed=42, α={non_iid_alpha})...")
        np.random.seed(42)
        yClients_int   = np.argmax(yClients, axis=1)
        client_indices = split_non_iid(
            xClients, yClients_int,
            numOfClients, alpha=non_iid_alpha, min_samples=5)
        xClientsList = [xClients[client_indices[c]]
                        for c in range(numOfClients)]
        yClientsList = [yClients[client_indices[c]]
                        for c in range(numOfClients)]

        # ── Load / initialise per-client models ────────────────────────────
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
                    cnn = tf.keras.models.clone_model(originclassificationmodel)
                    cnn.set_weights(originclassificationmodel.get_weights())
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

        # ── MOON: initialise PREVIOUS local model list ─────────────────────
        # For round 1 (or when no previous model exists), we use the global
        # model as the previous model.  This makes z_prev ≈ z_global, so
        # the contrastive loss starts near zero and grows as clients diverge.
        print("  Initialising MOON previous-model references...")
        prevClientsCNNModelist = []
        for cid in range(numOfClients):
            prev_path = f"{CLIENT_MODEL_DIR}/PREV_CNN_node_{cid}.keras"
            if os.path.exists(prev_path):
                prev = load_model(prev_path)
                prev.encoder = prev.get_layer("encoder")
                for layer in prev.layers:
                    layer.trainable = False
                prevClientsCNNModelist.append(prev)
                print(f"    ✅ Client {cid} previous model loaded")
            else:
                # Fall back: use global model as previous
                prev = clone_frozen_cnn(originclassificationmodel)
                prevClientsCNNModelist.append(prev)
                print(f"    ℹ️  Client {cid} — no previous model, "
                      f"using global as prev")

        # ── Distribution plots (round 1 only) ─────────────────────────────
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

        # ── Labeled / Unlabeled split ──────────────────────────────────────
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
        print(f"\n  Sample sizes : {client_sample_sizes}")
        print(f"  Total        : {sum(client_sample_sizes)}")

        # ── MOON aggregators (one per model) ──────────────────────────────
        agg_ae  = MoonAggregator()
        agg_cnn = MoonAggregator()

        start_time = time.time()
        process    = psutil.Process(os.getpid())

        print(f"\n{'='*65}")
        print(f"  {MODEL_NAME}  |  MOON  |  Round {start_round}/{numOfIterations}")
        print(f"  μ={MOON_MU} | τ={MOON_TEMPERATURE} | α={non_iid_alpha}")
        print(f"  K={numOfClients} | epochs={epochs} | "
              f"batch={batch_size} | dropout={DROPOUT_RATE}")
        print(f"  Checkpoint → {KAGGLE_USERNAME}/{CHECKPOINT_DATASET}")
        print(f"{'='*65}")

        # ══════════════════════════════════════════════════════════════════
        # TRAINING LOOP
        # ══════════════════════════════════════════════════════════════════
        for iterationNo in range(start_round, numOfIterations + 1):
            print(f"\n{'='*65}\n  Round {iterationNo}/{numOfIterations}"
                  f"  [MOON]\n{'='*65}")
            round_client_accs = []
            round_ce_losses   = []
            round_con_losses  = []

            # ── Global reference model for MOON (frozen, positive) ─────────
            global_cnn_ref = clone_frozen_cnn(originclassificationmodel)

            agg_ae.reset()
            agg_cnn.reset()

            for clientID in range(numOfClients):
                print(f"\n  ── Client {clientID} ──")

                if client_sample_sizes[clientID] < 50:
                    print(f"  ⚠️  Client {clientID} skipped — "
                          f"only {client_sample_sizes[clientID]} samples")
                    round_client_accs.append(0.0)
                    round_ce_losses.append(0.0)
                    round_con_losses.append(0.0)
                    continue

                monitor    = Monitor(1, "HAFSSLv4-MOON training",
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

                n_unlabeled = len(xClientsListUnLabel[clientID])
                n_labeled   = len(xClientsListLabel[clientID])

                # ── Stage 1: AE unsupervised ──────────────────────────────
                # (no contrastive loss here — AE is self-supervised)
                if n_unlabeled >= 50:
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
                # (frozen CNN head, no MOON contrastive loss)
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

                # ── Stage 2: CNN full training WITH MOON contrastive loss ──
                # Uses:
                #   positive = global_cnn_ref  (global model)
                #   negative = prevClientsCNNModelist[clientID]  (prev local)
                optimizer = tf.keras.optimizers.Adam()
                prev_model_ref = prevClientsCNNModelist[clientID]

                epoch_ce_vals, epoch_con_vals, epoch_acc_vals = [], [], []
                for ep in range(epochs):
                    ce_ep, con_ep, acc_ep = moon_train_epoch(
                        local_cnn   = subCNNmodel,
                        global_cnn  = global_cnn_ref,
                        prev_cnn    = prev_model_ref,
                        x_labeled   = xClientsListLabel[clientID],
                        y_labeled   = yClientsListLabel[clientID],
                        optimizer   = optimizer,
                        mu          = MOON_MU,
                        temperature = MOON_TEMPERATURE,
                        b_size      = batch_size,
                    )
                    epoch_ce_vals.append(ce_ep)
                    epoch_con_vals.append(con_ep)
                    epoch_acc_vals.append(acc_ep)
                    if verbose >= 1:
                        print(f"    Epoch {ep+1}/{epochs} — "
                              f"CE:{ce_ep:.4f}  "
                              f"Con:{con_ep:.4f}  "
                              f"Acc:{acc_ep:.4f}")

                monitor.stop()

                client_ce  = np.mean(epoch_ce_vals)
                client_con = np.mean(epoch_con_vals)
                round_ce_losses.append(client_ce)
                round_con_losses.append(client_con)

                # ── Evaluate on server test set ───────────────────────────
                y_pr   = subCNNmodel.predict(xServer, batch_size=300)
                report = classification_report(
                    yServer.argmax(1), y_pr.argmax(1),
                    target_names=LABELS, zero_division=1, output_dict=True)
                round_client_accs.append(report['accuracy'])
                print(f"  Client {clientID} — "
                      f"Acc:{report['accuracy']:.4f} | "
                      f"F1:{report['weighted avg']['f1-score']:.4f} | "
                      f"L_CE:{client_ce:.4f} | "
                      f"L_con:{client_con:.4f}")

                performancerdirct.update({
                    'stage':       'HAFSSLv4-MOON training',
                    'iterationNo': iterationNo,
                    'clientID':    clientID,
                    'train_acc':   epoch_acc_vals[-1],
                    'val_acc':     report['accuracy'],
                    'test_acc':    report['accuracy'],
                    'classification_report': report,
                    'ce_loss':     client_ce,
                    'con_loss':    client_con,
                })
                performancerdirctros.append(performancerdirct.copy())

                # ── MOON aggregation: add client update ───────────────────
                agg_ae.add_client_update(
                    subAEmodel.get_weights(),
                    client_sample_sizes[clientID])
                agg_cnn.add_client_update(
                    subCNNmodel.get_weights(),
                    client_sample_sizes[clientID])

                # ── Save updated previous-local model for NEXT round ───────
                # We store the CURRENT local weights before overwriting so
                # they become prev in the next round.
                prev_path = f"{CLIENT_MODEL_DIR}/PREV_CNN_node_{clientID}.keras"
                subCNNmodel.save(prev_path)          # will be loaded as "prev" next round

                subCNNmodel.save(f"{CLIENT_MODEL_DIR}/CNN_node_{clientID}.keras")
                subAEmodel.save(f"{CLIENT_MODEL_DIR}/AE_node_{clientID}.keras")
                print(f"  ✅ Client {clientID} models saved")

            # ── MOON Aggregation (weighted FedAvg) ─────────────────────────
            print("\n  ── MOON Aggregation (weighted FedAvg) ──")
            try:
                new_ae_weights  = agg_ae.aggregate()
                new_cnn_weights = agg_cnn.aggregate()
                originautoencoder.set_weights(new_ae_weights)
                originclassificationmodel.set_weights(new_cnn_weights)
            except ValueError as e:
                print(f"  ⚠️  Aggregation skipped this round: {e}")
            originautoencoder.save(AEmodelLocation)
            originclassificationmodel.save(CNNmodelLocation)
            reattach_attributes(originautoencoder, originclassificationmodel)
            print(f"  ✅ Global models saved (Round {iterationNo})")

            # ── After aggregation: reload prev-local models for next round ─
            prevClientsCNNModelist = []
            for cid in range(numOfClients):
                prev_path = f"{CLIENT_MODEL_DIR}/PREV_CNN_node_{cid}.keras"
                if os.path.exists(prev_path):
                    prev = load_model(prev_path)
                    prev.encoder = prev.get_layer("encoder")
                    for layer in prev.layers:
                        layer.trainable = False
                    prevClientsCNNModelist.append(prev)
                else:
                    prevClientsCNNModelist.append(
                        clone_frozen_cnn(originclassificationmodel))

            # ── Global evaluation ─────────────────────────────────────────
            y_gpr = originclassificationmodel.predict(
                xServer, batch_size=300)
            gr = classification_report(
                yServer.argmax(1), y_gpr.argmax(1),
                target_names=LABELS, zero_division=1, output_dict=True)

            mean_ce_round  = np.mean(round_ce_losses)  if round_ce_losses  else 0.0
            mean_con_round = np.mean(round_con_losses) if round_con_losses else 0.0

            tracker.record_round(
                iterationNo,
                gr['accuracy'],
                gr['weighted avg']['precision'],
                gr['weighted avg']['recall'],
                gr['weighted avg']['f1-score'],
                round_client_accs,
                mean_ce  = mean_ce_round,
                mean_con = mean_con_round,
            )
            print(f"\n  Round {iterationNo} Global (MOON) — "
                  f"Acc:{gr['accuracy']:.4f} | "
                  f"F1:{gr['weighted avg']['f1-score']:.4f} | "
                  f"L_CE:{mean_ce_round:.4f} | "
                  f"L_con:{mean_con_round:.4f}")

            save_checkpoint(iterationNo, tracker)
            tracker.save_csv(CONVERGENCE_CSV)
            push_checkpoint_to_kaggle(iterationNo)
            updateClientsModels()

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL EVALUATION
    # ═══════════════════════════════════════════════════════════════════════
    total_time = time.time() - start_time
    print(f"\n{'='*65}\n  FINAL EVALUATION — {MODEL_NAME}"
          f"\n  Aggregation : MOON (μ={MOON_MU}, τ={MOON_TEMPERATURE})"
          f"\n  Time: {total_time/60:.1f} min\n{'='*65}")

    fa, fp, fr, ff = [], [], [], []
    for cid in range(numOfClients):
        nm       = originclassificationmodel
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
            'stage':    'Global validation after MOON training',
            'clientID': cid,
            'test_acc': r['accuracy'],
            'classification_report': r,
        })
        performancerdirctros.append(performancerdirct.copy())

    recon_err = np.mean(
        np.square(xServer -
                  originautoencoder.predict(xServer, verbose=0)))
    print(f"\n  Avg Acc:{np.mean(fa):.4f} | "
          f"Avg F1:{np.mean(ff):.4f} | "
          f"Recon Err:{recon_err:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # VISUALISATIONS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}\n  GENERATING VISUALIZATIONS (MOON)\n{'='*65}")

    plot_convergence(tracker,
        f"{FIGURE_DIR}/{MODEL_NAME}_03_convergence.png")
    plot_per_client(tracker,
        f"{FIGURE_DIR}/{MODEL_NAME}_04_per_client.png")
    plot_contrastive_loss(tracker,          # ← MOON-specific (replaces τ plot)
        f"{FIGURE_DIR}/{MODEL_NAME}_05_contrastive_loss.png")
    plot_final_bars(fa, fp, fr, ff,
        f"{FIGURE_DIR}/{MODEL_NAME}_06_final_bars.png")
    yf = originclassificationmodel.predict(xServer, batch_size=300)
    plot_confusion(yServer.argmax(1), yf.argmax(1), LABELS,
        f"{FIGURE_DIR}/{MODEL_NAME}_07_confusion.png")
    plot_complexity(profiles,
        f"{FIGURE_DIR}/{MODEL_NAME}_08_complexity.png")
    plot_dashboard(tracker, fa, recon_err, profiles,
        f"{FIGURE_DIR}/{MODEL_NAME}_09_dashboard.png")

    tracker.save_csv(CONVERGENCE_CSV)

    # ── Final summary CSV ──────────────────────────────────────────────────
    pd.DataFrame([{
        'model':             MODEL_NAME,
        'aggregation':       'MOON',
        'moon_mu':           MOON_MU,
        'moon_temperature':  MOON_TEMPERATURE,
        'dataset':           'ISCX_5class',
        'setting':           f'non-IID α={non_iid_alpha}',
        'labels':            labelnum,
        'labels_per_class':  labelnum // len(LABELS),
        'clients':           numOfClients,
        'rounds':            numOfIterations,
        'epochs':            epochs,
        'batch_size':        batch_size,
        'dropout':           DROPOUT_RATE,
        'avg_acc':           np.mean(fa),
        'avg_f1':            np.mean(ff),
        'min_acc':           min(fa),
        'max_acc':           max(fa),
        'std_acc':           np.std(fa),
        'recon_err':         recon_err,
        'time_min':          total_time / 60,
        'params':            profiles[0]['total_params'],
        'inference_ms':      profiles[1]['inference_ms'],
        'mean_ce_loss':      (np.mean(tracker.mean_ce_loss)
                              if tracker.mean_ce_loss else 0),
        'mean_con_loss':     (np.mean(tracker.mean_con_loss)
                              if tracker.mean_con_loss else 0),
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

    print(f"\n{'='*65}\n  COMPLETE — {MODEL_NAME}")
    print(f"  Aggregation : MOON (μ={MOON_MU}, τ={MOON_TEMPERATURE})")
    print(f"  Acc:{np.mean(fa):.4f} | F1:{np.mean(ff):.4f} | "
          f"Time:{total_time/60:.1f} min")
    print(f"  K={numOfClients} | dropout={DROPOUT_RATE} | α={non_iid_alpha}")
    print(f"  Results saved in: {WORKING_BASE}/result/")
    print(f"  Checkpoint → {KAGGLE_USERNAME}/{CHECKPOINT_DATASET}")
    print(f"{'='*65}")
