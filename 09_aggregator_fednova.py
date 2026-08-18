# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HAFSSL — FedNova aggregation (Table VII)                              ║
# ║  Dataset : ISCX (5-class) | non-IID α=0.5 | K=10                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Required installations
!pip install GPUtil psutil gdown

# Import Libraries
from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Flatten,
    Convolution1D, MaxPooling1D, UpSampling1D, Add, LayerNormalization,
    GlobalAveragePooling1D, MultiHeadAttention, Dropout
)
from tensorflow.keras.models import Model, load_model
from google.colab import drive
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf
import os
import psutil
import gdown
from threading import Thread
import GPUtil
import time
import datetime
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ═════════════════════════════════════════════════════════════════════════════
# PLOT STYLE — IEEE paper quality
# ═════════════════════════════════════════════════════════════════════════════
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
labelnum = 500
latent_dim = 39

verbose, epochs, batch_size = 2, 5, 256
numOfIterations = 50
numOfClients =10


non_iid_alpha = 0.5
MODEL_NAME = "HAFSSL-FedNova"

os.makedirs("./result", exist_ok=True)
os.makedirs("./result/figures", exist_ok=True)
os.makedirs("./Models", exist_ok=True)
os.makedirs("./Models/AECNNmodel", exist_ok=True)

FIGURE_DIR = "./result/figures"
monitoring_filename = f"./result/{MODEL_NAME}_nonIID_{labelnum}label_monitoring.csv"
performance_filename = f"./result/{MODEL_NAME}_nonIID_{labelnum}label_performance.csv"
AEmodelLocation = f"./Models/{MODEL_NAME}_AE_{numOfClients}_nodes.keras"
CNNmodelLocation = f"./Models/{MODEL_NAME}_CNN_{numOfClients}_nodes.keras"


# ═════════════════════════════════════════════════════════════════════════════
# FedNova AGGREGATOR
#
# Standard FedAvg aggregation:
#   w_new = Σ (p_k * w_k)
#   Problem: clients with more local steps drift further, biasing the average
#
# FedNova aggregation (normalized averaging):
#   d_k = (w_global - w_k) / τ_k          ← normalize by local steps
#   d_avg = Σ (p_k * d_k)                 ← weighted average of normalized updates
#   τ_eff = Σ (p_k * τ_k)                 ← effective number of steps
#   w_new = w_global - τ_eff * d_avg       ← apply with correct scaling
#
# This ensures all clients contribute equally regardless of how many
# local steps they took, which is critical under non-IID + quantity skew.
# ═════════════════════════════════════════════════════════════════════════════
class FedNovaAggregator:
    """
    FedNova normalized averaging aggregator.

    Each client's update is normalized by its number of local optimization
    steps (τ_k) before averaging, then the aggregated update is scaled by
    the effective number of steps (τ_eff).
    """

    def __init__(self):
        self.client_updates = []   # List of (pseudo_grad, tau_k, p_k)

    def clear(self):
        self.client_updates = []

    def add_client_update(self, w_global, w_local, tau_k, p_k):
        """
        Record one client's update.

        Args:
            w_global: global model weights before local training
            w_local: local model weights after training
            tau_k: number of local optimization steps for this client
            p_k: client weight (n_k / n_total)
        """
        # Pseudo-gradient: normalized update direction
        pseudo_grad = []
        for i in range(len(w_global)):
            # d_k = (w_global - w_local) / τ_k
            d_k = (w_global[i] - w_local[i]) / max(tau_k, 1)
            pseudo_grad.append(d_k)

        self.client_updates.append({
            'pseudo_grad': pseudo_grad,
            'tau_k': tau_k,
            'p_k': p_k,
        })

    def aggregate(self, w_global):
        """
        Compute FedNova aggregated weights.

        Returns:
            w_new: updated global weights
            tau_eff: effective number of steps (for logging)
        """
        if not self.client_updates:
            return w_global, 0

        num_layers = len(w_global)

        # Compute τ_eff = Σ(p_k * τ_k)
        tau_eff = sum(u['p_k'] * u['tau_k'] for u in self.client_updates)

        # Compute d_avg = Σ(p_k * d_k)
        d_avg = [np.zeros_like(w) for w in w_global]
        for u in self.client_updates:
            for i in range(num_layers):
                d_avg[i] += u['p_k'] * u['pseudo_grad'][i]

        # w_new = w_global - τ_eff * d_avg
        w_new = []
        for i in range(num_layers):
            w_new.append(w_global[i] - tau_eff * d_avg[i])

        return w_new, tau_eff


# ═════════════════════════════════════════════════════════════════════════════
# CONVERGENCE TRACKER
# ═════════════════════════════════════════════════════════════════════════════
class ConvergenceTracker:
    def __init__(self):
        self.rounds = []
        self.global_acc = []
        self.global_prec = []
        self.global_recall = []
        self.global_f1 = []
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
            'round': self.rounds, 'accuracy': self.global_acc,
            'precision': self.global_prec, 'recall': self.global_recall,
            'f1_score': self.global_f1,
        }).to_csv(filepath, index=False)


# ═════════════════════════════════════════════════════════════════════════════
# NON-IID DIRICHLET PARTITION
# ═════════════════════════════════════════════════════════════════════════════
def split_non_iid(data, labels, num_clients, alpha=0.5, min_samples=10):
    n_classes = len(np.unique(labels))
    N = len(labels)
    min_size = 0
    attempts = 0
    while min_size < min_samples:
        attempts += 1
        if attempts > 100: break
        client_indices = [[] for _ in range(num_clients)]
        for c in range(n_classes):
            idx_k = np.where(labels == c)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([
                p * (len(idx_j) < N / num_clients)
                for p, idx_j in zip(proportions, client_indices)])
            proportions = proportions / (proportions.sum() + 1e-12)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            splits = np.split(idx_k, proportions)
            for cid in range(num_clients):
                if cid < len(splits):
                    client_indices[cid].extend(splits[cid].tolist())
        min_size = min(len(idx) for idx in client_indices)
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    return client_indices


# ═════════════════════════════════════════════════════════════════════════════
# HYBRID ATTENTION
# ═════════════════════════════════════════════════════════════════════════════
@register_keras_serializable()
class SEBlock1D(Layer):
    def __init__(self, ratio=16, **kwargs):
        super(SEBlock1D, self).__init__(**kwargs)
        self.ratio = ratio
    def build(self, input_shape):
        self.filters = input_shape[-1]
        self.squeeze = GlobalAveragePooling1D()
        self.excitation = Dense(self.filters // self.ratio, activation='relu')
        self.excitation_2 = Dense(self.filters, activation='sigmoid')
    def call(self, inputs):
        squeeze = self.squeeze(inputs)
        excitation = self.excitation(squeeze)
        excitation = self.excitation_2(excitation)
        excitation = K.reshape(excitation, [-1, 1, self.filters])
        return inputs * excitation

def hybrid_attention_block(x, num_heads=4, key_dim=32):
    """Hybrid attention block: multi-head attention followed by squeeze-and-excitation"""
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=0.1)(x, x)
    attention_output = SEBlock1D()(attention_output)
    x = Add()([x, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    ffn = Dense(key_dim * 4, activation="relu")(x)
    ffn = Dropout(0.1)(ffn)
    ffn = Dense(K.int_shape(x)[-1])(ffn)
    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


# ═════════════════════════════════════════════════════════════════════════════
# MODEL CREATION
# ═════════════════════════════════════════════════════════════════════════════
def createHAFSSLModel(inp_size, n_classes):
    input_shape = (inp_size, 1)
    input_e = Input(shape=input_shape)
    x = Convolution1D(64, 3, padding="same", activation="relu", name='conv_1')(input_e)
    x = Convolution1D(64, 3, padding="same", activation="relu", name='conv_2')(x)
    x = MaxPooling1D(name='maxpool_1')(x)
    x = hybrid_attention_block(x, num_heads=4, key_dim=32)
    x = Convolution1D(32, 3, padding="same", activation="relu", name='conv_3')(x)
    x = Convolution1D(32, 3, padding="same", activation="relu", name='conv_4')(x)
    s_shape = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latent_dim, activation="relu")(x)
    encoder = Model(input_e, latent, name="encoder")

    input_d = Input(shape=(latent_dim,))
    x = Dense(int(np.prod(s_shape[1:])))(input_d)
    x = Reshape((s_shape[1], s_shape[2]))(x)
    x = Convolution1D(32, 3, padding="same", activation="relu", name='conv_5')(x)
    x = Convolution1D(32, 3, padding="same", activation="relu", name='conv_6')(x)
    x = UpSampling1D(2, name='upsampling_1d_2')(x)
    x = Convolution1D(64, 3, padding="same", activation="relu", name='conv_7')(x)
    output = Convolution1D(1, 3, padding="same", activation="relu", name='conv_8')(x)
    decoder = Model(input_d, output, name="decoder")

    input_c = Input(shape=(latent_dim,))
    x = Dense(int(np.prod(s_shape[1:])))(input_c)
    x = Reshape((s_shape[1], s_shape[2]))(x)
    x = Convolution1D(64, 3, padding="same", activation="relu")(x)
    x = Convolution1D(64, 3, padding="same", activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Convolution1D(128, 3, padding="same", activation="relu")(x)
    x = Convolution1D(128, 3, padding="same", activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(n_classes, activation="softmax")(x)
    cnn = Model(input_c, x, name="AEcnn")

    autoencoder = Model(input_e, decoder(encoder(input_e)))
    autoencoder.encoder = encoder
    classificationmodel = Model(input_e, cnn(encoder(input_e)))
    classificationmodel.encoder = encoder
    classificationmodel.cnn = cnn
    return autoencoder, classificationmodel


def splitLabel(x_train, y_train, labels=LABELS):
    label_indices = {}
    for label in range(len(labels)):
        label_indices[label] = np.where(y_train.argmax(axis=1) == label)[0]
    samples_per_class = labelnum // len(labels)
    x_train_labeled, y_train_labeled = [], []
    all_labeled_indices = []
    for label in range(len(labels)):
        if len(label_indices[label]) < samples_per_class:
            selected_indices = label_indices[label]
        else:
            selected_indices = np.random.choice(label_indices[label], samples_per_class, replace=False)
        x_train_labeled.append(x_train[selected_indices])
        y_train_labeled.append(y_train[selected_indices])
        all_labeled_indices.extend(selected_indices)
    x_train_labeled = np.concatenate(x_train_labeled, axis=0)
    y_train_labeled = np.concatenate(y_train_labeled, axis=0)
    unlabeled_indices = np.setdiff1d(np.arange(len(x_train)), all_labeled_indices)
    x_train_unlabeled = x_train[unlabeled_indices]
    return x_train_labeled, y_train_labeled, x_train_unlabeled


def updateClientsModels():
    global clientsAEModelList, clientsCNNModelist
    clientsAEModelList = []
    clientsCNNModelist = []
    for _ in range(numOfClients):
        ae = tf.keras.models.clone_model(originautoencoder)
        ae.set_weights(originautoencoder.get_weights())
        clientsAEModelList.append(ae)
        cnn = tf.keras.models.clone_model(originclassificationmodel)
        cnn.set_weights(originclassificationmodel.get_weights())
        clientsCNNModelist.append(cnn)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL PROFILING
# ═════════════════════════════════════════════════════════════════════════════
def profile_model(model, input_shape, name="Model"):
    total_params = model.count_params()
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    dummy = np.random.randn(1, *input_shape).astype(np.float32)
    _ = model.predict(dummy, verbose=0)
    times = []
    for _ in range(50):
        start = time.time()
        _ = model.predict(dummy, verbose=0)
        times.append(time.time() - start)
    avg_time = np.mean(times) * 1000
    model.save("/tmp/_temp_model.keras")
    size_mb = os.path.getsize("/tmp/_temp_model.keras") / (1024 * 1024)
    print(f"\n  {name}: {total_params:,} params | {avg_time:.1f}ms | {size_mb:.2f}MB")
    return {'name': name, 'total_params': total_params, 'trainable_params': trainable,
            'inference_ms': avg_time, 'model_size_mb': size_mb}


# ═════════════════════════════════════════════════════════════════════════════
# ██████  VISUALIZATION FUNCTIONS  ██████
# ═════════════════════════════════════════════════════════════════════════════
def plot_non_iid_distribution(yClientsList, labels, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    dist = np.zeros((len(yClientsList), len(labels)))
    for c in range(len(yClientsList)):
        cl = np.argmax(yClientsList[c], axis=1)
        for cls in range(len(labels)): dist[c, cls] = np.sum(cl == cls)
    dist_norm = dist / (dist.sum(axis=1, keepdims=True) + 1e-12)
    sns.heatmap(dist_norm, annot=True, fmt='.2f', cmap='YlOrRd', xticklabels=labels,
                yticklabels=[f'Client {i}' for i in range(len(yClientsList))], ax=ax)
    ax.set_title(f'Non-IID Distribution (α={non_iid_alpha})', fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.show()

def plot_non_iid_bar(yClientsList, labels, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(yClientsList))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for cls in range(len(labels)):
        counts = [np.sum(np.argmax(yClientsList[c], axis=1) == cls) for c in range(len(yClientsList))]
        ax.bar([f'C{i}' for i in range(len(yClientsList))], counts, bottom=bottom,
               label=labels[cls], color=colors[cls], edgecolor='white')
        bottom += np.array(counts)
    ax.set_ylabel('Samples'); ax.set_title(f'Non-IID Samples (α={non_iid_alpha})', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.show()

def plot_convergence(tracker, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tracker.rounds, tracker.global_acc, 'o-', color='#1976D2', markersize=4, label='Accuracy')
    ax.plot(tracker.rounds, tracker.global_f1, 's-', color='#E53935', markersize=4, label='F1-Score')
    ax.plot(tracker.rounds, tracker.global_prec, '^-', color='#43A047', markersize=3, alpha=0.7, label='Precision')
    ax.plot(tracker.rounds, tracker.global_recall, 'v-', color='#FF9800', markersize=3, alpha=0.7, label='Recall')
    ax.fill_between(tracker.rounds, tracker.global_acc, alpha=0.1, color='#1976D2')
    ax.set_xlabel('FL Round'); ax.set_ylabel('Score')
    ax.set_title(f'{MODEL_NAME} — Convergence (non-IID α={non_iid_alpha})', fontweight='bold')
    ax.set_ylim([0, 1.05]); ax.legend(loc='lower right')
    if tracker.global_acc:
        ax.annotate(f'{tracker.global_acc[-1]:.3f}', xy=(tracker.rounds[-1], tracker.global_acc[-1]),
                    xytext=(-60, 10), textcoords='offset points', fontsize=11, fontweight='bold',
                    color='#1976D2', arrowprops=dict(arrowstyle='->', color='#1976D2'))
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.show()

def plot_per_client(tracker, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, numOfClients))
    for c in range(numOfClients):
        accs = [tracker.per_client_acc[r][c] for r in tracker.rounds]
        ax.plot(tracker.rounds, accs, '-', color=colors[c], linewidth=1.5, alpha=0.8, label=f'Client {c}')
    ax.plot(tracker.rounds, tracker.global_acc, 'k--', linewidth=2.5, label='Global', zorder=10)
    ax.set_xlabel('FL Round'); ax.set_ylabel('Accuracy')
    ax.set_title(f'{MODEL_NAME} — Per-Client Accuracy', fontweight='bold')
    ax.set_ylim([0, 1.05]); ax.legend(loc='lower right', fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.show()

def plot_final_bars(acc_l, prec_l, rec_l, f1_l, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(acc_l)); w = 0.2
    ax.bar(x-1.5*w, acc_l, w, color='#2196F3', label='Acc')
    ax.bar(x-0.5*w, prec_l, w, color='#4CAF50', label='Prec')
    ax.bar(x+0.5*w, rec_l, w, color='#FF9800', label='Rec')
    ax.bar(x+1.5*w, f1_l, w, color='#E53935', label='F1')
    ax.set_xticks(x); ax.set_xticklabels([f'C{i}' for i in range(len(acc_l))])
    ax.set_ylabel('Score'); ax.set_title(f'{MODEL_NAME} — Final Metrics', fontweight='bold')
    ax.set_ylim([0, 1.15]); ax.legend()
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.show()

def plot_confusion(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, annot=True, fmt='d', square=True, cmap='Blues', xticklabels=labels, yticklabels=labels, ax=a1)
    a1.set_title('Counts', fontweight='bold'); a1.tick_params(axis='x', rotation=45)
    sns.heatmap(cm_n, annot=True, fmt='.1%', square=True, cmap='YlOrRd', xticklabels=labels, yticklabels=labels, ax=a2)
    a2.set_title('Normalized', fontweight='bold'); a2.tick_params(axis='x', rotation=45)
    plt.suptitle(f'{MODEL_NAME} — Confusion Matrix', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.show()

def plot_complexity(profiles, save_path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    names = [p['name'] for p in profiles]; cols = plt.cm.Set2(np.linspace(0, 1, len(profiles)))
    a1.bar(names, [p['total_params']/1000 for p in profiles], color=cols)
    a1.set_ylabel('Params (K)'); a1.set_title('Complexity', fontweight='bold')
    a2.bar(names, [p['inference_ms'] for p in profiles], color=cols)
    a2.set_ylabel('Inference (ms)'); a2.set_title('Latency', fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.show()

def plot_dashboard(tracker, acc_list, recon_err, profiles, tau_effs, save_path):
    """Dashboard with FedNova-specific τ_eff tracking."""
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(tracker.rounds, tracker.global_acc, 'o-', color='#1976D2', markersize=3, label='Acc')
    ax1.plot(tracker.rounds, tracker.global_f1, 's-', color='#E53935', markersize=3, label='F1')
    ax1.set_xlabel('Round'); ax1.set_ylabel('Score'); ax1.set_title('Convergence', fontweight='bold')
    ax1.set_ylim([0, 1.05]); ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    colors = plt.cm.RdYlGn(np.array(acc_list))
    bars = ax2.barh([f'C{i}' for i in range(len(acc_list))], acc_list, color=colors)
    for b, a in zip(bars, acc_list):
        ax2.text(b.get_width()+0.01, b.get_y()+b.get_height()/2, f'{a:.3f}', va='center', fontsize=10, fontweight='bold')
    ax2.set_xlim([0, 1.15]); ax2.set_title('Per-Client Acc', fontweight='bold')

    ax3 = fig.add_subplot(gs[0, 2]); ax3.axis('off')
    txt = (f"{MODEL_NAME}\nNon-IID α={non_iid_alpha}\nLabels={labelnum}\nClients={numOfClients}\n"
           f"Rounds={numOfIterations}\n\nFinal Acc: {tracker.global_acc[-1]:.4f}\n"
           f"Final F1: {tracker.global_f1[-1]:.4f}\nRecon Err: {recon_err:.4f}\n\n"
           f"Params: {profiles[0]['total_params']:,}\nInference: {profiles[0]['inference_ms']:.1f}ms\n"
           f"Avg τ_eff: {np.mean(tau_effs):.1f}")
    ax3.text(0.5, 0.5, txt, ha='center', va='center', fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2))
    ax3.set_title('Summary', fontweight='bold')

    ax4 = fig.add_subplot(gs[1, 0])
    vars_ = [np.std(tracker.per_client_acc[r]) for r in tracker.rounds]
    ax4.plot(tracker.rounds, vars_, '-', color='#9C27B0'); ax4.fill_between(tracker.rounds, vars_, alpha=0.2, color='#9C27B0')
    ax4.set_xlabel('Round'); ax4.set_ylabel('Std Dev'); ax4.set_title('Client Variance', fontweight='bold')

    # FedNova-specific: τ_eff over rounds
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(range(1, len(tau_effs)+1), tau_effs, 'o-', color='#7B1FA2', markersize=4)
    ax5.set_xlabel('Round'); ax5.set_ylabel('τ_eff')
    ax5.set_title('Effective Local Steps (τ_eff)', fontweight='bold')
    ax5.axhline(y=np.mean(tau_effs), color='red', linestyle='--', alpha=0.5, label=f'Mean={np.mean(tau_effs):.1f}')
    ax5.legend(fontsize=9)

    ax6 = fig.add_subplot(gs[1, 2]); ax6.axis('off')
    bpr = sum(p['model_size_mb'] for p in profiles) * numOfClients * 2
    r90 = next((r for r, a in zip(tracker.rounds, tracker.global_acc) if a >= 0.90), 'N/A')
    ctxt = (f"FedNova Comm Cost\n(same as FedAvg — no extra state)\n\n"
            f"Per round: {bpr:.1f} MB\n({numOfClients} clients × up+down)\n\n"
            f"Rounds to 90%: {r90}\nTotal: {numOfIterations}\nTotal comm: {bpr*numOfIterations:.0f} MB\n\n"
            f"vs SCAFFOLD: NO control variates\n→ 1× comm cost (not 2×)")
    ax6.text(0.5, 0.5, ctxt, ha='center', va='center', fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='#E8F5E9', edgecolor='#43A047', linewidth=2))
    ax6.set_title('Communication', fontweight='bold')

    plt.suptitle(f'{MODEL_NAME} — Non-IID Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(save_path, bbox_inches='tight'); plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# MONITORING
# ═════════════════════════════════════════════════════════════════════════════
monitorheaders = ['stage','iterationNo','clientID','avg_GPU_mem','avg_GPU_load','avg_Memory_used','avg_cpu_used','used_time(us)']
Globalmonitordirct = {h: 0 for h in monitorheaders}; Globalmonitordirct['stage'] = ''
Globalmonitordirctrows = []
performanceheaders = ['stage','iterationNo','clientID','train_acc','val_acc','test_acc','classification_report']
performancerdirct = {h: 0 for h in performanceheaders}; performancerdirct['stage'] = ''; performancerdirct['classification_report'] = ''
performancerdirctros = []

class Monitor(Thread):
    def __init__(self, delay, stage, iterationNo, clientID, process):
        super(Monitor, self).__init__()
        self.stopped = False; self.delay = delay; self.stage = stage
        self.iterationNo = iterationNo; self.clientID = clientID; self.process = process
        self.gpu_mem_list = []; self.gpu_load_list = []; self.used_mem_list = []; self.cpu_load_list = []
        self.start()
    def run(self):
        st = datetime.datetime.now()
        while not self.stopped:
            try:
                Gpus = GPUtil.getGPUs()
                if Gpus: self.gpu_mem_list.append(Gpus[0].memoryUtil*100); self.gpu_load_list.append(Gpus[0].load*100)
                else: self.gpu_mem_list.append(0); self.gpu_load_list.append(0)
            except: self.gpu_mem_list.append(0); self.gpu_load_list.append(0)
            self.used_mem_list.append(self.process.memory_percent(memtype="uss"))
            self.cpu_load_list.append(self.process.cpu_percent(interval=1)); time.sleep(self.delay)
        et = datetime.datetime.now()
        Globalmonitordirct.update({'stage':self.stage,'iterationNo':self.iterationNo,'clientID':self.clientID,
            'avg_GPU_mem':np.mean(self.gpu_mem_list),'avg_GPU_load':np.mean(self.gpu_load_list),
            'avg_Memory_used':np.mean(self.used_mem_list),'avg_cpu_used':np.mean(self.cpu_load_list),
            'used_time(us)':(et-st).microseconds*23})
        Globalmonitordirctrows.append(Globalmonitordirct.copy())
    def stop(self): self.stopped = True


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    drive.mount('/content/drive')
    gdrive_file_url = 'https://drive.google.com/uc?id=1_g02J9Dzel490fU7N4rD7fNwp4vyFxwo'
    # Public alternative (same file, released by Wang et al.):
    # https://github.com/PrinceXuan12138/HGW-TC-Experimental-code/releases/download/v1.0.0/ISCX_5class_each_normalized_cuttedflowfeature.csv
    dataset_path = 'ISCX_5class_each_normalized_cuttedflowfeature.csv'
    gdown.download(gdrive_file_url, dataset_path, quiet=False)

    dfDS = pd.read_csv(dataset_path)
    X_full = dfDS.iloc[:, 1:len(dfDS.columns)].values
    Y_full = dfDS["label"].values
    num_classes = len(set(Y_full))
    Y_full = tf.keras.utils.to_categorical(Y_full, num_classes)

    xServer, xClients, yServer, yClients = train_test_split(X_full, Y_full, test_size=0.90, random_state=523)
    xServer = np.expand_dims(xServer, axis=2)
    inp_size = xServer.shape[1]

    originautoencoder, originclassificationmodel = createHAFSSLModel(inp_size, num_classes)

    # Profiling
    print(f"\n{'='*60}\n  MODEL PROFILING\n{'='*60}")
    profiles = [
        profile_model(originautoencoder, (inp_size, 1), "Autoencoder"),
        profile_model(originclassificationmodel, (inp_size, 1), "Classifier"),
    ]

    clientsAEModelList = []; clientsCNNModelist = []
    updateClientsModels()
    originautoencoder.save(AEmodelLocation)
    originclassificationmodel.save(CNNmodelLocation)

    # Non-IID partition
    print(f"\n{'='*60}\n  NON-IID PARTITION — α={non_iid_alpha}\n{'='*60}")
    yClients_int = np.argmax(yClients, axis=1)
    client_indices = split_non_iid(xClients, yClients_int, numOfClients, alpha=non_iid_alpha)

    xClientsList, yClientsList = [], []
    for cid in range(numOfClients):
        xClientsList.append(xClients[client_indices[cid]])
        yClientsList.append(yClients[client_indices[cid]])
        clientsAEModelList.append(load_model(AEmodelLocation))
        clientsCNNModelist.append(load_model(CNNmodelLocation))
        cl = np.argmax(yClientsList[cid], axis=1)
        print(f"  Client {cid}: {len(client_indices[cid])} — " +
              ", ".join(f"{LABELS[i]}:{np.sum(cl==i)}" for i in range(len(LABELS))))

    plot_non_iid_distribution(yClientsList, LABELS, f"{FIGURE_DIR}/{MODEL_NAME}_01_nonIID_heatmap.png")
    plot_non_iid_bar(yClientsList, LABELS, f"{FIGURE_DIR}/{MODEL_NAME}_02_nonIID_bars.png")

    # Split labeled/unlabeled
    xClientsListLabel, xClientsListUnLabel, yClientsListLabel = [], [], []
    client_sample_sizes = []
    for cid in range(numOfClients):
        xl, yl, xu = splitLabel(xClientsList[cid], yClientsList[cid])
        xClientsListLabel.append(np.expand_dims(xl, axis=2))
        xClientsListUnLabel.append(np.expand_dims(xu, axis=2))
        yClientsListLabel.append(yl)
        client_sample_sizes.append(len(xl) + len(xu))

    total_samples = sum(client_sample_sizes)
    print(f"\n  Sample sizes: {client_sample_sizes}")
    print(f"  Total: {total_samples}")

    # ── Initialize FedNova aggregators and tracker ─────────────────────────
    fednova_ae = FedNovaAggregator()
    fednova_cnn = FedNovaAggregator()
    tracker = ConvergenceTracker()
    tau_eff_history = []  # Track effective steps per round

    start_time = time.time()
    process = psutil.Process(os.getpid())

    print(f"\n{'='*60}\n  {MODEL_NAME} TRAINING — non-IID α={non_iid_alpha}")
    print(f"  Clients: {numOfClients} | Rounds: {numOfIterations} | Labels: {labelnum}")
    print(f"  Aggregation: Normalized averaging (τ_k normalization)\n{'='*60}")

    for iterationNo in range(1, numOfIterations + 1):
        print(f"\n{'='*60}\n  Round {iterationNo}/{numOfIterations} (FedNova)\n{'='*60}")
        round_client_accs = []
        fednova_ae.clear()
        fednova_cnn.clear()

        # Save global weights before any client trains
        w_global_ae = [w.copy() for w in originautoencoder.get_weights()]
        w_global_cnn = [w.copy() for w in originclassificationmodel.get_weights()]

        for clientID in range(numOfClients):
            monitor = Monitor(1, "HAFSSL-FedNova training", iterationNo, clientID, process)
            subAEmodel = originautoencoder
            subCNNmodel = originclassificationmodel
            subAEmodel.set_weights(clientsAEModelList[clientID].get_weights())
            subCNNmodel.set_weights(clientsCNNModelist[clientID].get_weights())

            # ══════════════════════════════════════════════════════════════
            # Stage 1: AE unsupervised (standard — no FedNova modification)
            # FedNova only changes AGGREGATION, not local training
            # ══════════════════════════════════════════════════════════════
            subAEmodel.compile(loss='mse', optimizer='adam', metrics=['mse'])
            subAEmodel.fit(xClientsListUnLabel[clientID], xClientsListUnLabel[clientID],
                           epochs=epochs, shuffle=True, validation_data=(xServer, xServer),
                           verbose=verbose)

            subCNNmodel.encoder.set_weights(subAEmodel.encoder.get_weights())

            # Stage 2: CNN supervised (standard)
            subCNNmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            history = subCNNmodel.fit(xClientsListLabel[clientID], yClientsListLabel[clientID],
                                      epochs=epochs, batch_size=batch_size, shuffle=True,
                                      validation_data=(xServer, yServer), verbose=verbose)
            monitor.stop()

            # ══════════════════════════════════════════════════════════════
            # FedNova: Compute local steps τ_k and client weight p_k
            # τ_k = total optimizer steps = epochs × ceil(samples / batch_size)
            # p_k = n_k / n_total (dataset-size weighting)
            # ══════════════════════════════════════════════════════════════
            n_ae = len(xClientsListUnLabel[clientID])
            n_cnn = len(xClientsListLabel[clientID])
            tau_ae = epochs * max(1, int(np.ceil(n_ae / batch_size)))
            tau_cnn = epochs * max(1, int(np.ceil(n_cnn / batch_size)))
            p_k = client_sample_sizes[clientID] / total_samples

            # Record normalized updates
            fednova_ae.add_client_update(
                w_global=w_global_ae,
                w_local=subAEmodel.get_weights(),
                tau_k=tau_ae,
                p_k=p_k
            )
            fednova_cnn.add_client_update(
                w_global=w_global_cnn,
                w_local=subCNNmodel.get_weights(),
                tau_k=tau_cnn,
                p_k=p_k
            )

            # Evaluate
            y_pr = subCNNmodel.predict(xServer, batch_size=300)
            report = classification_report(yServer.argmax(1), y_pr.argmax(1),
                                           target_names=LABELS, zero_division=1, output_dict=True)
            round_client_accs.append(report['accuracy'])
            print(f"  Client {clientID} (τ_ae={tau_ae}, τ_cnn={tau_cnn}, p={p_k:.3f}) — "
                  f"Acc: {report['accuracy']:.4f} | F1: {report['weighted avg']['f1-score']:.4f}")

            performancerdirct.update({'stage': 'HAFSSL-FedNova training', 'iterationNo': iterationNo,
                'clientID': clientID, 'train_acc': history.history["accuracy"][-1],
                'val_acc': history.history["val_accuracy"][-1], 'test_acc': report['accuracy'],
                'classification_report': report})
            performancerdirctros.append(performancerdirct.copy())
            subCNNmodel.save(f"./Models/AECNNmodel/CNN_node_{clientID}.keras")
            subAEmodel.save(f"./Models/AECNNmodel/AE_node_{clientID}.keras")

        # ══════════════════════════════════════════════════════════════════
        # FedNova AGGREGATION (normalized averaging)
        # Instead of: w_new = Σ(p_k * w_k)          ← FedAvg
        # FedNova:    w_new = w_g - τ_eff * Σ(p_k * d_k)
        #             where d_k = (w_g - w_k) / τ_k
        # ══════════════════════════════════════════════════════════════════
        print(f"\n  FedNova Aggregation (normalized by local steps)...")

        new_ae_weights, tau_eff_ae = fednova_ae.aggregate(w_global_ae)
        originautoencoder.set_weights(new_ae_weights)
        originautoencoder.save(AEmodelLocation)

        new_cnn_weights, tau_eff_cnn = fednova_cnn.aggregate(w_global_cnn)
        originclassificationmodel.set_weights(new_cnn_weights)
        originclassificationmodel.save(CNNmodelLocation)

        tau_eff_history.append(tau_eff_cnn)
        print(f"  τ_eff (AE): {tau_eff_ae:.1f} | τ_eff (CNN): {tau_eff_cnn:.1f}")

        # Evaluate global model
        y_gpr = originclassificationmodel.predict(xServer, batch_size=300)
        gr = classification_report(yServer.argmax(1), y_gpr.argmax(1),
                                    target_names=LABELS, zero_division=1, output_dict=True)
        tracker.record_round(iterationNo, gr['accuracy'], gr['weighted avg']['precision'],
                              gr['weighted avg']['recall'], gr['weighted avg']['f1-score'],
                              round_client_accs)
        print(f"\n  Round {iterationNo} Global — Acc: {gr['accuracy']:.4f} | "
              f"F1: {gr['weighted avg']['f1-score']:.4f}")

        updateClientsModels()

    # Final evaluation
    total_time = time.time() - start_time
    print(f"\n{'='*60}\n  FINAL EVALUATION — {MODEL_NAME}\n  Time: {total_time/60:.1f} min\n{'='*60}")
    fa, fp, fr, ff = [], [], [], []
    for cid in range(numOfClients):
        nm = originclassificationmodel
        nm.set_weights(load_model(f"./Models/AECNNmodel/CNN_node_{cid}.keras").get_weights())
        yp = nm.predict(xServer, batch_size=100)
        r = classification_report(yServer.argmax(1), yp.argmax(1), target_names=LABELS, zero_division=1, output_dict=True)
        fa.append(r['accuracy']); fp.append(r['weighted avg']['precision'])
        fr.append(r['weighted avg']['recall']); ff.append(r['weighted avg']['f1-score'])
        print(f"  Client {cid} — Acc: {r['accuracy']:.4f} | F1: {r['weighted avg']['f1-score']:.4f}")

    recon_err = np.mean(np.square(xServer - originautoencoder.predict(xServer, verbose=0)))
    print(f"\n  Avg Acc: {np.mean(fa):.4f} | Avg F1: {np.mean(ff):.4f} | Recon: {recon_err:.4f}")
    print(f"  Avg τ_eff: {np.mean(tau_eff_history):.1f}")

    # All visualizations
    print(f"\n{'='*60}\n  GENERATING VISUALIZATIONS\n{'='*60}")
    plot_convergence(tracker, f"{FIGURE_DIR}/{MODEL_NAME}_03_convergence.png")
    plot_per_client(tracker, f"{FIGURE_DIR}/{MODEL_NAME}_04_per_client.png")
    plot_final_bars(fa, fp, fr, ff, f"{FIGURE_DIR}/{MODEL_NAME}_05_final_bars.png")
    yf = originclassificationmodel.predict(xServer, batch_size=300)
    plot_confusion(yServer.argmax(1), yf.argmax(1), LABELS, f"{FIGURE_DIR}/{MODEL_NAME}_06_confusion.png")
    plot_complexity(profiles, f"{FIGURE_DIR}/{MODEL_NAME}_07_complexity.png")
    plot_dashboard(tracker, fa, recon_err, profiles, tau_eff_history,
                   f"{FIGURE_DIR}/{MODEL_NAME}_08_dashboard.png")

    # Save CSVs
    tracker.save_csv(f"./result/{MODEL_NAME}_convergence_{labelnum}label.csv")
    pd.DataFrame([{'model': MODEL_NAME, 'setting': f'non-IID α={non_iid_alpha}',
        'labels': labelnum, 'clients': numOfClients, 'rounds': numOfIterations,
        'avg_acc': np.mean(fa), 'avg_f1': np.mean(ff), 'recon_err': recon_err,
        'time_min': total_time/60, 'params': profiles[0]['total_params'],
        'inference_ms': profiles[1]['inference_ms'],
        'avg_tau_eff': np.mean(tau_eff_history)}]).to_csv(
        f"./result/{MODEL_NAME}_nonIID_summary_{labelnum}label.csv", index=False)

    with open(monitoring_filename, 'a+', newline='') as f:
        w = csv.DictWriter(f, monitorheaders); w.writeheader(); w.writerows(Globalmonitordirctrows)
    with open(performance_filename, 'a+', newline='') as f:
        w = csv.DictWriter(f, performanceheaders); w.writeheader(); w.writerows(performancerdirctros)

    print(f"\n{'='*60}\n  COMPLETE — {MODEL_NAME} non-IID")
    print(f"  Acc: {np.mean(fa):.4f} | F1: {np.mean(ff):.4f}")
    print(f"  Avg τ_eff: {np.mean(tau_eff_history):.1f}")
    print(f"  8 figures saved to: {FIGURE_DIR}/\n{'='*60}")
