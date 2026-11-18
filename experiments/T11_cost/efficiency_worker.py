# -*- coding: utf-8 -*-
# =====================================================================================
#  EFFICIENCY TABLE (Reviewer 1, Comment 4) — ONE KAGGLE CELL
#  Accelerator: GPU T4 (x1 is enough). Paste this whole file into one cell -> Save & Run All.
#  Takes about 10-15 minutes. No dataset needed (the models only need the input size).
#
#  Measures, for HAFSSL/SynAttnNet and FL-AECNN/AECNN (same models as all R2 runs):
#    parameters (encoder, decoder, head, inference path, sent per FL round), weight size (float32),
#    MACs and FLOPs per flow (layer-by-layer count, cross-checked with the TensorFlow profiler),
#    latency per flow at batch 1 (median and 95th percentile, after warm-up, compiled tf.function),
#    throughput at batch 1024, on: GPU (T4), CPU with 1 thread, CPU with all threads,
#    communication per client and per round (K = 10, float32, upload + download),
#    and Keras model.predict() at batch 1 (this is how the OLD 88-114 ms values were measured).
#  Output: /kaggle/working/efficiency/efficiency_table.csv  (+ one JSON per device)
# =====================================================================================
import os, sys, json, subprocess
import pandas as pd

CONFIGS = [["ISCX", 66, 5], ["Apps10", 66, 10]]
# When the USTC-TFC2016 input size is known, add e.g.:  ["USTC", 128, 20]
OUT = "/kaggle/working/efficiency"
os.makedirs(OUT, exist_ok=True)
WORKER = os.path.join(OUT, "efficiency_worker.py")

WORKER_SCRIPT = r'''# -*- coding: utf-8 -*-
"""Efficiency measurements for SynAttnNet/HAFSSL and AECNN/FL-AECNN (same architecture as all R2 runs).
Run by the one-cell launcher once per device setting (GPU, CPU 1 thread, CPU all threads)."""
import os, sys, json, time, platform
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

DEVICE     = os.environ.get("EFF_DEVICE", "GPU")            # GPU | CPU1 | CPUALL
OUT_JSON   = os.environ.get("EFF_OUT", "/tmp/eff.json")
CONFIGS    = json.loads(os.environ.get("EFF_CONFIGS", '[["ISCX", 66, 5], ["Apps10", 66, 10]]'))
NUM_CLIENTS = int(os.environ.get("EFF_K", "10"))
N_TIMED, N_WARM, BIG_BATCH = 1000, 100, 1024
if DEVICE == "CPU1":
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

LATENT_DIM, NUM_HEADS, KEY_DIM, SE_RATIO = 39, 4, 32, 16


# ---------------- model: identical to the R2 training scripts ----------------
def se_block(x, ratio=SE_RATIO):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(ch // ratio, activation='relu')(s)
    s = layers.Dense(ch, activation='sigmoid')(s)
    return layers.Multiply()([x, layers.Reshape((1, ch))(s)])


def attention_block(x, variant):
    if variant == "none":
        return x
    a = se_block(layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM, dropout=0.1)(x, x))
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, a]))
    f = layers.Dense(KEY_DIM * 4, activation='relu')(x)
    f = layers.Dense(x.shape[-1])(layers.Dropout(0.1)(f))
    return layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, f]))


def build_models(n_features, n_classes, variant):
    ie = layers.Input(shape=(n_features, 1))
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(ie)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D()(x)
    x = attention_block(x, variant)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    steps, ch = x.shape[1], x.shape[2]
    enc = Model(ie, layers.Dense(LATENT_DIM, activation='relu')(layers.Flatten()(x)), name='encoder')
    idd = layers.Input(shape=(LATENT_DIM,))
    d = layers.Reshape((steps, ch))(layers.Dense(steps * ch)(idd))
    d = layers.Conv1D(32, 3, padding='same', activation='relu')(d)
    d = layers.Conv1D(32, 3, padding='same', activation='relu')(d)
    d = layers.UpSampling1D()(d)
    d = layers.Conv1D(64, 3, padding='same', activation='relu')(d)
    dec = Model(idd, layers.Conv1D(1, 3, padding='same', activation='relu')(d), name='decoder')
    ic = layers.Input(shape=(LATENT_DIM,))
    c = layers.Reshape((steps, ch))(layers.Dense(steps * ch)(ic))
    c = layers.Conv1D(64, 3, padding='same', activation='relu')(c)
    c = layers.Conv1D(64, 3, padding='same', activation='relu')(c)
    c = layers.MaxPooling1D(2)(c)
    c = layers.Conv1D(128, 3, padding='same', activation='relu')(c)
    c = layers.Conv1D(128, 3, padding='same', activation='relu')(c)
    c = layers.MaxPooling1D(2)(c)
    c = layers.Dense(128, activation='relu')(layers.Flatten()(c))
    head = Model(ic, layers.Dense(n_classes, activation='softmax')(c), name='classifier_head')
    inp = layers.Input(shape=(n_features, 1))
    clf = Model(inp, head(enc(inp)), name='inference')
    return enc, dec, head, clf


# ---------------- multiply-accumulate (MAC) count per flow, layer by layer ----------------
def _shape(t):
    return [int(s) for s in t.shape[1:]]


def count_macs(model):
    """MACs for ONE flow (batch 1). Counts Conv1D, Dense (incl. per-position), MultiHeadAttention
    (Q/K/V/output projections + QK^T + attention x V) and the SE rescaling multiply.
    Element-wise ops (ReLU, add, LayerNorm, softmax, pooling) are not counted, as usual."""
    total, detail = 0, {}
    for layer in model.layers:
        if isinstance(layer, Model):
            sub, sub_detail = count_macs(layer)
            total += sub
            for k, v in sub_detail.items():
                detail[k] = detail.get(k, 0) + v
            continue
        name = type(layer).__name__
        macs = 0
        if isinstance(layer, layers.Conv1D):
            out_len, cout = _shape(layer.output)
            cin = _shape(layer.input)[-1]
            macs = out_len * layer.kernel_size[0] * cin * cout
        elif isinstance(layer, layers.Dense):
            in_shape = _shape(layer.input)
            positions = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            macs = positions * in_shape[-1] * layer.units
        elif isinstance(layer, layers.MultiHeadAttention):
            L, d_model = _shape(layer.output)
            h, dk = layer.num_heads, layer.key_dim
            dv = layer.value_dim or dk
            macs = (L * d_model * h * dk * 2          # Q and K projections
                    + L * d_model * h * dv            # V projection
                    + h * L * L * dk                  # Q K^T
                    + h * L * L * dv                  # attention x V
                    + L * h * dv * d_model)           # output projection
        elif isinstance(layer, layers.Multiply):
            macs = int(np.prod(_shape(layer.output)))
        if macs:
            total += macs
            detail[name] = detail.get(name, 0) + macs
    return total, detail


def profiler_flops(model, n_features):
    """TensorFlow profiler FLOPs (cross-check; it may not count Einsum ops used by attention)."""
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    f = tf.function(lambda x: model(x, training=False))
    cf = f.get_concrete_function(tf.TensorSpec([1, n_features, 1], tf.float32))
    gd = convert_variables_to_constants_v2(cf).graph.as_graph_def()
    with tf.Graph().as_default() as g:
        tf.graph_util.import_graph_def(gd, name='')
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts['output'] = 'none'
        prof = tf.compat.v1.profiler.profile(graph=g, run_meta=tf.compat.v1.RunMetadata(), cmd='op', options=opts)
    return int(prof.total_float_ops)


# ---------------- latency ----------------
def timed(fn, x, n):
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn(x).numpy()                      # .numpy() waits until the result is ready (GPU sync)
        out.append(time.perf_counter() - t)
    return np.array(out)


def measure_latency(model, n_features):
    infer = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    x1 = tf.constant(np.random.rand(1, n_features, 1).astype("float32"))
    xb = tf.constant(np.random.rand(BIG_BATCH, n_features, 1).astype("float32"))
    timed(infer, x1, N_WARM)
    t1 = timed(infer, x1, N_TIMED)
    timed(infer, xb, 20)
    tb = timed(infer, xb, 100)
    xp = np.random.rand(1, n_features, 1).astype("float32")      # what the old scripts measured
    for _ in range(10):
        model.predict(xp, verbose=0)
    tp = []
    for _ in range(50):
        t = time.perf_counter(); model.predict(xp, verbose=0); tp.append(time.perf_counter() - t)
    return dict(lat_b1_median_ms=1e3 * float(np.median(t1)), lat_b1_p95_ms=1e3 * float(np.percentile(t1, 95)),
                throughput_flows_per_s=BIG_BATCH / float(np.median(tb)),
                per_flow_at_b1024_us=1e6 * float(np.median(tb)) / BIG_BATCH,
                keras_predict_b1_median_ms=1e3 * float(np.median(tp)))


def main():
    if DEVICE == "GPU":
        gpus = tf.config.list_physical_devices('GPU')
        hw = tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'GPU') if gpus else "NO GPU"
    else:
        hw = f"{platform.processor() or platform.machine()} | {os.cpu_count()} cores | " + \
             ("1 thread" if DEVICE == "CPU1" else "all threads")
    rows = []
    for ds, n_feat, n_cls in CONFIGS:
        for model_name, variant in [("HAFSSL / SynAttnNet", "MHA->SE"), ("FL-AECNN / AECNN", "none")]:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(0)
            enc, dec, head, clf = build_models(n_feat, n_cls, variant)
            p_enc, p_dec, p_head = enc.count_params(), dec.count_params(), head.count_params()
            row = dict(dataset=ds, n_features=n_feat, n_classes=n_cls, model=model_name, device=DEVICE, hardware=hw,
                       params_encoder=p_enc, params_decoder=p_dec, params_head=p_head,
                       params_inference=p_enc + p_head, params_AE_pretrain=p_enc + p_dec,
                       params_sent_per_round=p_enc + p_dec + p_head,
                       weights_MB_inference=4 * (p_enc + p_head) / 1e6,
                       comm_MB_per_client_round=2 * 4 * (p_enc + p_dec + p_head) / 1e6,
                       comm_MB_per_round_K=NUM_CLIENTS * 2 * 4 * (p_enc + p_dec + p_head) / 1e6)
            if DEVICE == "GPU":                          # device-independent numbers: compute once
                macs, detail = count_macs(clf)
                row.update(MACs_inference=macs, FLOPs_inference=2 * macs,
                           MACs_attention=detail.get("MultiHeadAttention", 0),
                           FLOPs_tf_profiler=profiler_flops(clf, n_feat))
            row.update(measure_latency(clf, n_feat))
            rows.append(row)
            print(f"{DEVICE:6s} {ds:7s} {model_name:20s} params(inf) {row['params_inference']:,} | "
                  f"b1 {row['lat_b1_median_ms']:.3f} ms | {row['throughput_flows_per_s']:,.0f} flows/s", flush=True)
    with open(OUT_JSON, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
'''

with open(WORKER, "w") as f:
    f.write(WORKER_SCRIPT)

runs = [("GPU", None), ("CPU1", ""), ("CPUALL", "")]          # None = keep GPU visible; "" = hide GPU
for dev, cuda in runs:
    env = dict(os.environ, EFF_DEVICE=dev, EFF_OUT=os.path.join(OUT, f"eff_{dev}.json"),
               EFF_CONFIGS=json.dumps(CONFIGS))
    if cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda
    print(f"\n----- measuring on {dev} -----", flush=True)
    r = subprocess.run([sys.executable, WORKER], env=env, capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith(("GPU", "CPU"))))
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise RuntimeError(f"{dev} measurement failed")

g = pd.DataFrame(json.load(open(os.path.join(OUT, "eff_GPU.json"))))
c1 = pd.DataFrame(json.load(open(os.path.join(OUT, "eff_CPU1.json"))))
ca = pd.DataFrame(json.load(open(os.path.join(OUT, "eff_CPUALL.json"))))
key = ["dataset", "model"]
t = g[key + ["n_features", "n_classes", "hardware", "params_encoder", "params_decoder", "params_head",
             "params_inference", "params_AE_pretrain", "params_sent_per_round", "weights_MB_inference",
             "MACs_inference", "FLOPs_inference", "MACs_attention", "FLOPs_tf_profiler",
             "comm_MB_per_client_round", "comm_MB_per_round_K",
             "lat_b1_median_ms", "lat_b1_p95_ms", "throughput_flows_per_s", "keras_predict_b1_median_ms"]]
t = t.rename(columns={"hardware": "gpu", "lat_b1_median_ms": "GPU_b1_median_ms", "lat_b1_p95_ms": "GPU_b1_p95_ms",
                      "throughput_flows_per_s": "GPU_flows_per_s", "keras_predict_b1_median_ms": "GPU_keras_predict_b1_ms"})
for df, tag in [(c1, "CPU1"), (ca, "CPUall")]:
    d = df[key + ["hardware", "lat_b1_median_ms", "lat_b1_p95_ms", "throughput_flows_per_s"]].rename(columns={
        "hardware": f"{tag}_hw", "lat_b1_median_ms": f"{tag}_b1_median_ms", "lat_b1_p95_ms": f"{tag}_b1_p95_ms",
        "throughput_flows_per_s": f"{tag}_flows_per_s"})
    t = t.merge(d, on=key)
t.to_csv(os.path.join(OUT, "efficiency_table.csv"), index=False)

pd.set_option("display.width", 250)
print("\n" + "=" * 90 + "\nEFFICIENCY TABLE (per flow; latency = compiled model call after warm-up)\n" + "=" * 90)
show = t[["dataset", "model", "params_inference", "params_sent_per_round", "weights_MB_inference", "FLOPs_inference",
          "comm_MB_per_round_K", "GPU_b1_median_ms", "CPU1_b1_median_ms", "CPUall_b1_median_ms",
          "GPU_flows_per_s", "CPU1_flows_per_s", "GPU_keras_predict_b1_ms"]].copy()
show["FLOPs_inference"] = (show["FLOPs_inference"] / 1e6).round(2).astype(str) + " M"
print(show.round(3).to_string(index=False))
print("\nHardware:", t.gpu.iloc[0], "|", t.CPU1_hw.iloc[0], "|", t.CPUall_hw.iloc[0])
print("FLOPs check (layer count vs TensorFlow profiler):",
      [f"{a/1e6:.2f}M vs {b/1e6:.2f}M" for a, b in zip(t.FLOPs_inference, t.FLOPs_tf_profiler)])
print("Saved:", os.path.join(OUT, "efficiency_table.csv"))
