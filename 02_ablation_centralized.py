# -*- coding: utf-8 -*-
"""SynAttnNet - centralized attention ablation (Table III)

Four attention configurations are selected with the VARIANT switch below:
    "mha_se"   -> proposed ordering: MHA then SE
    "se_mha"   -> reversed ordering: SE then MHA
    "mha_only" -> multi-head attention only
    "se_only"  -> squeeze-and-excitation only

Set VARIANT, then run. Do not edit the attention block itself: the residual
connection, layer normalizations and feed-forward sub-layer are identical across
all four variants, so any difference in results is attributable to the attention
configuration alone. "se_mha" and "mha_se" contain exactly the same layers in a
different order and therefore have identical parameter counts.
"""

#Attention on Encoder
# Part 1: Data Preparation and Model Building
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Reshape, Flatten, Layer
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras import backend as K
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical
from google.colab import drive

# Global Variables
latent_dim = 39
labelnum = 1000 #2000/3000/4000
LABELS = ['chat', 'email', 'file', 'streaming', 'voip']

# --- ABLATION SWITCH ---------------------------------------------------------
# "mha_se" (proposed) | "se_mha" | "mha_only" | "se_only"
VARIANT = "mha_se"
# -----------------------------------------------------------------------------

def getlabelindex(Y_full, n_classes, labelnum):
    Y_full = pd.DataFrame(Y_full, columns=['label'])
    idxs_annot = []
    for idx in range(n_classes):
        labelindex = Y_full[Y_full['label'] == idx].index
        if len(labelindex) < labelnum:
            print(f"Insufficient labels for class {idx}, available: {len(labelindex)}, required: {labelnum}")
            idxs = np.random.choice(labelindex, len(labelindex), replace=True)
        else:
            idxs = np.random.choice(labelindex, labelnum, replace=False)
        idxs_annot.extend(idxs)
    return idxs_annot

# Squeeze-and-Excitation Block for 1D data
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
        scale = inputs * excitation
        return scale

def hybrid_attention_block(x, num_heads=4, key_dim=32, variant=VARIANT):
    if variant == "mha_se":
        # proposed: global cross-feature attention, then channel recalibration
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=0.1)(x, x)
        attention_output = SEBlock1D()(attention_output)
    elif variant == "se_mha":
        # reversed ordering; identical layers, identical parameter count
        attention_output = SEBlock1D()(x)
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=0.1)(attention_output, attention_output)
    elif variant == "mha_only":
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=0.1)(x, x)
    elif variant == "se_only":
        attention_output = SEBlock1D()(x)
    else:
        raise ValueError("VARIANT must be one of: mha_se, se_mha, mha_only, se_only")

    x = Add()([x, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    ffn = Dense(key_dim * 4, activation="relu")(x)
    ffn = Dropout(0.1)(ffn)
    ffn = Dense(K.int_shape(x)[-1])(ffn)
    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

def build_synattnnet_encoder(input_shape, latent_dim):
    input_e = Input(shape=input_shape)
    x = Conv1D(64, 3, padding="same", activation="relu", name='conv_1')(input_e)
    x = Conv1D(64, 3, padding="same", activation="relu", name='conv_2')(x)
    x = hybrid_attention_block(x, num_heads=4, key_dim=32, variant=VARIANT)
    x = MaxPooling1D(name='maxpool_1')(x)
    x = Conv1D(32, 3, padding="same", activation="relu", name='conv_3')(x)
    x = Conv1D(32, 3, padding="same", activation="relu", name='conv_4')(x)
    s_shape = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latent_dim, activation="relu")(x)
    encoder = Model(input_e, latent, name="synattnnet_encoder")
    encoder.summary()
    print(f"[{VARIANT}] encoder parameters: {encoder.count_params()}")
    return encoder, s_shape

def build_synattnnet_decoder(latent_dim, s_shape):
    input_d = Input(shape=(latent_dim,))
    x = Dense(np.prod(s_shape[1:]))(input_d)
    x = Reshape((s_shape[1], s_shape[2]))(x)
    x = Conv1D(32, 3, padding="same", activation="relu", name='conv_5')(x)
    x = Conv1D(32, 3, padding="same", activation="relu", name='conv_6')(x)
    x = UpSampling1D(name='upsampling_1d')(x)
    x = Conv1D(64, 3, padding="same", activation="relu", name='conv_7')(x)
    output = Conv1D(1, 3, padding="same", activation="relu", name='conv_8')(x)
    decoder = Model(input_d, output, name="synattnnet_decoder")
    decoder.summary()
    return decoder

def build_synattnnet_classifier(latent_dim, s_shape, n_classes):
    input_c = Input(shape=(latent_dim,))
    x = Dense(np.prod(s_shape[1:]))(input_c)
    x = Reshape((s_shape[1], s_shape[2]))(x)
    x = Conv1D(64, 3, padding="same", activation="relu")(x)
    x = Conv1D(64, 3, padding="same", activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(128, 3, padding="same", activation="relu")(x)
    x = Conv1D(128, 3, padding="same", activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(n_classes, activation="softmax")(x)
    classifier = Model(input_c, output, name="synattnnet_classifier")
    classifier.summary()
    return classifier

def prepare_data_and_models():
    # Mount Google Drive and load data
    drive.mount('/content/drive')
    root_path = '/content/drive/My Drive/'
    dfDS = pd.read_csv(root_path + 'ISCX_5class_each_normalized_cuttedfloefeature.csv')

    # Prepare data
    X_full = dfDS.iloc[:, 1:].values
    Y_full = dfDS["label"].values
    X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)
    inp_size = X_full.shape[1]
    n_classes = len(np.unique(Y_full))
    input_shape = (inp_size, 1)

    # Build models
    encoder, s_shape = build_synattnnet_encoder(input_shape, latent_dim)
    decoder = build_synattnnet_decoder(latent_dim, s_shape)
    classifier = build_synattnnet_classifier(latent_dim, s_shape, n_classes)

    # Complete models
    input_e = Input(shape=input_shape)
    synattnnet_ae = Model(input_e, decoder(encoder(input_e)))
    synattnnet_clf = Model(input_e, classifier(encoder(input_e)))
    synattnnet_joint = Model(input_e, [classifier(encoder(input_e)), decoder(encoder(input_e))])

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X_full, Y_full, test_size=0.1, random_state=5)
    idxs_annot = getlabelindex(y_train, n_classes, labelnum)
    x_train_labeled = x_train[idxs_annot]
    y_train_labeled = to_categorical(y_train[idxs_annot])
    y_test = to_categorical(y_test)

    # Save models and data (variant in the filename so the four runs do not overwrite each other)
    save_model(synattnnet_ae, root_path + f'synattnnet_ae_{VARIANT}.h5')
    save_model(synattnnet_clf, root_path + f'synattnnet_clf_{VARIANT}.h5')
    save_model(synattnnet_joint, root_path + f'synattnnet_joint_{VARIANT}.h5')

    # Save data
    np.save(root_path + 'x_train.npy', x_train)
    np.save(root_path + 'x_test.npy', x_test)
    np.save(root_path + 'y_train_labeled.npy', y_train_labeled)
    np.save(root_path + 'y_test.npy', y_test)
    np.save(root_path + 'x_train_labeled.npy', x_train_labeled)

# Part 2: Model Training and Evaluation
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def calculate_total_accuracy(synattnnet_ae, synattnnet_clf, x_test, y_test):
    """
    Calculate the combined accuracy of the SynAttnNet model
    """
    reconstruction_error = np.mean(np.square(x_test - synattnnet_ae.predict(x_test)))
    classifier_predictions = synattnnet_clf.predict(x_test)
    classifier_accuracy = np.mean(np.argmax(classifier_predictions, axis=1) == np.argmax(y_test, axis=1))

    reconstruction_weight = 0.3
    classification_weight = 0.7

    total_accuracy = (
        (1 - reconstruction_error) * reconstruction_weight +
        classifier_accuracy * classification_weight
    )

    return total_accuracy, reconstruction_error, classifier_accuracy

def train_and_evaluate():
    # Mount Google Drive
    drive.mount('/content/drive')
    root_path = '/content/drive/My Drive/'

    # Load models with custom_objects
    synattnnet_ae = load_model(root_path + f'synattnnet_ae_{VARIANT}.h5', custom_objects={'SEBlock1D': SEBlock1D})
    synattnnet_clf = load_model(root_path + f'synattnnet_clf_{VARIANT}.h5', custom_objects={'SEBlock1D': SEBlock1D})
    synattnnet_joint = load_model(root_path + f'synattnnet_joint_{VARIANT}.h5', custom_objects={'SEBlock1D': SEBlock1D})

    # Load data
    x_train = np.load(root_path + 'x_train.npy')
    x_test = np.load(root_path + 'x_test.npy')
    y_train_labeled = np.load(root_path + 'y_train_labeled.npy')
    y_test = np.load(root_path + 'y_test.npy')
    x_train_labeled = np.load(root_path + 'x_train_labeled.npy')

    LABELS = ['chat', 'email', 'file', 'streaming', 'voip']

    # Train Autoencoder
    print(f"\n[{VARIANT}] Training SynAttnNet Autoencoder...")
    synattnnet_ae.compile(loss='mse', optimizer='adam', metrics=['mse'])
    synattnnet_ae_history = synattnnet_ae.fit(x_train, x_train, epochs=50, batch_size=128,
                                        validation_data=(x_test, x_test), verbose=1)

    # Train Classifier
    print(f"\n[{VARIANT}] Training SynAttnNet Classifier...")
    synattnnet_clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    synattnnet_clf_history = synattnnet_clf.fit(x_train_labeled, y_train_labeled, epochs=50,
                                              batch_size=128, validation_data=(x_test, y_test), verbose=1)

    # Evaluate joint model
    print(f"\n[{VARIANT}] Evaluating SynAttnNet Joint Model...")
    synattnnet_joint.compile(optimizer="adam", loss=["categorical_crossentropy", "mse"],
                      metrics=[["accuracy"], ["mse"]])
    total_scores = synattnnet_joint.evaluate(x_test, [y_test, x_test], verbose=1)

    # Calculate and print individual component performances
    total_accuracy, reconstruction_error, classifier_accuracy = calculate_total_accuracy(
        synattnnet_ae, synattnnet_clf, x_test, y_test)

    print(f"\n=== Final Model Performance [variant = {VARIANT}] ===")
    print(f"Encoder+classifier parameters: {synattnnet_clf.count_params()}")
    print(f"SynAttnNet Reconstruction Error: {reconstruction_error:.4f}")
    print(f"Classifier Accuracy: {classifier_accuracy:.4f}")
    print(f"Total SynAttnNet Model Accuracy: {total_accuracy:.4f}")

    # Generate predictions and create reports
    y_pred = synattnnet_clf.predict(x_test, batch_size=100)

    print("\nDetailed Classification Report:")
    report = classification_report(y_test.argmax(axis=-1), y_pred.argmax(axis=-1),
                                 target_names=LABELS, digits=4)
    print(report)

    # Save results
    plt.figure(figsize=(12, 10))
    conf_matrix = confusion_matrix(y_test.argmax(axis=-1), y_pred.argmax(axis=-1))
    sns.heatmap(conf_matrix, annot=True, fmt='d', square=True, xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f'Confusion Matrix ({VARIANT})')
    plt.savefig(root_path + f'Confusion_Matrix_SynAttnNet_{VARIANT}.png', dpi=500, bbox_inches='tight')

    # Save reports
    df_report = pd.DataFrame(classification_report(y_test.argmax(axis=-1), y_pred.argmax(axis=-1),
                                                 target_names=LABELS, digits=4,
                                                 output_dict=True)).transpose()
    df_report.to_csv(root_path + f"classification_report_SynAttnNet_{VARIANT}.csv", index=True)

    accuracy_results = {
        'Variant': VARIANT,
        'Labels_per_class': labelnum,
        'Parameters': synattnnet_clf.count_params(),
        'Reconstruction_Error': reconstruction_error,
        'Classifier_Accuracy': classifier_accuracy,
        'Total_SynAttnNet_Accuracy': total_accuracy
    }
    pd.DataFrame([accuracy_results]).to_csv(root_path + f"accuracy_results_SynAttnNet_{VARIANT}.csv", index=False)

    # Append to a single ablation table so the four runs accumulate into Table III
    import os
    ablation_path = root_path + "ablation_table3.csv"
    pd.DataFrame([accuracy_results]).to_csv(
        ablation_path, mode='a', header=not os.path.exists(ablation_path), index=False)

    print("\nAll results have been saved to Google Drive.")

if __name__ == "__main__":
    prepare_data_and_models()
    train_and_evaluate()
