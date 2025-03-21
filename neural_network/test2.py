import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def tokenize_smiles(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("not valid mol")
        return np.zeros(n_bits)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprint = generator.GetFingerprint(mol)

    return np.array(fingerprint)

class CompoundFlavorGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_directory, batch_size=64, shuffle=True):
        self.raw_data = pd.read_csv(data_directory, usecols=['Canonicalized Taste', 'Standardized SMILES'])
        self.raw_data.dropna(inplace=True)

        # Group by SMILES to aggregate all associated taste labels
        grouped = self.raw_data.groupby("Standardized SMILES")["Canonicalized Taste"].apply(list).reset_index()
        grouped.columns = ['Standardized SMILES', 'Taste_List']
        grouped['Taste_List'] = grouped['Taste_List'].apply(lambda x: list(set(x)))

        # Multi-label binarization
        self.label_encoder = MultiLabelBinarizer()
        grouped['Taste_Label'] = self.label_encoder.fit_transform(grouped['Taste_List']).tolist()

        self.data = grouped
        self.num_classes = len(self.label_encoder.classes_)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        if len(batch_indices) == 0:
            raise IndexError(f"Batch index {index} is out of bounds.")

        batch_data = self.data.iloc[batch_indices]
        x = self.process_inputs(batch_data)
        y = np.array(batch_data['Taste_Label'].tolist())

        return x, y

    def process_inputs(self, batch_data):
        smiles_vectorized = np.array([tokenize_smiles(smiles) for smiles in batch_data['Standardized SMILES']])
        return smiles_vectorized

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

if __name__ == "__main__":
    # train_gen = CompoundFlavorGenerator('old_dataset/train.csv', 64)
    # val_gen = CompoundFlavorGenerator('old_dataset/val.csv', 64)

    # model = tf.keras.Sequential([
    #     tf.keras.layers.InputLayer(input_shape=(2048,)),
    #     tf.keras.layers.Dense(1024, activation='relu'),
    #     tf.keras.layers.Dropout(0.3),
    #     tf.keras.layers.Dense(512, activation='relu'),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Dense(256, activation='relu'),
    #     tf.keras.layers.Dropout(0.3),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(train_gen.num_classes, activation='sigmoid'),
    # ])

    # model.compile(optimizer=tf.keras.optimizers.SGD(), loss="binary_crossentropy", metrics=["accuracy"])
    # model.summary()

    # model.fit(x=train_gen, validation_data=val_gen, epochs=20, verbose=1)
    # model.save("flavor_prediction_model_multilabel.keras")
    
    test_gen = CompoundFlavorGenerator('old_dataset/test.csv', 64)
    model = tf.keras.models.load_model('flavor_prediction_model_multilabel.keras')
    
    # test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    # print("Test data Loss: ", test_loss, "\n", "Test Accuracy: ", test_accuracy)




    sample_smiles = "Sc1ccccc1"
    sample_fingerprint = tokenize_smiles(sample_smiles).reshape(1, -1)

    # Predict multi-label probabilities
    prediction = model.predict(sample_fingerprint)

    # Apply threshold (e.g., 0.5) to get binary predictions
    threshold = 0.4
    multi_hot = (prediction >= threshold).astype(int)

    # Decode predicted labels
    predicted_labels = test_gen.label_encoder.inverse_transform(multi_hot)[0]

    print(f"Predicted Labels: {predicted_labels}")
