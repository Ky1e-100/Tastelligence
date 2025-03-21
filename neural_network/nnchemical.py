import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
from tensorflow.keras.utils import to_categorical

RDLogger.DisableLog('rdApp.*')

def tokenize_smiles(smiles, radius=2, n_bits=2048):
    """
    Convert SMILES into an RDKit Morgan fingerprint using the new recommended API.
    
    - `radius`: Defines the fingerprint search depth (2 is standard)
    - `n_bits`: Number of bits in the fingerprint (2048 is typical)

    Returns:
        A NumPy array of shape (n_bits,)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("not valid mol")
        return np.zeros(n_bits)  # Return a zero vector for invalid SMILES
    
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprint = generator.GetFingerprint(mol)

    return np.array(fingerprint)

class CompoundFlavorGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_directory, batch_size=64, shuffle=True):
        self.data = pd.read_csv(data_directory, usecols=['Canonicalized Taste', 'Standardized SMILES'])
        self.data.dropna(inplace=True)
        
        # Encode taste labels
        self.label_encoder = LabelEncoder()
        self.data['Taste_Label'] = self.label_encoder.fit_transform(self.data['Canonicalized Taste'])
        self.num_classes = len(self.label_encoder.classes_)
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Ensure self.indices aligns with the cleaned dataset
        self.indices = np.arange(len(self.data))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
        
    def __len__(self):
        """Return number of batches per epoch"""
        return int(np.floor(len(self.data) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        if len(batch_indices) == 0:
            raise IndexError(f"Batch index {index} is out of bounds.")

        batch_data = self.data.iloc[batch_indices]

        x = self.process_inputs(batch_data)
        y = batch_data['Taste_Label'].values
        y = to_categorical(y, num_classes=self.num_classes)


        return x, y
    
    def process_inputs(self, batch_data):
        """Convert SMILES into numerical format (Morgan Fingerprints)"""
        
        # Convert SMILES to RDKit Morgan fingerprints
        smiles_vectorized = np.array([tokenize_smiles(smiles) for smiles in batch_data['Standardized SMILES']])
        
        return smiles_vectorized
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
            

# Test
if __name__ == "__main__":
    train_gen = CompoundFlavorGenerator('dataset/train.csv', 64)
    val_gen = CompoundFlavorGenerator('dataset/val.csv', 64)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2048,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax'),
    ])

    # Compile Model
    optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Model Summary
    model.summary()
    
    model.fit(x=train_gen, validation_data=val_gen, epochs=20, verbose=1)
    model.save("flavor_prediction_model2.keras")
    
    # test_gen = CompoundFlavorGenerator('old_dataset/test.csv', 64)
    # model = tf.keras.models.load_model('flavor_prediction_model2.keras')
    
    # test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    # print("Test data Loss: ", test_loss, "\n", "Test Accuracy: ", test_accuracy)




    # sample_smiles = "Nc1ncnc2c1ncn2C1OC(COP(=O)(O)O)C(O)C1O"
    # sample_fingerprint = tokenize_smiles(sample_smiles).reshape(1, -1)

    # prediction = model.predict(sample_fingerprint)

    # # Decode prediction (probabilities)
    # print(prediction)
    # predicted_class = np.argmax(prediction)  # Get class with highest probability
    # print(f"Predicted Class: {predicted_class}")
    # predicted_label = test_gen.label_encoder.inverse_transform([predicted_class])[0]
    # print(f"Predicted Taste: {predicted_label}")
    




    

    
     

    