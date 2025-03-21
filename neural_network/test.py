import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import pickle  # For saving the LabelEncoder
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# Function to convert SMILES to Morgan Fingerprints
def tokenize_smiles(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)  # Return zero vector for invalid SMILES
    
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprint = generator.GetFingerprint(mol)
    
    return np.array(fingerprint, dtype=np.float32)  # Ensure tensor-compatible dtype

# PyTorch Dataset Class
class CompoundFlavorDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file, usecols=['Canonicalized Taste', 'Standardized SMILES'])
        self.data.dropna(inplace=True)

        # Encode taste labels
        self.label_encoder = LabelEncoder()
        self.data['Taste_Label'] = self.label_encoder.fit_transform(self.data['Canonicalized Taste'])
        self.num_classes = len(self.label_encoder.classes_)

        # Save label encoder for decoding predictions later
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data.iloc[idx]['Standardized SMILES']
        fingerprint = tokenize_smiles(smiles)  # Convert SMILES to fingerprint
        label = self.data.iloc[idx]['Taste_Label']
        
        return torch.tensor(fingerprint, dtype=torch.float32), torch.tensor(label, dtype=torch.long)  # PyTorch tensors


class FlavorPredictionNN(nn.Module):
    def __init__(self, input_dim=2048, num_classes=5):
        super(FlavorPredictionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
import collections
def train():
    # Training Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and create DataLoaders
    batch_size = 64
    train_dataset = CompoundFlavorDataset("old_dataset/train.csv")
    print(collections.Counter(train_dataset.data['Taste_Label']))

    val_dataset = CompoundFlavorDataset("old_dataset/val.csv")  # Load validation data

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for validation

    # Define model, loss, and optimizer
    model = FlavorPredictionNN(input_dim=2048, num_classes=train_dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training Loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training Phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # Validation Phase (No Gradient Computation)
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():  # Disable gradient computation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Compute validation accuracy
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct_val / total_val

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Save trained model
    torch.save(model.state_dict(), "flavor_prediction_model.pth")
    print("Model training complete and saved!")


if __name__ == "__main__":
    train()
    
    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FlavorPredictionNN(input_dim=2048, num_classes=5)
    model.load_state_dict(torch.load("flavor_prediction_model.pth", map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Load Label Encoder
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Function to predict flavor for a given SMILES string
    def predict_flavor(smiles):
        fingerprint = tokenize_smiles(smiles).reshape(1, -1)  # Reshape to match model input
        fingerprint_tensor = torch.tensor(fingerprint, dtype=torch.float32).to(device)
        
        with torch.no_grad():  # No gradients needed for inference
            prediction = model(fingerprint_tensor)
        
        predicted_class = torch.argmax(prediction).item()
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]

        return predicted_label

    # Example Prediction
    sample_smiles = "C(C1C(C(C(O1)O)O)O)OC2C(C(C(C(O2)O)O)O)O"  # Example molecule
    predicted_taste = predict_flavor(sample_smiles)
    print(f"Predicted Taste: {predicted_taste}")
