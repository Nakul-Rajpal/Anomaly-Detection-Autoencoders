from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

def get_data():

    df = pd.read_csv('data\Android_Malware.csv')

    df_clean = df.drop(columns=["Flow ID", "Source IP", "Destination IP", "Timestamp"])

    numeric = ['Flow duration', 'Total Fwd Packets']
    categorical = ['Protocol', 'Source Port', 'Destination Port']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric),
            ('cat', OneHotEncoder(handle_unknown="ignore"), categorical)
        ]
    )

    X_processed = preprocessor.fit_transform(df_clean)
    X_tensor = torch.tensor(X_processed.toarray(), dtype=torch.float32)

    # Split data: 60% train, 20% val, 20% test
    X_train, X_temp = train_test_split(X_tensor, test_size=0.4, random_state=42)
    X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

    # Wrap in DataLoaders
    train_loader = DataLoader(TensorDataset(X_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=64)
    test_loader = DataLoader(TensorDataset(X_test), batch_size=64)

    return train_loader, val_loader, test_loader, X_tensor.shape[1]

