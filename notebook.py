from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from lab_utils.visualization import extract_feature_maps, plot_feature_maps_like_reference, plot_training_history, show_tensor_batch
from typing import Tuple, Dict, List, Union

def find_project_root() -> Path:
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / 'data').exists():
            return candidate
    return Path.cwd().resolve()
PROJECT_ROOT = find_project_root()
DATA_ROOT = PROJECT_ROOT / 'data'
METADATA_PATH = DATA_ROOT / 'metadata.csv'
ARTIFACT_DIR = PROJECT_ROOT / 'artifacts'
LABELS = ('cat', 'dog')
SPLITS = ('train', 'val', 'test')
SEED = 1234
EPOCHS = 20
NUMPY_PRED_PATH = ARTIFACT_DIR / 'lab3_pytorch_predictions.csv'

def seed_index(length: int, offset: int=0) -> int:
    if length <= 0:
        raise ValueError('Cannot choose an index from an empty collection.')
    return int((SEED + offset) % length)

def build_metadata_from_folders(data_root: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for label in LABELS:
            label_dir = data_root / split / label
            for path in sorted(label_dir.glob('*.jpg')) + sorted(label_dir.glob('*.png')):
                with Image.open(path) as image:
                    image = image.convert('RGB')
                    width, height = image.size
                rows.append({'filepath': str(path.relative_to(data_root)), 'label': label, 'split': split, 'width': width, 'height': height})
    return pd.DataFrame(rows)
from sklearn.model_selection import train_test_split
import pandas as pd

def build_label_mapping(frame: pd.DataFrame) -> Tuple[Dict[str, int], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_to_index = {'cat': 0, 'dog': 1}
    labelled = frame.copy()
    labelled['label_id'] = labelled['label'].map(label_to_index)
    train_df, temp_df = train_test_split(labelled, test_size=0.3, random_state=42, stratify=labelled['label_id'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label_id'])
    return (label_to_index, labelled, train_df, val_df, test_df)

def image_to_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert('RGB')
    image = image.resize((64, 64))
    image_array = np.array(image, dtype=np.float32)
    image_array /= 255.0
    image_tensor = torch.tensor(image_array)
    image_tensor = image_tensor.permute(2, 0, 1)
    return image_tensor

class CatsDogsDataset(Dataset):

    def __init__(self, frame: pd.DataFrame, data_root: Path):
        self.frame = frame.reset_index(drop=True)
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = self.data_root / row['filepath']
        image_tensor = image_to_tensor(image_path)
        label_tensor = torch.tensor(row['label_id'], dtype=torch.long)
        return (image_tensor, label_tensor)
BATCH_SIZE = 32

def build_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, data_root: Path, batch_size: int=32, seed: int=SEED, dataset_cls: type[Dataset]=CatsDogsDataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = dataset_cls(train_df, data_root)
    val_dataset = dataset_cls(val_df, data_root)
    test_dataset = dataset_cls(test_df, data_root)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return (train_loader, val_loader, test_loader)

def inspect_first_batch(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    if loader is None:
        raise ValueError('Complete Question 3 before inspecting a batch.')
    batch_images, batch_labels = next(iter(loader))
    print('Image batch:', batch_images.shape, batch_images.dtype)
    print('Label batch:', batch_labels.shape, batch_labels.dtype)
    assert batch_images.ndim == 4, 'Batches of images should have shape (B, C, H, W).'
    assert batch_images.shape[1] == 3, 'Color images should have 3 channels.'
    assert batch_labels.dtype == torch.long, 'Labels should be torch.long class indices.'
    return (batch_images, batch_labels)
import torch
import torch.nn as nn

class CatsDogsSimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.stage2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 16 * 16, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        return self.classifier(x)

def setup_training(model: nn.Module, device: Union[torch.device, None]=None, learning_rate: float=0.001) -> Tuple[torch.device, nn.Module, nn.Module, torch.optim.Optimizer]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return (device, model, criterion, optimizer)

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        predictions = torch.argmax(logits, dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size
    average_loss = total_loss / total_examples
    average_accuracy = total_correct / total_examples
    return (average_loss, average_accuracy)

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_examples += batch_size
    average_loss = total_loss / total_examples
    average_accuracy = total_correct / total_examples
    return (average_loss, average_accuracy)

def run_training_experiment(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, epochs: int=5, plot: bool=True) -> Tuple[list[dict[str, float]], float, float, float | None]:
    history = []
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
        print(f'Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | train_acc={train_acc:.3f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}')
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'\nTest: loss={test_loss:.4f}, acc={test_acc:.3f}')
    return (history, test_loss, test_acc)
