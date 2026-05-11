#!/usr/bin/env python
# coding: utf-8

# # Lab 3: PyTorch for Cat vs Dog Faces
# 
# This notebook is the next step after Labs 1 and 2, but it is self-contained.
# 
# - Lab 1 worked with image arrays, normalization, and a hand-crafted feature matrix.
# - Lab 2 worked with metadata tables, split handling, and label mapping in Pandas.
# - Lab 3 turns the same kind of image workflow into PyTorch tensors, datasets, loaders, and a trainable CNN.
# 
# In this notebook, you will train a small **binary image classifier** with PyTorch.
# 
# We will focus on the core training pipeline:
# 
# - turning a metadata table into training inputs
# - converting image files into PyTorch tensors
# - building `Dataset` and `DataLoader` objects
# - defining a simple CNN model
# - choosing a loss and optimizer
# - writing training and evaluation loops
# - comparing learned features with handcrafted NumPy features from Lab 1
# 
# Set the fixed random seed `1234` in the first code cell. Each notebook uses it for sampling, split suggestions, and visualization so the results are reproducible.
# 
# **Questions in this lab**
# 
# 1. Map labels to integers in the dataframe loaded in this notebook
# 2. Build a dataset that returns tensors
# 3. Create train, validation, and test DataLoaders
# 4. Inspect one mini-batch
# 5. Define a simple CNN classifier
# 6. Set up loss, optimizer, and device
# 7. Complete one training epoch
# 8. Evaluate the model on a validation or test loader
# 9. Train for a few epochs and compare it with the Lab 1 NumPy feature pipeline
# 

# In[1]:


from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from lab_utils.visualization import (
    extract_feature_maps,
    plot_feature_maps_like_reference,
    plot_training_history,
    show_tensor_batch,
)
from typing import Tuple, Dict, List, Union

def find_project_root() -> Path:
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / "data").exists():
            return candidate
    return Path.cwd().resolve()

PROJECT_ROOT = find_project_root()
DATA_ROOT = PROJECT_ROOT / "data"
METADATA_PATH = DATA_ROOT / "metadata.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

LABELS = ("cat", "dog")
SPLITS = ("train", "val", "test")
SEED = 1234
EPOCHS = 5
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
NUMPY_PRED_PATH = ARTIFACT_DIR / "lab3_pytorch_predictions.csv"

def seed_index(length: int, offset: int = 0) -> int:
    if length <= 0:
        raise ValueError("Cannot choose an index from an empty collection.")
    return int((SEED + offset) % length)

def build_metadata_from_folders(data_root: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for label in LABELS:
            label_dir = data_root / split / label
            for path in sorted(label_dir.glob("*.jpg")) + sorted(label_dir.glob("*.png")):
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    width, height = image.size
                rows.append(
                    {
                        "filepath": str(path.relative_to(data_root)),
                        "label": label,
                        "split": split,
                        "width": width,
                        "height": height,
                    }
                )
    return pd.DataFrame(rows)

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        "Dataset not found. Place the prepared subset at data/."
    )

if METADATA_PATH.exists():
    df = pd.read_csv(METADATA_PATH)
else:
    df = build_metadata_from_folders(DATA_ROOT)

print(f"Fixed seed: {SEED}")
print(df.head())
print(df["split"].value_counts())


# ## Question 1: Turn string labels into integer labels
# 
# Complete `build_label_mapping(...)`.
# 
# It should:
# 
# - a dictionary `label_to_index`
# - a dataframe with `label_id`
# - train, validation, and test dataframes
# - a new column `label_id`
# 
# Use `cat -> 0` and `dog -> 1`.
# 
# This is the same Pandas-style label handling from Lab 2, but now the labels become the targets for PyTorch training.
# 

# In[3]:


from sklearn.model_selection import train_test_split
import pandas as pd

def build_label_mapping(frame: pd.DataFrame) -> Tuple[Dict[str, int], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # TODO: build the label_to_index mapping using LABELS.
    # TODO: copy the input dataframe, called "labelled", and add the integer label_id column
    # TODO: split the labeled dataframe into train_df, val_df, and test_df.

    # build label mapping from LABELS
    label_to_index = {
        label: idx
        for idx, label in enumerate(LABELS)
    }

    # copy dataframe and add integer labels
    labelled = frame.copy()

    labelled["label_id"] = (
        labelled["label"]
        .map(label_to_index)
        .astype(int)
    )

    # use existing split column if available
    if "split" in labelled.columns:

        train_df = (
            labelled[labelled["split"] == "train"]
            .copy()
            .reset_index(drop=True)
        )

        val_df = (
            labelled[labelled["split"] == "val"]
            .copy()
            .reset_index(drop=True)
        )

        test_df = (
            labelled[labelled["split"] == "test"]
            .copy()
            .reset_index(drop=True)
        )

    else:
        # fallback split
        train_df, temp_df = train_test_split(
            labelled,
            test_size=0.3,
            random_state=42,
            stratify=labelled["label_id"]
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df["label_id"]
        )

    return (
        label_to_index,
        labelled,
        train_df,
        val_df,
        test_df
    )


label_to_index, df, train_df, val_df, test_df = build_label_mapping(df)

print(label_to_index)
print(train_df.head())

assert set(label_to_index.keys()) == {"cat", "dog"}, "Both classes should appear in the mapping."


# ## Question 2: Build a dataset that returns tensors
# 
# In Lab 1, you turned images into NumPy arrays and normalized pixel values.
# Now do the same preprocessing in PyTorch.
# 
# Complete:
# 
# - `image_to_tensor`
# - `CatsDogsDataset.__getitem__`
# 
# Your dataset should return:
# 
# - an image tensor with shape `(3, 64, 64)`
# - an integer label tensor
# 

# In[4]:


def image_to_tensor(path: Path) -> torch.Tensor:
    # TODO: open the image, convert to RGB, normalize to [0, 1],
    # and permute to channel-first format (C, H, W).
    # This mirrors the image preprocessing you already practiced in Lab 1.
    #raise NotImplementedError("Convert an image file into a float tensor.")
    image = Image.open(path).convert("RGB")
    image = image.resize((64, 64))

    image_array = np.array(image, dtype=np.float32)
    image_array /= 255.0
    # The image is currently in (H, W, C) format. Permuting it to (C, H, W).
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    return image_tensor

class CatsDogsDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data_root: Path):
        self.frame = frame.reset_index(drop=True)
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        # TODO: load the image tensor and return (image_tensor, label_tensor)
        # The label should be a torch.long tensor containing the class index.
        #raise NotImplementedError("Complete __getitem__.")
        image_path = self.data_root / row["filepath"]
        image_tensor = image_to_tensor(image_path)
        label_tensor = torch.tensor(row["label_id"], dtype=torch.long)
        return image_tensor, label_tensor

train_dataset = CatsDogsDataset(train_df, DATA_ROOT)
first_image, first_label = train_dataset[0]
print(first_image.shape, first_image.dtype, first_label)


# ## Question 3: Create DataLoaders
# 
# Complete `build_dataloaders(...)`.
# 
# Turn the metadata table in this notebook into training inputs.
# 
# Build three DataLoaders:
# 
# - training loader with `shuffle=True`
# - validation loader with `shuffle=False`
# - test loader with `shuffle=False`
# 
# Use a batch size of `32`.
# 
# Reuse `train_df`, `val_df`, and `test_df` here.
# 

# In[7]:


BATCH_SIZE = 32
train_loader_generator = torch.Generator().manual_seed(SEED)

def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_root: Path,
    batch_size: int = 32,
    seed: int = SEED,
    dataset_cls: type[Dataset] = CatsDogsDataset,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # TODO: create train_dataset, val_dataset, and test_dataset with dataset_cls.
    # Hint: each dataset should receive one split dataframe and data_root.
    train_dataset = dataset_cls(train_df, data_root)
    val_dataset = dataset_cls(val_df, data_root)
    test_dataset = dataset_cls(test_df, data_root)

    # TODO: build the three dataloaders.
    # Hint: only the training loader should shuffle and use the seeded generator.
    num_workers = 2
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers = 0,
        pin_memory = torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers = 0,
        pin_memory = torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers = 0,
        pin_memory = torch.cuda.is_available()
    )
    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = build_dataloaders(
    train_df,
    val_df,
    test_df,
    DATA_ROOT,
    batch_size=BATCH_SIZE,
    seed=SEED,
)

train_loader, val_loader, test_loader


# ## Question 4: Inspect one mini-batch
# 
# Complete `inspect_first_batch(...)`.
# 
# Pull one batch from the training loader and verify:
# 
# - image batch shape
# - label batch shape
# - image dtype
# - label dtype
# 

# In[8]:


def inspect_first_batch(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    if loader is None:
        raise ValueError("Complete Question 3 before inspecting a batch.")

    # TODO: read the first mini-batch from the loader.
    batch_images, batch_labels = next(iter(loader))

    print("Image batch:", batch_images.shape, batch_images.dtype)
    print("Label batch:", batch_labels.shape, batch_labels.dtype)

    assert batch_images.ndim == 4, "Batches of images should have shape (B, C, H, W)."
    assert batch_images.shape[1] == 3, "Color images should have 3 channels."
    assert batch_labels.dtype == torch.long, "Labels should be torch.long class indices."

    return batch_images, batch_labels


batch_images, batch_labels = inspect_first_batch(train_loader)

show_tensor_batch(
    batch_images[:8].cpu().numpy(),
    batch_labels[:8].cpu().numpy(),
    class_names=LABELS,
    max_items=8,
    ncols=4,
)


# ## Shape checkpoint
# 
# Before moving on to the CNN, check that your mini-batch looks right:
# 
# - images: `(batch, 3, 64, 64)`
# - labels: integer class IDs with dtype `torch.long`
# 
# If either one looks off, fix the dataset or preprocessing step before Question 5.
# 

# 
# ## Question 5: Define a simple CNN classifier
# 
# In Lab 1, the features were hand-crafted.
# Here, the CNN learns its own features automatically.
# 
# Complete the model below.
# 
# Suggested architecture:
# 
# - stage 1: `Conv2d(3, 16, kernel_size=3, padding=1)` -> `ReLU` -> `MaxPool2d(2)`
# - stage 2: `Conv2d(16, 32, kernel_size=3, padding=1)` -> `ReLU` -> `MaxPool2d(2)`
# - classifier: `Flatten` -> `Linear(32 * 16 * 16, 64)` -> `ReLU` -> `Linear(64, 2)`

# In[9]:


import torch
import torch.nn as nn

class CatsDogsSimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        # TODO: build a small CNN that still runs comfortably on CPU.
        # This replaces the hand-crafted feature pipeline from Lab 1 with learned features.
        # Hint:
        #   stage1: Conv2d(3, 16, 3, padding=1) -> ReLU -> MaxPool2d(2)
        #   stage2: Conv2d(16, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
        #   classifier: Flatten -> Linear(32 * 16 * 16, 64) -> ReLU -> Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        return self.classifier(x)


model = CatsDogsSimpleCNN()
example_logits = model(batch_images[:4])
print(example_logits.shape)

assert example_logits.shape == (4, 2), "The classifier should output two logits per image."


# ## Question 6: Set up the training ingredients
# 
# Complete `setup_training(...)`.
# 
# Choose:
# 
# - a device (`cuda` if available, otherwise `cpu`)
# - a loss function
# - an optimizer
# 
# Use `CrossEntropyLoss` and `Adam` for this lab.
# 
# Set the learning rate to `1e-3`.
# 

# In[10]:


def setup_training(
    model: nn.Module,
    device: Union[torch.device, None] = None,
    learning_rate: float = 1e-3,
) -> Tuple[torch.device, nn.Module, nn.Module, torch.optim.Optimizer]:
    # TODO: pick cuda when available, otherwise cpu.
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    model = model.to(device)
    # TODO: move the model to the device and create criterion + optimizer.
    # TODO: use CrossEntropyLoss for the criterion and Adam for the optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return device, model, criterion, optimizer


device, model, criterion, optimizer = setup_training(model)

print("Using device:", device)


# ## Question 7: Complete one training epoch
# 
# Fill in the missing logic for:
# 
# - zeroing gradients
# - forward pass
# - loss computation
# - backward pass
# - optimizer step
# - batch accuracy tracking
# 

# In[11]:


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Step 1: clear old gradients from the previous batch.
        optimizer.zero_grad()
        # Step 2: run the model forward to get class scores (logits).
        logits = model(images)
        # Step 3: compare logits with the true labels to compute the loss.
        loss = criterion(logits, labels)
        # Step 4: run backpropagation so PyTorch computes gradients.
        loss.backward()
        # Step 5: take one optimizer step to update the model weights.
        optimizer.step()
        # Step 6: turn logits into predicted class ids with argmax.
        predictions = torch.argmax(logits, dim=1)
        # Step 7: update the running loss, correct count, and example count.
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size

        # Hint: multiply loss.item() by the batch size before adding to total_loss. 
        # Due to averaging in the loss function, loss.item() gives the average loss
        # per example in the batch, so we need to multiply by the batch size to get
        # the total loss for the batch before adding it to total_loss.
        # TODO: complete the training step
        #raise NotImplementedError("Implement the training loop for one epoch.")

    average_loss = total_loss / total_examples
    average_accuracy = total_correct / total_examples
    return average_loss, average_accuracy


# ## Question 8: Evaluate one model on one loader
# 
# Complete `evaluate(...)`.
# 
# It should run the model in evaluation mode on one loader and return average loss and accuracy.
# 
# Reflection prompts:
# 
# 1. Did the simple CNN outperform the Lab 1 NumPy feature pipeline?
# 2. Did the model start to overfit?
# 3. What would you try next: horizontal flips, more filters, one more conv block, or longer training?
# 4. Why might learned features from a CNN outperform the handcrafted NumPy features from Lab 1?
# 

# In[12]:


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
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
            # TODO: compute logits, loss, predictions, and batch size.

    average_loss = total_loss / total_examples
    average_accuracy = total_correct / total_examples
    return average_loss, average_accuracy

val_loss, val_acc = evaluate(model, val_loader, criterion, device)
print(f"Validation: loss={val_loss:.4f}, acc={val_acc:.3f}")


# ## Question 9: Train for a few epochs and evaluate
# 
# Complete `run_training_experiment(...)`.
# 
# It should train for a few epochs, evaluate on the test set.
# 
# Reflection prompts:
# 
# 1. Did the model start to overfit?
# 2. What would you try next: horizontal flips, more filters, one more conv block, or longer training?
# 

# In[13]:


def run_training_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 5,
    plot: bool = True,
) -> Tuple[list[dict[str, float]], float, float, float | None]:
    # TODO: run the training/validation loop, evaluate on the test set,
    history = []

    for epoch in range(epochs):

        # train
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        # validation
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        # save history
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.3f}"
        )

    # evaluate on test set
    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(
        f"\nTest: "
        f"loss={test_loss:.4f}, "
        f"acc={test_acc:.3f}"
    )
    return history, test_loss, test_acc


history, test_loss, test_acc = run_training_experiment(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    epochs=EPOCHS,
    plot=True,
)

history


# ## Optional Visualization: Feature Maps Like the AlexNet Notebook
# 
# The main lab now uses a simple CNN, so we can visualize feature maps from the trained model directly.
# 
# The visualization utility is intentionally styled like the tiled activation-grid view from the referenced AlexNet notebook:
# 
# - one tile per channel
# - `viridis` colormap
# - light gaps between maps
# - fixed `vmin` and `vmax` controls so maps are easier to compare
# 
# We will inspect stage-1 and stage-2 activations for one seed-specific training image.
# 

# In[14]:


#if device is None:
#    raise ValueError("Complete Questions 5-9 before visualizing feature maps.")

feature_map_device = device if device is not None else torch.device("cpu")
model = model.to(feature_map_device)
model.eval()

feature_map_index = seed_index(len(train_dataset), offset=500)
feature_map_row = train_df.iloc[feature_map_index]
example_image, _ = train_dataset[feature_map_index]
print(
    f"Feature-map example for seed 1234: "
    f"{feature_map_row['label']} -> {feature_map_row['filepath']}"
)

stage1_maps = extract_feature_maps(
    model.stage1,
    example_image,
    device=feature_map_device,
)
stage2_maps = extract_feature_maps(
    nn.Sequential(model.stage1, model.stage2),
    example_image,
    device=feature_map_device,
)

plot_feature_maps_like_reference(
    stage1_maps,
    title="Simple CNN feature maps after stage 1",
    figsize=(10, 10),
)

plot_feature_maps_like_reference(
    stage2_maps,
    title="Simple CNN feature maps after stage 2",
    figsize=(10, 10),
)


# ## Optional extension and recap
# 
# If you finish early, try one of these:
# 
# - increase the filters from `16, 32` to `24, 48`
# - add one more convolution block before the classifier
# - compare stage-1 and stage-2 feature maps for the same image
# - add random horizontal flips to the training data
# 
# Recap:
# 
# - Which ideas from Labs 1 and 2 showed up again in Lab 3?
# - What changed when you moved from NumPy and Pandas into PyTorch?
# 
# Then record which change helped validation accuracy most and how the feature maps changed.
# 
