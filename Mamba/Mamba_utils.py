import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from mamba_ssm import Mamba
from dataset_loader import MatFileDataset
import os
from typing import Optional

# Optional: Hugging Face Transformers for loading pretrained Mamba (e.g., mamba-2.8b)
try:
    from transformers import AutoModel
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

"""
This script sets up the Mamba Vision Classifier and runs experiments based on configurations defined in config.py.
"""
class MambaVisionClassifier(nn.Module):
    """
    A Mamba-based vision classifier.
    It initializes a Mamba model with the specified dimensions and a final linear layer for classification.
    Parameters:
    - d_model: Dimension of the input features 
    - num_classes: Number of output classes for classification.
    - d_state: Dimension of the state in the Mamba model.
    - d_conv: Dimension of the convolutional layers in the Mamba model.
    - expand: Expansion factor for the Mamba model.
    """
    def __init__(self, d_model, num_classes, d_state, d_conv, expand):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.fc = nn.Linear(d_model, num_classes)
    
    """
    Forward pass through the Mamba model and the final linear layer.
    Parameters:
    - x: Input tensor of shape (batch_size, sequence_length, d_model). Note the shape corresponds to (batch_size, H, W * C)
        Each image in the batch is processed row by row. Each row is represented as a vector of length d_model. These vectors form a sequence, so the sequence length corresponds to the number of rows in the image.
    Returns:
    - Output tensor of shape (batch_size, num_classes).
      For each image in the batch, it outputs a vector of scores for each class
    """
    def forward(self, x):
    
        # self.mamba(x): the input data `x` flows through the Mamba layer. 
        # The output of self.mamba(x) is a tensor that has the same shape as the input, (batch, sequence_length, d_model)
        # However, it returns a new sequence where each row contains information not only about itself, but also about the previous rows. 
        mamba_out = self.mamba(x)

        # [:, -1, :] This is a slicing operation that selects the last step of the sequence.
        # The idea is that after processing the whole sequence (all rows of the
        # image), by selecting the last time step, we obtain a feature vector that summarizes the full sequence. This vector is then used for classification.
        # - ':' on the first dimension: "Get all items in the batch."
        # - '-1' on the second dimension: "Get only the VERY LAST item from the sequence."
        # - ':' on the third dimension: "Get all the features for that item."
        last_hidden_state = mamba_out[:, -1, :]

        # self.fc is a linear layer that transforms the image's summary feature vector (last_hidden_state) into scores for each class.
        output = self.fc(last_hidden_state)
        return output


class PretrainedMambaVisionClassifier(nn.Module):
    """
    Vision classifier that adapts image-row sequences to a pretrained Mamba backbone
    (e.g., state-spaces/mamba-2.8b from Hugging Face) via a learnable projection.

    Notes:
    - Expects inputs of shape (batch, seq_len, d_local), where d_local is derived from image width*channels.
    - Projects inputs to the backbone hidden size (d_pretrained), forwards them as `inputs_embeds`,
      pools the last token, and applies a linear classifier.
    - Can freeze the backbone and train only projection+head, or fine-tune end-to-end.
    """
    def __init__(
        self,
        num_classes: int,
        d_local: int,
        pretrained_model_name: str = "state-spaces/mamba-2.8b",
        freeze_backbone: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        super().__init__()

        if not _HF_AVAILABLE:
            raise ImportError(
                "transformers is not available. Please install it to use pretrained Mamba, e.g.\n"
                "pip install transformers accelerate safetensors --upgrade\n"
                "and ensure you have access to the checkpoint (e.g., state-spaces/mamba-2.8b)."
            )

        # Load pretrained backbone. device_map requires accelerate for sharded loading.
        # When device_map is None, the model is loaded on CPU, and you can then .to(device).
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        # Discover backbone hidden size; common attribute names
        d_pretrained = getattr(self.backbone.config, "hidden_size", None)
        if d_pretrained is None:
            d_pretrained = getattr(self.backbone.config, "d_model", None)
        if d_pretrained is None:
            raise ValueError("Unable to infer backbone hidden size (d_model/hidden_size) from config.")

        self.d_local = d_local
        self.d_pretrained = int(d_pretrained)

        # Project local row vectors (W or W*3) to pretrained hidden size
        self.input_proj = nn.Linear(self.d_local, self.d_pretrained)

        # Classification head on top of pooled representation
        self.fc = nn.Linear(self.d_pretrained, num_classes)

        # Optionally freeze backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.freeze_backbone = freeze_backbone

    def forward(self, x):
        # x: (B, L, d_local)
        x = self.input_proj(x)
        # Forward as continuous embeddings to the pretrained backbone
        # Many HF models support `inputs_embeds`; if not, this will raise a clear error upstream.
        outputs = self.backbone(inputs_embeds=x, use_cache=False)
        # Prefer last_hidden_state for pooling
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden = outputs.last_hidden_state
        else:
            # Fallback: if outputs is a tuple, use the first item
            hidden = outputs[0]
        pooled = hidden[:, -1, :]
        logits = self.fc(pooled)
        return logits

"""
Run an experiment with the given configuration.
This function initializes the dataset, model, optimizer, and loss function,
and runs the training and validation loops.
It also stores the results and saves the test logits to a .npy file.
Parameters:
- config: Configuration dictionary containing experiment parameters.
- results_dict: Dictionary to store results of the experiment.
"""
def run_experiment(config, results_dict):
    """
    Runs a complete training and evaluation experiment based on a configuration dictionary.
    This version saves the output .npy files into a 'Results' subdirectory.
    """
    experiment_name = config['experiment_name']
    print(f"\n{'='*60}\n -- Starting Experiment: {experiment_name} --\n{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Data Loading
    dataset = MatFileDataset(config["file_path"], config)
    config['d_model'] = dataset.d_model
    
    # Split Dataset
    train_size = int(config["train_split"] * len(dataset))
    val_size = int(config["validation_split"] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Model Initialization
    model = MambaVisionClassifier(
        d_model=config['d_model'], num_classes=dataset.num_classes, 
        d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"]).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # Training Loop
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    print("Starting Training...")
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            train_loss += loss.item()
        history['train_loss'].append(train_loss / len(train_loader))
        
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_accuracy'].append(100 * correct / total)
        
        print(f"Epoch [{epoch+1}/{config['epochs']}] | Val Accuracy: {history['val_accuracy'][-1]:.2f}%")


    print("\nStarting final evaluation on the test set...")
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            all_logits.append(outputs.cpu())
            all_labels.append(labels.cpu())
    
    all_logits_tensor = torch.cat(all_logits, dim=0).numpy()
    all_labels_tensor = torch.cat(all_labels, dim=0).numpy()
    
    test_accuracy = 100 * (all_logits_tensor.argmax(1) == all_labels_tensor).sum() / len(all_labels_tensor)
 
    results_dir = 'Results'
    os.makedirs(results_dir, exist_ok=True)
    
    logits_filename = os.path.join(results_dir, f"logits_{experiment_name}.npy")
    labels_filename = os.path.join(results_dir, f"labels_{experiment_name}.npy")
    
    np.save(logits_filename, all_logits_tensor)
    np.save(labels_filename, all_labels_tensor)
    
    print(f"Saved test logits to: {logits_filename}")
    print(f"Saved test labels to: {labels_filename}")
    
    # Store and Plot Results
    results_dict[experiment_name] = {
        "config": config, "history": history, "test_accuracy": test_accuracy,
        "logits_file": logits_filename, "labels_file": labels_filename
    }
    
    print(f"\n--- Experiment '{experiment_name}' Finished ---")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    hist = results_dict[experiment_name]['history']
    ax1.plot(hist['train_loss'], label='Training Loss'); ax1.plot(hist['val_loss'], label='Validation Loss')
    ax1.set_title('Loss Over Epochs'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(hist['val_accuracy'], label='Validation Accuracy', color='green')
    ax2.set_title('Validation Accuracy Over Epochs'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.legend(); ax2.grid(True)
    plt.suptitle(f"Training Results for: {experiment_name}", fontsize=16)
    plt.show()


def run_experiment_pretrained(config, results_dict):
    """
    Run training/evaluation using a pretrained Mamba backbone (e.g., mamba-2.8b) via Hugging Face.

    Required config keys (in addition to dataset keys):
    - experiment_name: str
    - pretrained_model_name: str, e.g. "state-spaces/mamba-2.8b" or "state-spaces/mamba-2.8b-hf"

    Optional config keys:
    - freeze_backbone: bool (default: True)
    - learning_rate: float (default: 2e-4)
    - epochs: int (default: 5)
    - batch_size: int (default: 16)
    - use_amp: bool (default: False)  # mixed precision if CUDA available
    - torch_dtype: str ("bfloat16", "float16", "float32") or None
    - device_map: str or None (e.g., "auto" to shard across devices; requires accelerate)
    """
    experiment_name = config['experiment_name']
    print(f"\n{'='*60}\n -- Starting Pretrained Experiment: {experiment_name} --\n{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Data Loading
    dataset = MatFileDataset(config["file_path"], config)
    config['d_model'] = dataset.d_model

    # Splits
    train_size = int(config.get("train_split", 0.7) * len(dataset))
    val_size = int(config.get("validation_split", 0.15) * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 16), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get("batch_size", 16), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.get("batch_size", 16), shuffle=False)

    # Pretrained settings
    pretrained_model_name = config.get("pretrained_model_name", "state-spaces/mamba-2.8b")
    freeze_backbone = bool(config.get("freeze_backbone", True))
    lr = float(config.get("learning_rate", 2e-4))
    epochs = int(config.get("epochs", 5))
    use_amp = bool(config.get("use_amp", False)) and torch.cuda.is_available()

    # Resolve torch dtype if specified
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        None: None,
    }
    torch_dtype = dtype_map.get(config.get("torch_dtype", None), None)
    device_map = config.get("device_map", None)

    # Model
    print("Initializing pretrained backbone and adapter...")
    model = PretrainedMambaVisionClassifier(
        num_classes=dataset.num_classes,
        d_local=config['d_model'],
        pretrained_model_name=pretrained_model_name,
        freeze_backbone=freeze_backbone,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model = model.to(device)

    # Optimizer on trainable params only
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    print("Starting Training (pretrained)...")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            train_loss += float(loss.item())
        history['train_loss'].append(train_loss / max(1, len(train_loader)))

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        batch_loss = criterion(outputs, labels).item()
                else:
                    outputs = model(images)
                    batch_loss = criterion(outputs, labels).item()
                val_loss += float(batch_loss)
                total += labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
        history['val_loss'].append(val_loss / max(1, len(val_loader)))
        history['val_accuracy'].append(100.0 * correct / max(1, total))

        print(f"Epoch [{epoch+1}/{epochs}] | Val Accuracy: {history['val_accuracy'][-1]:.2f}%")

    # Final evaluation on test set
    print("\nStarting final evaluation on the test set (pretrained)...")
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            all_logits.append(outputs.detach().cpu())
            all_labels.append(labels)

    all_logits_tensor = torch.cat(all_logits, dim=0).numpy()
    all_labels_tensor = torch.cat(all_labels, dim=0).numpy()

    test_accuracy = 100.0 * (all_logits_tensor.argmax(1) == all_labels_tensor).sum() / len(all_labels_tensor)

    results_dir = 'Results'
    os.makedirs(results_dir, exist_ok=True)
    logits_filename = os.path.join(results_dir, f"logits_pretrained_{experiment_name}.npy")
    labels_filename = os.path.join(results_dir, f"labels_pretrained_{experiment_name}.npy")
    np.save(logits_filename, all_logits_tensor)
    np.save(labels_filename, all_labels_tensor)
    print(f"Saved test logits to: {logits_filename}")
    print(f"Saved test labels to: {labels_filename}")

    results_dict[experiment_name] = {
        "config": config,
        "history": history,
        "test_accuracy": float(test_accuracy),
        "logits_file": logits_filename,
        "labels_file": labels_filename,
        "pretrained": True,
    }

    print(f"\n--- Pretrained Experiment '{experiment_name}' Finished ---")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    hist = results_dict[experiment_name]['history']
    ax1.plot(hist['train_loss'], label='Training Loss'); ax1.plot(hist['val_loss'], label='Validation Loss')
    ax1.set_title('Loss Over Epochs (Pretrained)'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(hist['val_accuracy'], label='Validation Accuracy', color='green')
    ax2.set_title('Validation Accuracy Over Epochs (Pretrained)'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.legend(); ax2.grid(True)
    plt.suptitle(f"Training Results (Pretrained) for: {experiment_name}", fontsize=16)
    plt.show()
