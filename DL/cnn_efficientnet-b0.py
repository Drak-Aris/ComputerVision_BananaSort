import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,roc_curve,auc
import matplotlib.pyplot as plt
from dataset.data_processing import get_dataset_loaders
from sklearn.utils.class_weight import compute_class_weight

def combined_to_binary(combined_label):
    if not isinstance(combined_label, str):
        return "rejeté"
    parts = combined_label.split('_')
    if len(parts) >= 2:
        maturite = parts[0]
        defaut = parts[1]
        if maturite == 'vert' and defaut == "sain":
            return 'export'
    return 'rejeté'

#TODO explain
def plot_roc_curve(all_labels_combined, all_probs, class_names, save_path=None):
    y_true = np.array([1 if combined_to_binary(class_names[l]) == 'rejeté' else 0
                       for l in all_labels_combined])
    rejet_indices = [i for i, name in enumerate(class_names)
                     if combined_to_binary(name) == 'rejeté']
    y_score = np.sum(all_probs[:, rejet_indices], axis=1)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Aléatoire")
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color="red", s=80,
                label=f"Seuil optimal = {optimal_threshold:.2f}")
    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR = Rappel)")
    plt.title("Courbe ROC – Classification binaire (export vs rejeté)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[ROC] Courbe sauvegardée : {save_path}")
    plt.show()
    print(f"AUC = {roc_auc:.4f}, Seuil optimal = {optimal_threshold:.2f} "
          f"(TPR={tpr[optimal_idx]:.3f}, FPR={fpr[optimal_idx]:.3f})")
    return roc_auc


def get_model(num_classes, pretrained=True):
    model = models.efficientnet_b0(pretrained=pretrained)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc="Entraînement")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, device, num_epochs=25, patience=5):
    best_val_loss = float('inf')
    early_stop_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f"\n--- Époque {epoch+1}/{num_epochs} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            early_stop_counter = 0
            print("--> Meilleur modèle sauvegardé.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping déclenché après {patience} époques sans amélioration.")
                break

    model.load_state_dict(torch.load('best_model.pth'))
    return model, history

#TODO Explain
def measure_inference_time(model, device, input_size=(1, 3, 224, 224), n_warmup=10, n_runs=100):
    model.eval()
    dummy = torch.randn(*input_size).to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(dummy)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                start = time.perf_counter()
                _ = model(dummy)
                end = time.perf_counter()
                times.append((end - start) * 1000.0)

    mean_ms = np.mean(times)
    std_ms = np.std(times)
    print(f"[Inference] Temps moyen : {mean_ms:.2f} ± {std_ms:.2f} ms/image")
    if mean_ms < 200:
        print("  Compatible convoyeur PHP (< 200ms)")
    else:
        print("  ATTENTION : > 200ms, risque de ralentir le convoyeur")
    return mean_ms

#TODO explain
def evaluate_model(model, dataloader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    # Rapport de classification multi-classes
    print("Rapport de classification (multi-classes) :")
    print(classification_report(all_labels_np, all_preds_np, target_names=class_names))

    # Matrice de confusion combinée
    cm = confusion_matrix(all_labels_np, all_preds_np)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation='vertical')
    plt.tight_layout()
    plt.title("Matrice de confusion (classes combinées)")
    plt.show()

    # Conversion binaire export/rejeté pour analyse métier
    binary_true = np.array([1 if combined_to_binary(class_names[l]) == 'rejeté' else 0
                            for l in all_labels_np])
    binary_pred = np.array([1 if combined_to_binary(class_names[p]) == 'rejeté' else 0
                            for p in all_preds_np])

    tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred).ravel()
    print("\n--- Analyse métier PHP (binaire export/rejeté) ---")
    print(f"✅ Export correctement accepté (TN) : {tn}")
    print(f"❌ Rejeté classé Export par erreur (FN) : {fn}  ← Risque pénalité conteneur")
    print(f"⚠️  Export classé Rejeté par erreur (FP) : {fp}  ← Perte de revenu")
    print(f"✅ Rejeté correctement rejeté (TP) : {tp}")

    # Coûts estimés (valeurs fictives, à adapter selon vos sources)
    cout_fp_par_fruit = 0.45  # €/fruit perdu si on jette un bon fruit
    cout_fn_par_fruit = 2.0  # €/fruit de pénalité si mauvais fruit exporté
    perte_fp = fp * cout_fp_par_fruit
    risque_fn = fn * cout_fn_par_fruit
    print(f"Perte estimée liée aux FP : {perte_fp:.2f} €")
    print(f"Risque estimé lié aux FN : {risque_fn:.2f} €")

    # Courbe ROC binaire
    plot_roc_curve(all_labels_np, all_probs, class_names, save_path="roc_curve.png")

    return all_labels_np, all_preds_np, all_probs


def plot_training_history(history, save_path="training_history.png"):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Historique] Graphiques sauvegardés : {save_path}")
    plt.show()


def save_embeddings(model, dataloader, device, class_names, output_csv="embeddings.csv"):
    embedding_model = nn.Sequential(*list(model.children())[:-1])
    embedding_model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extraction des embeddings"):
            images = images.to(device)
            emb = embedding_model(images)
            emb = torch.flatten(emb, start_dim=1)
            all_embeddings.append(emb.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    embeddings = np.vstack(all_embeddings)
    labels_array = np.array(all_labels)

    print(f"Nombre d'images traitées : {len(labels_array)}")
    print(f"Forme des embeddings : {embeddings.shape}")

    cols = [f"feat_{i}" for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=cols)

    df['label'] = [class_names[l] for l in labels_array]

    df.to_csv(output_csv, index=False)
    print(f"Embeddings sauvegardés dans {output_csv} (forme : {embeddings.shape})")


def main():
    data_dir = "../dataset"
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 25
    patience = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")

    dataloaders, class_names = get_dataset_loaders(data_dir, batch_size=batch_size)
    train_loader = dataloaders['train']
    val_loader = dataloaders['valid']
    num_classes = len(class_names)
    print(f"Classes détectées ({num_classes}) : {class_names}")

    # ---------- Gestion du déséquilibre ----------
    train_labels = train_loader.dataset.targets  # pour ImageFolder
    if not isinstance(train_labels, list):
        train_labels = list(train_labels)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Poids des classes : {class_weights_tensor}")
    # ---------------------------------------------

    model = get_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                               patience=3, factor=0.5)

    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        patience=patience
    )

    print("\nEntraînement terminé.")

    print("Extraction des embeddings...")
    save_embeddings(trained_model, train_loader, device, class_names, output_csv="embeddings_train.csv")

    if 'test' in dataloaders:
        test_loss, test_acc = validate(trained_model, dataloaders['test'], criterion, device)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        evaluate_model(model, dataloaders['test'], class_names, device)

    print("\nMesure du temps d'inférence...")
    inference_time = measure_inference_time(trained_model, device)

    plot_training_history(history, save_path="training_history.png")

if __name__ == "__main__":
    main()