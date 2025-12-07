import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNET
from my_dataset import RoadDataset
from tqdm import tqdm
import matplotlib.pyplot as plt


# --- HYPERPARAMÈTRES ---
LEARNING_RATE = 1e-4  # Vitesse à laquelle le modèle modifie ses poids (trop grand = instable, trop petit = lent)       
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # Nombre d'images traitées en même temps (si ta mémoire sature, baisse à 2 ou 1)              
NUM_EPOCHS = 10  # Nombre de fois que le modèle va voir l'ensemble du dataset           
NUM_WORKERS = 2  # Nombre de processeurs utilisés pour charger les données           
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
TRAIN_IMG_DIR = "dataset/images" 
TRAIN_MASK_DIR = "dataset/masks"


if __name__ == "__main__":
    # Instancie le Dataset
    Dataset = RoadDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)

    # Créer le DataLoader
    train_loader = DataLoader(Dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    # Le modèle 
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    # La Loss (Sigmoid + Binary Cross Entropy)
    loss_fn = nn.BCEWithLogitsLoss()

    # L'Optimiseur
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history = []

    # Boucle d'entraînement
    print("Début de l'entraînement ! 🚀")
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(train_loader)

        for batch_idx, (img, mask) in enumerate(loop):
            # Force la lecture sur le GPU
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)

            # Calcul de la prédiction produite par le modèle
            predictions = model(img)

            # Calcul de l'erreur
            loss = loss_fn(predictions, mask)
            
            # Mise à zéro des gradients
            optimizer.zero_grad()
            
            # Rétropropagation
            loss.backward()

            # Mise à jour des poids
            optimizer.step()

        # Affichage Loss
        print(f"Époque {epoch+1}/{NUM_EPOCHS} : Loss = {loss.item():.4f}")
        loss_history.append(loss.item())

    # Sauvegarde des poids
    torch.save(model.state_dict(), "mon_UNET.pth")

    # Création du graphique
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Progression de l\'entraînement')
    plt.xlabel('Époques')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig("courbe_loss.png") # On sauvegarde l'image
    print("Graphique sauvegardé sous 'courbe_loss.png' 📈")
