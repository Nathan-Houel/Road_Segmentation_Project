import torch
from torchvision import transforms
from model import UNET
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Instancier le modèle vide
model = UNET().to(DEVICE)

# Chargement des poids
weights = torch.load("mon_UNET.pth")

# Charger les poids dans le modèle
model.load_state_dict(weights)

# Passer le modèle en mode évaluation
model.eval()

# Transformation des images
Transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Chargement de l'image Test
img = Image.open("Test_image.jpg")

# Appliquer la transformation à notre image test
img_tensor = Transform(img)
image = img_tensor.unsqueeze(0).to(DEVICE)

# Calculer la prédiction
prediction = model(image)

# Conversion prédiction en proba
probs = torch.sigmoid(prediction)

print("Probabilité Max détectée :", probs.max().item())
print("Probabilité Moyenne :", probs.mean().item())

# Classification
preds = (probs > 0.5).float()

# Préparation pour l'affichage
pred_mask = preds.cpu().squeeze().numpy()
img_resized = img.resize((256, 256))

# On masque les pixels à 0 pour qu'ils soient invisibles
mask_visual = np.ma.masked_where(pred_mask == 0, pred_mask)

# Affichage 
plt.figure(figsize=(8, 8))
plt.title("Superposition : Image + Prédiction (Rose)")
plt.imshow(img_resized)
plt.imshow(mask_visual, cmap='Reds_r', alpha=0.6)
plt.axis('off') 
plt.show()
