import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class RoadDataset(Dataset):
    """
    Notre classe personnalisée pour charger les paires Images/Masques.
    """
    def __init__(self, images_dir, masks_dir):
        # 1. Stocke les chemins 'images_dir' et 'masks_dir' dans l'objet (self. ...)
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        # 2. Crée une liste contenant les noms de fichiers des images (utilise os.listdir)
        self.images_names = os.listdir(images_dir)
        self.images_names.sort()

    def __len__(self):
        # PyTorch doit savoir combien d'images il y a au total.
        return len(self.images_names)

    def __getitem__(self, index):
        # Etape A : Récupérer le nom
        file_name = self.images_names[index]

        # Etape B : Chemins
        path_image = os.path.join(self.images_dir, file_name)
        path_mask = os.path.join(self.masks_dir, file_name)
        
        # Etape C : Chargement + Conversion couleur (seulement pour l'image)
        image = cv2.imread(path_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)    
        
        # Etape D : Normalisation (0 à 1) + Float32
        image = image.astype("float32") / 255.0
        mask = mask.astype("float32") / 255.0
        mask[mask < 0.5] = 0.0
        mask[mask >= 0.5] = 1.0

        # Etape E : Numpy -> Tensor
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)
        
        # Etape F : Dimensions
        # Image : (H, W, 3) -> (3, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)
        # Masque : (H, W) -> (1, H, W)
        mask_tensor = mask_tensor.unsqueeze(0)

        # Etape G : Return
        return image_tensor, mask_tensor



# ==========================================
# ZONE DE TEST (Pour vérifier si ton code marche)
# ==========================================
if __name__ == "__main__":
    # Change les chemins selon ton organisation
    img_dir = "dataset/images"
    msk_dir = "dataset/masks"

    # On instancie ta classe
    my_dataset = RoadDataset(img_dir, msk_dir)

    print(f"J'ai trouvé {len(my_dataset)} images dans le dataset.")

    # On essaie de charger la première image (index 0)
    first_image, first_mask = my_dataset[0]

    print(f"Forme de l'image tensor : {first_image.shape}") # Doit être [3, 256, 256]
    print(f"Forme du masque tensor : {first_mask.shape}")   # Doit être [256, 256] ou [1, 256, 256]
    print("Test réussi si les formes sont correctes !")