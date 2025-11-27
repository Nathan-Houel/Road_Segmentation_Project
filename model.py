import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    (Conv3x3 -> BN -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

# ----------------------------------------------------------------------------


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # --- LA DESCENTE (Encoder) ---
        # On va créer 4 étages qui descendent.
        # À chaque étage :
        # 1. On applique la DoubleConv pour extraire les features.
        # 2. On applique un MaxPool pour diviser la taille par 2 (Zoom Out).
        
        # Etage 1 : On passe de 3 canaux (RGB) à 64
        self.ups = nn.ModuleList() # On stockera la montée ici plus tard
        self.downs = nn.ModuleList() # On stocke la descente ici
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Divise la taille par 2

        # Définissons les canaux pour chaque étage de la descente
        # 64 -> 128 -> 256 -> 512
        features = [64, 128, 256, 512]
        
        # TODO: Coder la boucle de création des couches de descente
        # Pour chaque feature dans la liste 'features' :
        #   On ajoute une DoubleConv(in_channels, feature) dans self.downs
        #   On met à jour in_channels pour qu'il soit égal à feature (relais)
        
        # --- LE FOND (Bottleneck) ---
        # C'est le point le plus bas (1024 canaux)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # --- LA MONTÉE (Decoder) ---
        # (On verra ça juste après, concentrons-nous sur la descente d'abord)