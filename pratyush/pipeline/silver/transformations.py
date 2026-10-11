from torchvision.transforms import v2 
import torch 
augmentation_factor = 3

probs = 1 - (1 - (augmentation_factor - 1) 
         / augmentation_factor) ** 0.25 

transform = v2.Compose([
    v2.RandomApply(
        transforms=[
            v2.ElasticTransform(
                alpha=90, 
                sigma=8, 
            )
        ], 
        p = probs
    ), 

    v2.RandomApply(
        transforms = [
            v2.RandomAffine(
                degrees=10, 
                translate=(0.1, 0.1), 
                scale=(0.9, 1.1), 
                shear=10
            ), 
        ], 
        p = probs, 
    ), 
    
    v2.RandomApply(
        transforms = [
            v2.ColorJitter(
                brightness=0.2, 
                contrast=0.2,
            )
        ], p = probs 
    ), 

    v2.RandomApply(
        transforms = [
            v2.RandomRotation(20)
        ], p = probs 
    ), 

    v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ])

])