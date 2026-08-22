from torch.utils.data import Dataset 
from PIL import Image
import os 
import torch 

class CellDataset(Dataset): 
    def __init__(
        self, 
        cell_types: list[str], 
        augmentation_factor: int = 1,  
        set_type: str = "train", 
        transform=None
    ): 
        self.cell_types = cell_types 
        self.set_type = set_type 
        self.images: list[tuple] = []
        self.transform = transform 
        self.augmentation_factor = augmentation_factor

        for cell_type in self.cell_types: 
            for movie_id in ["01", "02"]: 
            
                cell_path: str = os.path.join("..", "..", self.set_type, cell_type, movie_id) 
                segment_gt_path: str = os.path.join("..", "..", self.set_type, cell_type, f"{movie_id}_GT", "SEG")

                for image_name in os.listdir(segment_gt_path): 
                    cell_segment_image_path: str = os.path.join(segment_gt_path, image_name)
                    image_number: str = image_name.split("man_seg")[1].split(".tif")[0]
                    cell_image_path: str = os.path.join(cell_path, f"t{image_number}.tif")

                    self.images.append((
                        cell_type, 
                        cell_image_path,
                        cell_segment_image_path, 
                        movie_id 
                    ))
    
    def __len__(self): 
        return len(self.images) * self.augmentation_factor
    
    def __getitem__(self, index: int): 
        real_index = index % len(self.images)
        cell_type, cell_image_path, cell_segment_image_path, movie_id = self.images[real_index]
        cell_image = Image.open(cell_image_path).convert("L")
        cell_segment_image = Image.open(cell_segment_image_path).convert("L")

        if self.transform:
            cell_image, cell_segment_image = self.transform(cell_image, cell_segment_image)

        return (cell_type, cell_image, cell_segment_image, movie_id)
     