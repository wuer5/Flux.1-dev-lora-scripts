import json
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class MyDataset(Dataset):
    def __init__(self, dataset_json_path, resluotion=512):
        super().__init__()
        with open(dataset_json_path, "r") as f:
            self.items = json.load(f)
        self.resluotion = resluotion
        self.transform = transforms.Compose(
            [
                transforms.Resize((resluotion, resluotion)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __getitem__(self, index: int):
        item = self.items[index]
        image = Image.open(item["img_path"]).convert("RGB")
        image = self.transform(image)
        return {"img": image, "captions": item["captions"]}

    def __len__(self) -> int:
        return len(self.gt_paths)
