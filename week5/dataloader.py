import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
from PIL import Image

from torch.utils.data import Dataset


# TODO: HABRA QUE MIRARLO BIEN PARA LA TASK C)

# adapted from:
# https://pytorch.org/vision/stable/_modules/torchvision/datasets/flickr.html#Flickr30k
class Flickr30k(Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file) + '.json'

        # Read annotations and store in a dict
        with open(self.ann_file) as fh:
            data = fh.read()
        self.annotations = json.loads(data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index]

        # Image
        filename = os.path.join(self.root, img_id)
        img = Image.open(filename).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = self.annotations[img_id]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.annotations)