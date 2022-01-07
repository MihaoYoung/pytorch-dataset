import torch
import os
import numpy as np
from torch.utils.data.dataset import random_split
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive

class Aircraft(VisionDataset):

    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', "manufacturer")
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

    def __init__(self, root: str, split: str ="trainval", class_type: str ="variant",
                 download=False, transform=None, target_transform=None) -> None:
        super(Aircraft, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.loader = default_loader

        if split not in self.splits:
            raise ValueError("Split '{}' is not found. Valid splits are: {}".format(
                split, ','.join(self.splits)
            ))
        if class_type not in self.class_types:
            raise ValueError("Class_type '{}' is not found. Class tyep are: {}".format(
                class_type, ",".join(self.class_types)
            ))
        
        self.split = split
        self.class_type = class_type
        self.root = os.path.join(self.root, "/aircraft/")

        if self._check_exists():
            print("Files already downloaded and verified")
        elif download:
            self._download()
        else:
            raise RuntimeError(
                "Dataset isn't found! You can use download=Ture to download it."
            )
        
        self.class_files = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                        "images_%s_%s.txt" % (self.class_type, self.split))

        (img_ids, targets, classes, class_to_id) = self.find_classes()
        samples = self.make_dataset(img_ids, targets)

        self.samples = samples
        self.classes = classes
        self.class_to_id = class_to_id
    
    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_folder) and 
                os.path.join(self.class_files))
    
    def _download(self):
        return

    def find_classes(self):
        img_ids = []
        targets = []

        with open(self.class_files, "r") as f:
            for line in f:
                split_line = line.split(' ')
                img_ids.append(split_line[0])
                targets.append(" ".join(split_line[1:]))

        
        classes = np.unique(targets)
        class_to_id = {classes[i] : i for i in range(len(classes))}
        targets = [class_to_id[i] for i in targets]

        return img_ids, targets, classes, class_to_id
    
    def make_dataset(self, img_ids, targets):
        assert (len(img_ids) == len(targets))
        samples = []

        for i in range(len(img_ids)):
            item = (os.path.join(self.root, self.img_folder,
                    "%s.jpg" % img_ids[i]), targets[i])
            samples.append(item)
        
        return samples