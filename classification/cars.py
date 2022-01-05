from _typeshed import Self
from genericpath import exists
import os
from posixpath import join
from sys import path
from numpy import roots
import scipy.io as sio
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive

class Cars(VisionDataset):

    url_list = {
        'imgs': ('http://imagenet.stanford.edu/internal/car196/car_ims.tgz', 'car_ims.tgz'),
        'annos': ('http://imagenet.stanford.edu/internal/car196/cars_annos.mat', 'cars_annos.mat')
    }

    mat_files = ["cars_annos.mat"]
    img_floder = ["car_ims"]
    
    def __init__(self, root: str, train: bool=True, transform=None, target_transform=None, download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train
        self.root = os.path.join(self.root, "/stanford cars/")

        if self._check_exists():
            print("Files already downloaded and verified")
        elif download:
            self._download()
        else:
            raise RuntimeError(
                "Dataset isn't found! You can use download=Ture to download it."
            )
    
        mat_file_path = os.path.join(self.root, "standford cars", self.mat_files[0])
        loaded_mat = sio.loadmat(mat_file_path)
        loaded_mat = loaded_mat["annotations"][0]

        self.samples = []

        for item in loaded_mat:
            if self.train != bool(item['test'][0]):
                path = str(item['relative_im_path'][0])
                label = int(item['class'][0]) - 1
                self.samples.append((path, label))

        
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        path = os.path.join(self.root, path)

        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.samples)
    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_floder[0])) \
                and os.path.exists(os.path.join(self.root, self.mat_files[0]))
    
    def _download(self):
        return
    