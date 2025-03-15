import numpy as np
from PIL import Image
import pickle


class CIFAR2:
    root = './data'
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    
    train_file = 'cifar2_train'
    test_file = 'cifar2_test'
    
    def __init__(self,  train=True,  transform=None):
        self.train=train
        self.transform=transform
        self.target_transform = None
        
        self.load(train)
    

    def load(self, train=True):
        if train:
            filepath = self.root + '/' + self.train_file
        else:
            filepath = self.root + '/' + self.test_file
            
        with open(filepath, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data = entry['data']
            self.targets = entry['labels']
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

