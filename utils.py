import os
import numpy as np
import pandas as pd
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.datasets.utils import check_integrity,\
    extract_archive, verify_str_arg, download_and_extract_archive
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from ADP_utils.classesADP import classesADP
from typing import Any
import pickle

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum_accuracy = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum_accuracy += val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# for ADP dataset (also used for BCSS dataset)
def accuracyADP(preds, targets):
    acc5 = 0
    targets_all = targets.data.int()
    acc1 = torch.sum(preds == targets_all)
    preds_cpu = preds.cpu()
    targets_all_cpu = targets_all.cpu()

    for i, pred_sample in enumerate(preds_cpu):
        labelv = targets_all_cpu[i]
        numerator = torch.sum(np.bitwise_and(pred_sample, labelv))
        denominator = torch.sum(np.bitwise_or(pred_sample, labelv))
        acc5 += (numerator.double()/denominator.double())

    return acc1, acc5

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

# for ADP dataset
def _data_transforms_adp(args):
    ADP_MEAN = [0.81233799, 0.64032477, 0.81902153]
    ADP_STD = [0.18129702, 0.25731668, 0.16800649]

    degrees = 45
    horizontal_shift, vertical_shift = 0.1, 0.1

    ###### train transform
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=degrees, translate=(horizontal_shift, vertical_shift)),
        transforms.ToTensor(),
        transforms.Normalize(ADP_MEAN, ADP_STD)
    ])
    
    if args.image_size != 272:
        train_transform.transforms.insert(0,transforms.Resize((args.image_size, args.image_size),interpolation=transforms.functional.InterpolationMode.BICUBIC))
        
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    ###### valid transform
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(ADP_MEAN, ADP_STD)
    ])
    
    if args.image_size != 272:
        valid_transform.transforms.insert(0,transforms.Resize((args.image_size, args.image_size),interpolation=transforms.functional.InterpolationMode.BICUBIC))

    return train_transform, valid_transform

# for BCSS dataset
def _data_transforms_bcss(args):
    BCSS_MEAN = [0.7107, 0.4878, 0.6726]
    BCSS_STD = [0.1788, 0.2152, 0.1615]

    degrees = 45
    horizontal_shift, vertical_shift = 0.1, 0.1

    ###### train transform
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=degrees, translate=(horizontal_shift, vertical_shift)),
        transforms.ToTensor(),
        transforms.Normalize(BCSS_MEAN, BCSS_STD)
    ])
    
    if args.image_size != 272:
        train_transform.transforms.insert(0,transforms.Resize((args.image_size, args.image_size),interpolation=transforms.functional.InterpolationMode.BICUBIC))
        
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))
    
    ###### valid transform
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(BCSS_MEAN, BCSS_STD)
    ])

    if args.image_size != 272:
        valid_transform.transforms.insert(0,transforms.Resize((args.image_size, args.image_size),interpolation=transforms.functional.InterpolationMode.BICUBIC))

    return train_transform, valid_transform

# for BACH dataset
def _data_transforms_bach(args):
    BACH_MEAN = [0.6880, 0.5881, 0.8209]
    BACH_STD = [0.1632, 0.1841, 0.1175]

    degrees = 45
    horizontal_shift, vertical_shift = 0.1, 0.1

    ###### train transform
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=degrees, translate=(horizontal_shift, vertical_shift)),
        transforms.ToTensor(),
        transforms.Normalize(BACH_MEAN, BACH_STD)
    ])

    if args.image_size != 272:
        train_transform.transforms.insert(0,transforms.Resize((args.image_size, args.image_size),interpolation=transforms.functional.InterpolationMode.BICUBIC))
        
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    ###### valid transform
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(BACH_MEAN, BACH_STD)
    ])

    if args.image_size != 272:
        valid_transform.transforms.insert(0,transforms.Resize((args.image_size, args.image_size),interpolation=transforms.functional.InterpolationMode.BICUBIC))
    
    return train_transform, valid_transform

# for OS dataset
def _data_transforms_os(args):
    OS_MEAN = [0.8414, 0.6492, 0.7377]
    OS_STD = [0.1379, 0.2508, 0.1979]

    degrees = 45
    horizontal_shift, vertical_shift = 0.1, 0.1

    ###### train transform
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=degrees, translate=(horizontal_shift, vertical_shift)),
        transforms.ToTensor(),
        transforms.Normalize(OS_MEAN, OS_STD)
    ])

    if args.image_size != 272:
        train_transform.transforms.insert(0,transforms.Resize((args.image_size, args.image_size),interpolation=transforms.functional.InterpolationMode.BICUBIC))
        
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    ###### valid transform
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(OS_MEAN, OS_STD)
    ])

    if args.image_size != 272:
        valid_transform.transforms.insert(0,transforms.Resize((args.image_size, args.image_size),interpolation=transforms.functional.InterpolationMode.BICUBIC))
    
    return train_transform, valid_transform

# for ADP dataset
class ADP_dataset(Dataset):
    db_name = 'ADP V1.0 Release'
    ROI = 'img_res_1um_bicubic'
    csv_file = 'ADP_EncodedLabels_Release1_Flat.csv'
    
    def __init__(self, 
                level, 
                transform, 
                root, 
                split = 'train', 
                portion = 0.5,
                loader = default_loader): 
        '''
        Args:
            level (str): a string corresponding to a dict
                defined in "ADP_scripts\classes\classesADP.py"
                defines the hierarchy to be trained on
            transform (callable, optional): A function/transform that  takes in an
                PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            root (string): Root directory of the ImageNet Dataset.
            split (string, optional): The dataset split, supports ``train``, 
                ``valid``, or ``test``.
            loader (callable, optional): A function to load an image given its
                path. Defaults to default_loader defined in torchvision

        Attributes:
            self.full_image_paths (list) : a list of image paths
            self.class_labels (np.ndarray) : a numpy array of class labels 
                (num_samples, num_classes)
        '''
        
        self.root = root
        self.split = verify_str_arg(split, "split", ("train", "valid", "test", "train_search", "valid_search"))
        self.transform = transform
        self.loader = loader
        self.portion = portion

        # getting paths:
        csv_file_path = os.path.join(self.root, self.db_name, self.csv_file)

        ADP_data = pd.read_csv(filepath_or_buffer=csv_file_path, header=0) # reads data and returns a pd.dataframe
        # rows are integers starting from 0, columns are strings: e.g. "Patch Names", "E", ...

        split_folder = os.path.join(self.root, self.db_name, 'splits')

        if self.split == "train":
            train_inds = np.load(os.path.join(split_folder, 'train.npy'))
            out_df = ADP_data.loc[train_inds, :]

        elif self.split == "valid":
            valid_inds = np.load(os.path.join(split_folder, 'valid.npy'))
            out_df = ADP_data.loc[valid_inds, :]

        elif self.split == "test":
            test_inds = np.load(os.path.join(split_folder, 'test.npy'))
            out_df = ADP_data.loc[test_inds, :]

        # for darts search
        elif self.split == "train_search":
            train_inds = np.load(os.path.join(split_folder, 'train.npy'))
            train_search_inds = train_inds[: int(np.floor(self.portion * len(train_inds)))]
            out_df = ADP_data.loc[train_search_inds, :]

        elif self.split == "valid_search":
            train_inds = np.load(os.path.join(split_folder, 'train.npy'))
            valid_search_inds = train_inds[int(np.floor(self.portion * len(train_inds))) :]
            out_df = ADP_data.loc[valid_search_inds, :]

        self.full_image_paths = [os.path.join(self.root, self.db_name, self.ROI, image_name) for image_name in out_df['Patch Names']]
        self.class_labels = out_df[classesADP[level]['classesNames']].to_numpy(dtype=np.float32)

    def __getitem__(self, idx) -> torch.Tensor:
        
        path = self.full_image_paths[idx]
        label = self.class_labels[idx]

        sample = self.loader(path) # Loading image
        if self.transform is not None: # PyTorch implementation
            sample = self.transform(sample)

        return sample, torch.tensor(label)

    def __len__(self) -> int:
        return(len(self.full_image_paths))

# for BCSS dataset
class BCSSDataset(Dataset):
    db_name = 'BCSS_transformed'

    def __init__(self,
                 root,
                 split="train",
                 transform=None,
                 loader=default_loader,
                 multi_labelled=True) -> None:
        """
        Retrieved from: https://bcsegmentation.grand-challenge.org/
        Args:
            root (string):
                Directory of the transformed dataset, e.g. "/home/BCSS_transformed"
            split (string, optional): The dataset split, supports ``train``,
                ``valid``, or ``test``.
            transform (callable, optional): A function/transform that  takes in an
                PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            loader (callable, optional): A function to load an image given its
                path. Defaults to default_loader defined in torchvision
            multi_labelled (bool): a boolean controlling whether the output labels are a multilabelled array
                or an index corresponding to the single label
        """

        self.root = root
        self.split = verify_str_arg(split, "split", ("train", "valid", "test"))
        self.transform = transform
        self.loader = loader

        # getting samples from preprocessed pickle file
        if multi_labelled:
            df = pd.read_csv(os.path.join(self.root, self.db_name, self.split + ".csv"), index_col="image")
        else:
            df = pd.read_csv(os.path.join(self.root, self.db_name, self.split + "_with_norm_mass.csv"), index_col="image")
        self.samples = [(image.replace('\\', '/'), label) for image, label in zip(df.index, df.to_records(index=False))]

        if multi_labelled:
            self.samples = [(os.path.join(self.root, self.db_name, path), list(label)) for path, label in self.samples]
        else:
            self.samples = [(os.path.join(self.root, self.db_name, path), np.argmax(list(label))) for path, label in self.samples]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(df.columns)}

        self.class_labels = df.to_numpy(dtype=np.float32)


    def __getitem__(self, idx) -> [Any, torch.Tensor]:

        path, label = self.samples[idx]

        sample = self.loader(path)  # Loading image
        if self.transform is not None:  # PyTorch implementation
            sample = self.transform(sample)

        return sample, torch.tensor(label, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.samples)

# for BACH dataset
class BACH_transformed(Dataset):
    db_name = 'BACH_transformed'

    def __init__(self, root, split="train", transform=None, loader=default_loader) -> None:
        """

        Args:
            root (string):
                Directory of the transformed dataset, e.g. /home/BACH_transformed
            split (string, optional): The dataset split, supports ``train``,
                ``valid``, or ``test``.
            transform (callable, optional): A function/transform that  takes in an
                PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            loader (callable, optional): A function to load an image given its
                path. Defaults to default_loader defined in torchvision
        """

        self.root = root
        self.split = verify_str_arg(split, "split", ("train", "valid", "test"))
        self.transform = transform
        self.loader = loader

        # getting samples from preprocessed pickle file
        self.samples = pickle.load(open(os.path.join(self.root, self.db_name, self.split+".pickle"), "rb"))
        self.samples = [(os.path.join(self.root, self.db_name, path), label) for path, label in self.samples]
        self.class_to_idx = pickle.load(open(os.path.join(self.root, self.db_name, "class_to_idx.pickle"), "rb"))

    def __getitem__(self, idx) -> [Any, torch.Tensor]:

        path, label = self.samples[idx]

        sample = self.loader(path)  # Loading image
        if self.transform is not None:  # PyTorch implementation
            sample = self.transform(sample)

        return sample, torch.tensor(label)

    def __len__(self) -> int:
        return len(self.samples)

# for OS dataset
class OS_transformed(Dataset):
    db_name = 'OS_transformed'

    def __init__(self, root, split="train", transform=None, loader=default_loader) -> None:
        """

        Args:
            root (string):
                Directory of the transformed dataset, e.g. /home/OS_transformed
            split (string, optional): The dataset split, supports ``train``,
                ``valid``, or ``test``.
            transform (callable, optional): A function/transform that  takes in an
                PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            loader (callable, optional): A function to load an image given its
                path. Defaults to default_loader defined in torchvision
        """

        self.root = root
        self.split = verify_str_arg(split, "split", ("train", "valid", "test"))
        self.transform = transform
        self.loader = loader

        # getting samples from preprocessed pickle file
        self.samples = pickle.load(open(os.path.join(self.root, self.db_name, self.split+".pickle"), "rb"))
        self.samples = [(os.path.join(self.root, self.db_name, path), label) for path, label in self.samples]
        self.class_to_idx = pickle.load(open(os.path.join(self.root, self.db_name, "class_to_idx.pickle"), "rb"))

    def __getitem__(self, idx) -> [Any, torch.Tensor]:

        path, label = self.samples[idx]

        sample = self.loader(path)  # Loading image
        if self.transform is not None:  # PyTorch implementation
            sample = self.transform(sample)

        return sample, torch.tensor(label)

    def __len__(self) -> int:
        return len(self.samples)

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
