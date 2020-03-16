from torch.utils.data import DataLoader
from dataloaders import face_dataset
from torch.utils.data.sampler import *
import numpy as np

#############################################################
class TwoBalanceClassSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        label = (self.dataset.df['label'].values)
        label = np.asarray(label).reshape(-1, 1).T

        self.neg_index = np.where(label == 0)[1]
        self.pos_index = np.where(label == 1)[1]

        # 2x
        self.num_image = len(self.neg_index)
        self.length = self.num_image * 2

    def __iter__(self):
        neg = np.random.choice(self.neg_index, self.num_image, replace=False)
        pos = np.random.choice(self.pos_index, self.num_image, replace=True)

        l = np.stack([neg, pos]).T
        l = l.reshape(-1)
        return iter(l)

    def __len__(self):
        return self.length


def make_data_loader(args, **kwargs):
    #     def __init__(self, data_shape, shuffle=False, aug_list = None, mean = None,
                #  rand_mirror = False, cutoff = 0, color_jittering = 0,
                #  images_filter = 0, split='train'):
    train_set = face_dataset.FaceDataset(args.image_shape, shuffle=True, rand_mirror = True, split='train')
    val_set = None

    # num_class = train_set.NUM_CLASSES

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=4,
        pin_memory=True)

    val_loader = None
    test_loader = None

    return train_loader, val_loader, test_loader