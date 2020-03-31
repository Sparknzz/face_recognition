from torch.utils.data import DataLoader
from dataloaders import face_dataset
from torch.utils.data.sampler import *
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

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



def make_weights_for_balanced_classes(images, nclasses):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])

    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight
    
def make_data_loader(config):

    INPUT_SIZE = config['INPUT_SIZE']
    train_transform = transforms.Compose([
        # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]),  # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['RGB_MEAN'],
                             std=config['RGB_STD']),
    ])

    dataset_train = datasets.ImageFolder(config['DATA_ROOT'], train_transform)

    # create a weighted random sampler to process imbalanced data
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=config['BATCH_SIZE'], sampler=sampler, pin_memory=config['PIN_MEMORY'],
        num_workers=config['NUM_WORKERS'], drop_last=config['DROP_LAST']
    )

    NUM_CLASS = len(train_loader.dataset.classes)
    print("Number of Training Classes: {}".format(NUM_CLASS))
    
    return train_loader, NUM_CLASS
