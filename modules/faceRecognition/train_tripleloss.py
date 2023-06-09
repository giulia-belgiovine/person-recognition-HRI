from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


from pytorch_metric_learning import losses, miners, samplers, trainers, testers
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import pytorch_metric_learning


import numpy as np
import os
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import umap
from cycler import cycler
from PIL import Image
from sklearn.model_selection import train_test_split


logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s"%pytorch_metric_learning.__version__)
torch.manual_seed(1234)

# TRAIN PARAMETERS
# data_dir = '/home/icub/PycharmProjects/SpeakerRecognitionYarp/data/face'

# model_save_path = "/home/icub/PycharmProjects/SpeakerRecognitionYarp/project/faceRecognition/incr_learn_analysis" \
#                   "training_saved_models/test.pt"


LR = 1e-6
workers = 0 if os.name == 'nt' else 8
margin_p = 0.3
# Set other training parameters
batch_size = 80
num_epochs = 150


class MyDataset(Dataset):

    def __init__(self, img_path_list, label_list, dict, classes, trans):

        self.samples = img_path_list
        self.targets = label_list
        self.imgs = self.get_imgs(self.samples, self.targets)
        self.classes = classes
        self.class_to_idx = dict
        self.transform = trans

    def __getitem__(self, index):
        image = Image.open(self.samples[index])
        label = self.targets[index]
        x = self.transform(image)
        return x, label

    def get_label(self):
        label = self.targetsc
        return label

    def get_imgs(self, img_list, label_list):
        imgs = []
        for img, l in zip(img_list, label_list):
            imgs.append((img, l))
        return imgs

    def __len__(self):
        return len(self.samples)


def get_item_list(train_dataset):
    img_list = []
    label_list = []
    for img, label in train_dataset:
        img_list.append(img)
        label_list.append(label)

    return img_list, label_list


def training_pipeline(train_subset, dict, name_list, save_model_folder, checkpoint=""):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    # Data transformations
    trans_train = transforms.Compose([
        transforms.RandomApply(transforms=[
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomHorizontalFlip()]),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization,
    ])

    trans_val = transforms.Compose([
        # transforms.CenterCrop(120),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization,
    ])

    img_list, label_list = get_item_list(train_subset)
    X_train, X_val, y_train, y_val = train_test_split(img_list, label_list, test_size=0.2, stratify=label_list)

    train_dataset = MyDataset(X_train, y_train, dict, name_list, trans_train)
    val_dataset = MyDataset(X_val, y_val, dict, name_list, trans_val)
    print("Length of training dataset is {}. Length od Validation Dataset is {}".format(len(train_dataset), len(val_dataset)))

    # train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train_aligned"), transform=trans_train)
    # val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val_aligned"), transform=trans_val)

    # Prepare the model
    model = InceptionResnetV1(
        classify=False,
        pretrained="vggface2",
        dropout_prob=0.5
    ).to(device)

    # checkpoint = torch.load(checkpoint_path)
    if checkpoint:
        print("Training Pipeline found new checkpoint")
        model.load_state_dict(checkpoint)

    # for param in list(model.parameters()):
    #     param.requires_grad = False
    #
    # for param in list(model.parameters())[:-1]:
    #     param.requires_grad = True

    trunk_optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)

    # Set the loss function
    loss = losses.TripletMarginLoss(margin=margin_p)

    # Set the mining function
    # miner = miners.BatchEasyHardMiner(
    #     pos_strategy=miners.BatchEasyHardMiner.EASY,
    #     neg_strategy=miners.BatchEasyHardMiner.SEMIHARD)

    miner = miners.BatchHardMiner()

    # Set the dataloader sampler
    sampler = samplers.MPerClassSampler(train_dataset.targets, m=4, length_before_new_iter=len(train_dataset)) # train_dataset.label_list

    # Package the above stuff into dictionaries.
    models = {"trunk": model}
    optimizers = {"trunk_optimizer": trunk_optimizer}
    loss_funcs = {"metric_loss": loss}
    mining_funcs = {"tuple_miner": miner}

    # Create the tester
    record_keeper, _, _ = logging_presets.get_record_keeper("logs5", "tensorboard5")
    hooks = logging_presets.get_hook_container(record_keeper)


    dataset_dict = {"val": val_dataset, "train": train_dataset}
    model_folder = save_model_folder

    # def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    #     logging.info("UMAP plot for the {} split and label set {}".format(split_name, keyname))
    #     label_set = np.unique(labels)
    #     num_classes = len(label_set)
    #     fig = plt.figure(figsize=(8, 7))
    #     plt.gca().set_prop_cycle(cycler("color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]))
    #     for i in range(num_classes):
    #         idx = labels == label_set[i]
    #         plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
    #     plt.show()

    tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook=hooks.end_of_testing_hook,
                                                dataloader_num_workers=4,
                                                # accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
                                                accuracy_calculator=AccuracyCalculator(include=['mean_average_precision_at_r'], k="max_bin_count"))

    end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder, patience=15, splits_to_eval=[('val', ['train'])])

    # Create the trainer
    trainer = trainers.MetricLossOnly(models,
                                      optimizers,
                                      batch_size,
                                      loss_funcs,
                                      mining_funcs,
                                      train_dataset,
                                      sampler=sampler,
                                      dataloader_num_workers=8,
                                      end_of_iteration_hook=hooks.end_of_iteration_hook,
                                      end_of_epoch_hook=end_of_epoch_hook)

    trainer.train(num_epochs=num_epochs)


def main():
    training_pipeline()


if __name__ == '__main__':
    main()

