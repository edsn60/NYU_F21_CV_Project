import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset
from torchvision import transforms

import FKP_dataloader
import FKP_utils
import FKP_opt
import FKP_models
import FKP_train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = pd.read_csv('./datas/training.csv')
test_data = pd.read_csv('./datas/test.csv')


train_data_large = train_data[FKP_opt.datasets_tags['large_kpset']].dropna()
train_data_small = train_data[FKP_opt.datasets_tags['small_kpset']].dropna()


transform_large = transforms.Compose([ FKP_utils.RandomHorizontalFlip(p=0.8, data_type="large_kpset"),
                                       FKP_utils.Normalize(),
                                       FKP_utils.ToTensor()])
transform_small = transforms.Compose([ FKP_utils.RandomHorizontalFlip(p=0.8, data_type="small_kpset"),
                                       FKP_utils.Normalize(),
                                       FKP_utils.ToTensor()])
transform = transforms.Compose([ FKP_utils.Normalize(),
                                 FKP_utils.ToTensor()])

train_set_large = FKP_dataloader.FKDataset(train_data_large, transform=transform_large)
train_set_small = FKP_dataloader.FKDataset(train_data_small, transform=transform_small)
test_set = FKP_dataloader.FKDataset(test_data, train=False, transform=transform)

train_loader_large, valid_loader_large = FKP_dataloader.generate_valid_set(train_set_large, FKP_opt.valid_partition, FKP_opt.batch_size)
train_loader_small, valid_loader_small = FKP_dataloader.generate_valid_set(train_set_small, FKP_opt.valid_partition, FKP_opt.batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=FKP_opt.batch_size)

if FKP_opt.train:
    print("train first model")
    model1 = FKP_models.CNN(out_size=FKP_opt.out_size_large)
    model1 = model1.to(device)
    loss_function1 = nn.MSELoss()
    optimizer1 = optim.AdamW(model1.parameters(), lr=1e-3, weight_decay=0.01) if FKP_opt.optimizer == 'adam' else optim.SGD(
        model1.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler1 = ExponentialLR(optimizer1, gamma=0.9)
    FKP_train.train(model1, train_loader_large, valid_loader_large, loss_function1, optimizer1, scheduler1, 0)

    print("train second model")
    model2 = FKP_models.CNN(out_size=FKP_opt.out_size_small)
    model2 = model2.to(device)
    loss_function2 = nn.MSELoss()
    optimizer2 = optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=0.01) if FKP_opt.optimizer == 'adam' else optim.SGD(
        model2.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler2 = ExponentialLR(optimizer2, gamma=0.9)
    FKP_train.train(model2, train_loader_small, valid_loader_small, loss_function2, optimizer2, scheduler2, 1)
else:
    model1 = FKP_models.CNN(out_size=FKP_opt.out_size_large)
    model1 = model1.to(device)
    model1.load_state_dict(torch.load(FKP_opt.saved_model_large_path))
    model1.eval()
    predictions1 = FKP_train.predict(model1, test_loader)
    print(predictions1)

    model2 = FKP_models.CNN(out_size=FKP_opt.out_size_small)
    model2 = model2.to(device)
    model2.load_state_dict(torch.load(FKP_opt.saved_model_small_path))
    model2.eval()
    predictions2 = FKP_train.predict(model2, test_loader)

    predictions = np.hstack((predictions1, predictions2))
    pred_kp = FKP_utils.generate_pred_kps(list(train_data_large.drop('Image', axis=1).columns)+list(train_data_small.drop('Image', axis=1).columns), test_data, predictions)
    FKP_utils.show_images(pred_kp, 11, 100, save=True)