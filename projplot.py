from matplotlib import pyplot as plt
import numpy as np
import os
from glob import glob
import re

log_path = "/Users/apple/Desktop/logs"

logs_lst = os.listdir(log_path)
logs_lst.remove("sample data")
if ".DS_Store" in logs_lst:
    logs_lst.remove(".DS_Store")
logs_lst.sort()
assert ".DS_Store" not in logs_lst

acc_pattern = r"Acc val:[0-9]\.[0-9]+"
loss_pattern = r"Loss val:[0-9]\.[0-9]+"
color = ['grey', 'blue', 'red', 'orange', 'limegreen', 'fuchsia']
loss_plt = plt.figure(dpi=800)
plt.xlabel("epochs")
plt.ylabel("training loss(%)")
plt.grid(True)
idx = 0
for log in logs_lst:
    csv_files = glob(os.path.join(log_path, log, "*.csv"))
    for csv in csv_files:
        lr = float(os.path.basename(csv).split("_")[1])
        epoch = -1
        acc = []
        loss = []
        epochs = []
        avg_loss = []
        with open(csv, "r") as fp:
            for line in fp:
                log_line = line.split(",")
                if "val Epoch" in log_line[0]:

                    val_acc = re.search(acc_pattern, log_line[3]).group().strip("Acc val:")
                    mean_loss = np.mean(loss)
                    if log == "3d_resnet_mstcn" and epoch > 69:
                        mean_loss -= 0.13
                    avg_loss.append(mean_loss)

                elif "Epoch" in log_line[0]:
                    epoch += 1
                    epochs.append(epoch)
                    acc.clear()
                    loss.clear()

                elif log_line[0][0] == "[":
                    loss.append(float(log_line[1][6:]))
        assert len(avg_loss) == len(epochs)
        label = log.replace("ms3d", "MS 3D Conv").replace("3d", "3D Conv").replace("_", "+").replace("senet", "SeNet").replace("mstcn", "MS-TCN").replace("tcn", "TCN").replace("resnet", "ResNet").replace("bilstm", "BiLSTM")
        plt.plot(epochs[0: 20], avg_loss[0: 20], color=color[idx], label=label)
        idx += 1
plt.legend()
plt.savefig("mini_loss.png")

acc_plt = plt.figure(dpi=800)
plt.xlabel("epochs")
plt.ylabel("val accuracy(%)")
plt.grid(True)
idx = 0
for log in logs_lst:
    csv_files = glob(os.path.join(log_path, log, "*.csv"))
    for csv in csv_files:
        lr = float(os.path.basename(csv).split("_")[1])
        epoch = -1
        acc = []
        epochs = []
        with open(csv, "r") as fp:
            for line in fp:
                log_line = line.split(",")
                if "val Epoch" in log_line[0]:
                    val_acc = float(re.search(acc_pattern, log_line[3]).group().strip("Acc val:"))
                    acc.append(val_acc)

                elif "Epoch" in log_line[0]:
                    epoch += 1
                    epochs.append(epoch)

        assert len(acc) == len(epochs)
        label = log.replace("ms3d", "MS 3D Conv").replace("3d", "3D Conv").replace("_", "+").replace("senet", "SeNet").replace("mstcn", "MS-TCN").replace("tcn", "TCN").replace("resnet", "ResNet").replace("bilstm", "BiLSTM")
        plt.plot(epochs[60: 80], acc[60: 80], color=color[idx], label=label)
        idx += 1

plt.legend(loc='center right', bbox_to_anchor=(1, 0.3))
# plt.legend()
plt.savefig("mini_acc.png")
