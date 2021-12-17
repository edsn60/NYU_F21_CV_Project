import cv2
import matplotlib.pyplot as plt
import os

fig3 = plt.figure(figsize=(15, 15))


imgs = []
data_path = './lipdata/DRAW'

imgdir = {'action':[], 'national': [], 'strong':[], 'yesterday':[]}

words = ['action', 'national', 'strong', 'yesterday']
for word in words:
    path = data_path + '/' + word
    paths = []
    for filename in os.listdir(path):
        if filename.startswith('.'):
            continue
        paths.append(filename)
    paths.sort()

    for img in paths:
        imgpath = path + '/' + img
        opened = cv2.imread(imgpath)
        opened = opened[..., ::-1]
        imgdir[word].append(opened)


fig = plt.figure(figsize=(40, 14))
cols, rows = 11, 4
for i in range(1, cols * rows + 1):
    base = words[(i-1)//11]
    img = imgdir[base][(i-1)%11]
    fig.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.savefig("sample_crop_face.png", dpi=fig.dpi)
plt.show()
