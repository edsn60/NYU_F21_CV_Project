import cv2
import torch
import os

import FKP_opt
import FKP_models

face_cascade = cv2.CascadeClassifier('./face_cascade/haarcascade_frontalface_default.xml')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kp_model = FKP_models.CNN(out_size=FKP_opt.out_size_large)
kp_model = kp_model.to(device)
kp_model.eval()

path = './LRWdata'
data_save_base = './lipdata/'
for word in os.listdir(path):
    if word.startswith('.'):
        continue
    word_path = path +"/" + word
    for data_type in os.listdir(word_path):
        if data_type.startswith('.'):
            continue
        file_path = word_path + "/" + data_type
        for file in os.listdir(file_path):
            if file.startswith('.'):
                continue
            if file.endswith('.mp4'):
                file = file_path + "/" + file
                cap = cv2.VideoCapture(file)

                # frame counter
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cnt = 0

                while cap.isOpened() and cnt < frame_count:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        frame_gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    if len(faces) == 1:
                        x, y, w, h = faces[0]
                        x1, y1, w1, h1 = x - 10, y - 10, w + 20, h + 20
                        rec = cv2.rectangle(frame, (x-5,y), (x-5+w, y+h), (0, 0, 255), 2)
                        face_img = frame_gray[y : y + h, x : x + w]
                        face_img_show = frame_gray[y1: y1 + h1, x1: x1 + w1]

                        faceROI_path = data_save_base + file.split('/')[-1].split('.')[0] + '_' + 'faceROI' + '_' + str(cnt) + '.jpg'
                        cv2.imwrite(faceROI_path, face_img_show)

                        crop_size = h1 // 2
                        center_x, center_y = w1 // 2, h1 // 2
                        cp_y = y1 + center_y
                        cp_x = x1 + center_x - crop_size // 2
                        mouth_img = frame_gray[cp_y: cp_y + crop_size, cp_x:cp_x + crop_size]
                        rec = cv2.rectangle(rec, (cp_x-5, cp_y), (cp_x-5 + crop_size, cp_y + crop_size), (0, 255, 0), 2)
                        rec_path = data_save_base + file.split('/')[-1].split('.')[0] + '_' + 'rec' + '_' + str(
                            cnt) + '.jpg'
                        mouthROI_path = data_save_base + file.split('/')[-1].split('.')[0] + '_' + 'mouthROI' + '_' + str(
                            cnt) + '.jpg'
                        cv2.imwrite(mouthROI_path, mouth_img)
                        cv2.imwrite(rec_path, rec)

                    cnt += 1

                cap.release()
                cv2.destroyAllWindows()