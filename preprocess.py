import sys
import cv2
import numpy as np
import os
from tqdm import tqdm

delta = 3
resize = (20, 20)
face_cascade = cv2.CascadeClassifier(
    r'C:\ProgramData\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')


def align(path):
    if not os.path.exists(path + "_cropped"):
        os.mkdir(path + "_cropped")
    for file in tqdm(os.listdir(path)):
        img = cv2.imread(path + "/" + file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10))
        for (x, y, w, h) in faces:
            cv2.imwrite(path + "_cropped/" + file,
                        # cv22.resize(img[y - delta:y + h + delta, x - delta:x + w + delta], resize),
                        img[y - delta:y + h + delta, x - delta:x + w + delta],
                        [cv2.IMWRITE_JPEG_QUALITY, 100])


def align_folder(path):
    for folder in tqdm(os.listdir(path)):
        if not os.path.exists(path + "_cropped/" + folder):
            os.mkdir(path + "_cropped/" + folder)
        for file in os.listdir(path + "/" + folder):
            img = cv2.imread(path + "/" + folder + "/" + file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.0, minNeighbors=10, minSize=(10, 10))
            for (x, y, w, h) in faces:
                cv2.imwrite(path + "_cropped/" + folder + "/" + file,
                            cv2.resize(img[y - delta:y + h + delta, x - delta:x + w + delta], resize)
                            [cv2.IMWRITE_JPEG_QUALITY, 100])


def delete(path):
    os.chdir(path)
    for folder in os.listdir(path):
        for f in os.listdir(folder):
            if "._" in f:
                print(f)
                os.remove(os.path.join(path, folder, f))


if __name__ == '__main__':
    delete(r"C:\Users\Jeffy\Downloads\Data\hw2\hw2p2_check\train_data\large")
