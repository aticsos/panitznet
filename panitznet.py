###################################################################################
### 2. PyTorch
###################################################################################
import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

print(torch.__version__)

###################################################################################
### 2. Daten einlesen
###################################################################################
'''
Prof. Panitz macht von sich tägliche Selfies. Diese lesen wir ein und verwenden Sie 
für PanitzNetz. In der Datei
    panitznet.zip 
die Sie aus dem Read.MI heruntergeladen haben befinden sich die Selfies unter 
    imgs/small/*
Führen Sie den folgenden Code aus. Passen Sie vorher ggfs. die Variable PATH an. 
Es sollten ca. 1800 Bilder der Dimension 32×32×3 eingelesen werden. Am Ende 
wird eines der Bilder als Beispiel geplottet.
HINWEIS: Sollten Sie auf dem Server supergpu arbeiten wollen, werden beim Plotten
         von Daten evtl. noch Fehler auftauchen. Wir besprechen dies im Praktikum.
'''

from datetime import timedelta, date
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

PATH = 'imgs/small'  # FIXME
D = 32


def read_jpg(path):
    '''liest ein JPEG ein und gibt ein DxDx3-Numpy-Array zurück.'''
    img = Image.open(path)
    w, h = img.size
    # schneide etwas Rand ab.
    img = img.crop((5, 24, w - 5, h - 24))
    # skaliere das Bild
    img = img.resize((D, D), Image.ANTIALIAS)
    img = np.asarray(img)
    return img


def read_panitz(directory):
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    start_date = date(2010, 10, 30)
    end_date = date(2019, 1, 1)

    imgs = []

    for date_ in daterange(start_date, end_date):
        img_path = '%s/small-b%s.jpg' % (directory, date_.strftime("%Y%m%d"))
        if os.path.exists(img_path):
            img = read_jpg(img_path)
            imgs.append(img)

    return np.array(imgs)


imgs = read_panitz(PATH)

print('Dimension der gelesenen Bilder:', imgs.shape)

# zeigt ein Bild
plt.imshow(imgs[17])
plt.show()

###################################################################################
### 3. Hifsmethode zum Plotten
###################################################################################
'''
Während wir PanitzNet trainieren, möchten wir beobachten wie die Rekonstruktionen
des Netzes den Eingabebildern immer ähnlicher werden. Hierzu können Sie die 
folgende Methode verwenden: Übergeben Sie eine Liste von z.B. 10 Bildern (imgs) 
und die  zugehörigen Rekonstruktionen Ihres Netzes (recs) als Listen von 
numpy-Arrays. Es sollte ein Plot erstellt werden, in dem Sie neben jedem Bild 
die Rekonstruktion sehen, ähnlich dem Bild
   panitzplot.png
Überprüfen Sie kurz die Methode, indem Sie 10 zufällige Bilder und (anstelle der 
noch nicht vorhandenen Rekonstruktionen) noch einmal dieselben Bilder übergeben. 
'''
import math


def plot_reconstructions(imgs, recs, iteration=None):
    recs = recs.detach().numpy()
    # Erstellt ein NxN-Grid zum Plotten der Bilder
    N = int(np.ceil(math.sqrt(2 * len(imgs))))
    f, axarr = plt.subplots(nrows=N, ncols=N, figsize=(18, 18))

    # Fügt die Bilder in den Plot ein
    for i in range(min(len(imgs), 100)):
        axarr[2 * i // N, 2 * i % N].imshow(imgs[i].reshape((D, D, 3)),
                                            interpolation='nearest')
        axarr[(2 * i + 1) // N, (2 * i + 1) % N].imshow(recs[i].reshape((D, D, 3)),
                                                        interpolation='nearest')
    f.tight_layout()
    # plt.show()
    plt.savefig('recs/recs-%.4d.png' % iteration)
    plt.close()


###################################################################################
### 4. Vorverarbeitung
###################################################################################
'''
Momentan ist jedes der Bild noch ein D×D×3-Tensor. Machen Sie hieraus einen 
eindimensionalen Vektor. Skalieren Sie den Pixelbereich außerdem von 0,...,255 
auf [0,1].
'''
imgs = np.reshape(imgs, (len(imgs), D * D * 3)).astype("float")
imgs /= 255.

###################################################################################
### 5. Sie sind am Zug!
###################################################################################
'''
Implementieren Sie PanitzNet, d.h. erstellen Sie die Netzstruktur und trainieren
Sie Ihr Netz. Orientieren Sie sich am in der Vorlesung vorgestellten Programmgerüst.
'''
import torch.nn as nn


class PanitzNet(nn.Module):
    def __init__(self, D=0):
        super(PanitzNet, self).__init__()

        self.l_in = nn.Linear(D * D * 3, 1000)
        self.l_1 = nn.Linear(1000, 100)
        self.l_2 = nn.Linear(100, 50)
        self.l_3 = nn.Linear(50, 100)
        self.l_4 = nn.Linear(100, 1000)
        self.l_out = nn.Linear(1000, D * D * 3)

    def forward(self, x):
        x = nn.functional.sigmoid(self.l_in(x))
        x = nn.functional.sigmoid(self.l_1(x))
        x = nn.functional.sigmoid(self.l_2(x))
        x = nn.functional.sigmoid(self.l_3(x))
        x = nn.functional.sigmoid(self.l_4(x))
        x = nn.functional.sigmoid(self.l_out(x))

        return x


import torch.optim as optim


def run_training(net, imgs, iters):
    i = 1

    optimizer = torch.optim.Adam(net.parameters())

    for iter in iters:
        imgs_float = imgs.float()
        i += 1
        y_ = net(imgs_float)
        loss = nn.functional.mse_loss(y_, imgs_float)

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"iter: {i}, Loss: {loss.item()}")
            plot_reconstructions(imgs_float[5:16], net(imgs_float[5:16]), i)


# define the network+data, and train.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
net = PanitzNet(D=D)
net.to(device)
run_training(net, torch.from_numpy(imgs), (i for i in range(100000000)))