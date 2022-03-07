'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import pickle
import os, cv2
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import numpy as np

def twin_mask(folder, file1,file2):

    X = np.empty([1,3])
    Y = np.empty([1,3])
    for filename in os.listdir(folder):
        if filename[-3:] != 'jpg':
            continue

        # read the first training image
        
        img = cv2.imread(os.path.join(folder,filename))
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # display the image and use roipoly for labeling
        fig, ax = plt.subplots()
        ax.imshow(img)
        my_roi = RoiPoly(fig=fig, ax=ax, color='r')

        # get the image mask
        mask = my_roi.get_mask(img)

        # x and y coordinates for the pixels in mask
        [x,y] = np.where(np.array(mask)==True)
        n = len(x)

        temp_r = np.empty([n,3])
        temp_y = np.empty([n,3])
        for i in range(n):
            temp_r[i] = img[x[i],y[i]].astype(np.float64)/255
            temp_y[i] = yuv[x[i],y[i]].astype(np.float64)/255
        
        X = np.concatenate((X,temp_r))
        Y = np.concatenate((Y,temp_y))

        # display the labeled region and the image mask
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('%d pixels selected\n' % img[mask,:].shape[0])

        ax1.imshow(img)
        ax1.add_line(plt.Line2D(my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
        ax2.imshow(mask)

        plt.show(block=True)
        
    with open('{}.pickle'.format(file1), 'wb') as r:
        pickle.dump(X, r)
    with open('{}.pickle'.format(file2), 'wb') as y:
        pickle.dump(Y, y)
        
#folder = 'data/training/bin_blue'
#X1 = twin_mask(folder,'bin_blue_rgb','bin_blue_yuv')
        
#folder = 'data/training/blue_not_bin'
#X2 = twin_mask(folder,'blue_not_bin_rgb','blue_not_bin_yuv')

#folder = 'data/training/green_yel'
#X3 = twin_mask(folder,'green_yel_rgb','green_yel_yuv')

folder = 'data/training/red_black'
X4 = twin_mask(folder,'red_black_rgb','red_black_yuv')





