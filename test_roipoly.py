'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import pickle
import os, cv2
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

    # read the first training image
    folder = 'data/training/bin_blue'
    filename = '0001.jpg'  
    img = cv2.imread(os.path.join(folder,filename))
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

    temp = np.empty([n,3])
    for i in range(n):
        temp[i] = img[x[i],y[i]].astype(np.float64)/255


    # display the labeled region and the image mask
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('%d pixels selected\n' % img[mask,:].shape[0])

    ax1.imshow(img)
    ax1.add_line(plt.Line2D(my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
    ax2.imshow(mask)

    plt.show(block=True)
    
    with open('test.pickle', 'wb') as f:
        pickle.dump(temp, f)




