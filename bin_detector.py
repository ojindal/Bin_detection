'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import cv2
from skimage.measure import label, regionprops
from numpy.linalg import inv
import math

class BinDetector():
    
    def __init__(self):
        
        
        '''
        Initilized the bin detector with attributes as the parameters
        that comes in Gaussian Discriminator.
        
        '''
        self.mu = np.loadtxt('bin_detection/mu.txt')

        v1 = np.loadtxt('bin_detection/var1.txt')
        v2 = np.loadtxt('bin_detection/var2.txt')
        v3 = np.loadtxt('bin_detection/var3.txt')
        v4 = np.loadtxt('bin_detection/var4.txt')

        self.var = np.array([v1,v2,v3,v4])

        self.theta = np.loadtxt('bin_detection/theta.txt')
        
        pass
                    
    def segment_image(self, img):
        '''
			Obtain a segmented image using Gaussian Discriminator as a color classifier
			
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image belongs to bin_blue and 0 otherwise.
		'''
        
        def decesion(X):
            
            '''
            Classify a set of pixels into {0,1}: 1 if the pixel belongs to the bin_blue category
            and Zero otherwise.

            Inputs:
              X: n x 3 matrix of RGB values
            Outputs:
              y: n x 1 vector of with one of the {0,1} values corresponding to {bin_blue, others(3)}, respectively
            '''
            ''' Splitting covariances of different classes as v1,v2,v3,v4 '''
            
            v1 = np.array(self.var[0:1, 0:3])
            v2 = np.array(self.var[1:2, 0:3])
            v3 = np.array(self.var[2:3, 0:3])
            v4 = np.array(self.var[3:4, 0:3])
            
            ''' Calculating inverse 'inv1..' and determinants 'det1..' of those covariances'''

            inv1, det1 = inv(v1), np.linalg.det(v1)
            inv2, det2 = inv(v2), np.linalg.det(v2)
            inv3, det3 = inv(v3), np.linalg.det(v3)
            inv4, det4 = inv(v4), np.linalg.det(v4)
            
            y = []
            for i in range(len(X)):
                
                ''' Calculating the two components of Mahalanobis Distance of each pixel
                    with all the four classes that appers in multi-variate gaussian discriminator '''
                
                d1 = np.matmul(np.matmul(np.transpose(X[i]/255- self.mu[0]), inv1),(X[i]/255- self.mu[0]))
                a1 = (((math.log(2*3.14))**3)*det1) - 2*math.log(self.theta[0])

                d2 = np.matmul(np.matmul(np.transpose(X[i]/255- self.mu[1]), inv2),(X[i]/255- self.mu[1]))
                a2 = (((math.log(2*3.14))**3)*det2) - 2*math.log(self.theta[1])

                d3 = np.matmul(np.matmul(np.transpose(X[i]/255- self.mu[2]), inv3),(X[i]/255- self.mu[2]))
                a3 = (((math.log(2*3.14))**3)*det3) - 2*math.log(self.theta[2])
                
                d4 = np.matmul(np.matmul(np.transpose(X[i]/255 - self.mu[3]), inv4),(X[i]/255 - self.mu[3]))
                a4 = (((math.log(2*3.14))**3)*det4) - 2*math.log(self.theta[3])
                
                ''' Appending all the mahalanobis distances into a list to see which one is minimum '''

                l = [d1+a1,d2+a2,d3+a3,d4+a4]
                
                '''
                if the pixel location a 3-D euclidean space is closest to the class 1 i.e., bin_blue,
                then '1' is appended and '0' otherwise.
                
                '''
                
                if min(l) == l[0]:
                    y.append(1)
                else:
                    y.append(0)
                    
                    
            return np.array(y)


        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ''' an RGB mask and a corresponding binary image are initiated '''
        
        mask = np.zeros((int(np.shape(img)[0]),int(np.shape(img)[1]),3))
        bi = np.zeros((int(np.shape(img)[0]),int(np.shape(img)[1])))

        img_X = img.reshape((img.size//3, 3))
        mask_y = np.array(decesion(img_X))
        bi = mask_y.reshape((img.shape[0], img.shape[1]))
        mask = np.dstack([bi, bi, bi])
        
        return bi

    def get_bounding_boxes(self, bi):
        
        '''
        
        Finding the bounding boxes of the recycling bins
		by calling segment_image function
		
		Inputs:
			img - binary image output of segment_image function
		Outputs:
			boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
			where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
        
        '''
        
        b_i = bi.astype(np.uint8)
        contours = cv2.findContours(b_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        boxes = []
        for c in contours:
            
            ''' a bounding rectangle will be generated '''
            
            x, y, width, height = cv2.boundingRect(c)
            
            ''' Conditions on the bounding boxes so that only the required ones will be displayed '''
            
            if height*width > 4000 and width < height <2*width :
                boxes.append([x, y, width+x, height+y])
                
        return boxes
    def train(self):

        '''
        The training is done according to the Multi-variate Gaussian Discriminator.
        
        The data from three folders is loaded as X1,X2,X3,X4
        then the classes are assigned in vector y as given
        '''
        
        x1 = np.array(binn)
        x2 = np.array(gy)
        x3 = np.array(rb)
        x4 = np.array(bnb)
        
        y1 = np.full(x1.shape[0],1)
        y2 = np.full(x2.shape[0], 2)
        y3 = np.full(x3.shape[0],3)
        y4 = np.full(x4.shape[0],4)
        
        X, y = np.concatenate((x1,x2,x3,x4)), np.concatenate((y1,y2,y3,y4))
        '''
        number of datapoints in each class is calculated
        '''
        c1,c2,c3,c4 = len(y1),len(y2),len(y3),len(y4)
        total_l = len(y)
        self.theta = np.array([c1/total_l,c2/total_l,c3/total_l,c4/total_l])
        
        '''
        datapoints are split according to each class again and then
        the code runs to calculate three parameters of the Multi-variate Gaussian Discriminator
        i.e., mean, variance and theta.
        '''


        self.mu = np.zeros((4, 3)) #classes,dim
        
        mu1 =np.sum(np.array(x1),axis=0)/c1
        mu2 = np.sum(np.array(x2),axis=0)/c2
        mu3 = np.sum(np.array(x3),axis=0)/c3
        mu4 = np.sum(np.array(x4),axis=0)/c4
        
        self.mu = np.concatenate(([mu1],[mu2],[mu3],[mu4]))
        
        v1=np.cov(x1.T)
        v2=np.cov(x2.T)
        v3=np.cov(x3.T)
        v4=np.cov(x4.T)
        
        '''
        everything calculated parameter is saved as text files
        so that it can be called anywhere in the class.
        '''
        
        self.var = np.concatenate(([v1],[v2],[v3],[v4]))
        np.savetxt('mu.txt', self.mu, fmt = "%s")
        np.savetxt('var1.txt', self.var[0], fmt = "%s")
        np.savetxt('var2.txt', self.var[1], fmt = "%s")
        np.savetxt('var3.txt', self.var[2], fmt = "%s")
        np.savetxt('var4.txt', self.var[3], fmt = "%s")
        np.savetxt('theta.txt', self.theta, fmt = "%s")