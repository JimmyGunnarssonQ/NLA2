import matplotlib.image as im
import matplotlib.pyplot as plt 
import numpy as np 
import numpy.linalg as nl 

#flag = im.imread("./Yugoslav.png")
image = im.imread("./kvinnagr.jpg")

def imagtogray(image):
    grayscale = [0.2989, 0.5870, 0.1140]
    grayimage = np.dot(flag[...,:3], grayscale)
    return grayimage 

u,s,vt = nl.svd(image)

maxi = max(s)

sigma = np.zeros((u.shape[0], vt.shape[0]))

j=0
for i in s:
    i/=maxi 
    if i<3e-2:
        j+=1
        pass 
    else:
        sigma[j,j] = maxi*i
        j+=1 

A = u@sigma@vt 

plt.imshow(A, cmap = "gray")
plt.show()