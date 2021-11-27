import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
from sklearn.cluster import KMeans

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels

    Args:
        codebook ([numpy array]): [centers of clusters]
        labels ([numpy array]): [label for each pixel in clustring]
        w ([int]): [width of photo]
        h ([int]): [height of photo]

    Returns:
        [numpy array]: [recreated rbg image from labels of 
        vectorize image in clustring]
    """    
    d = 3
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            #assigning color of center of cluster for each pixel
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

#reading images
bee_img=mpimg.imread('bee.jpg')
parrots_img=mpimg.imread('parrots.jpg')

for img , name in [(bee_img , 'bee') , (parrots_img , 'parrots')]:
    
    #normalizing image
    train_img = np.array(img, dtype=np.float64) / 255
    w , h , d = img.shape
    #vectorizing image
    train_img = np.reshape(train_img, (w * h, d))
    
    for k in [2,3,4,5,6,10,15,20]:
        #performing kmeans
        kmeans = KMeans(n_clusters=k, random_state=0).fit(train_img)
        labels = kmeans.predict(train_img)
        new_img = recreate_image(kmeans.cluster_centers_, labels, w, h)
        mpimg.imsave('k='+str(k)+name +'.jpg'  , new_img)
        