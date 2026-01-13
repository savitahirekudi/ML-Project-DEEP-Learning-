
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs

# X,y = make_blobs(n_samples = 500,n_features = 2,centers = 3,random_state = 23)
# fig = plt.figure(0)
# plt.grid(True)
# plt.scatter(X[:,0],X[:,1])
# plt.show()


# k = 7 
# clusters = {}
# np.random.seed(23)
# for idx in range(k):
#  center = 2*(2*np.random.random((X.shape[1],))-1)
#  points = []
#  cluster = {
#  'center' : center,
#  'points' : []
#  }

#  clusters[idx] = cluster

# clusters

# plt.scatter(X[:,0],X[:,1])
# plt.grid(True)
# for i in clusters:
#  center = clusters[i]['center']
#  plt.scatter(center[0],center[1],marker = '*',c = 'red')
# plt.show()

# def distance(p1,p2):
#  return np.sqrt(np.sum((p1-p2)**2))

# def assign_clusters(X, clusters):
#  for idx in range(X.shape[0]):
#     dist = []

#     curr_x = X[idx]

#     for i in range(k):
#         dis = distance(curr_x,clusters[i]['center'])
#         dist.append(dis)
#     curr_cluster = np.argmin(dist)
#     clusters[curr_cluster]['points'].append(curr_x)
#     return clusters
# def update_clusters(X, clusters):
#  for i in range(k):
#     points = np.array(clusters[i]['points'])
#     if points.shape[0] > 0:
#         new_center = points.mean(axis =0)
#         clusters[i]['center'] = new_center

#         clusters[i]['points'] = []
#  return clusters

# def pred_cluster(X, clusters):
#  pred = []
#  for i in range(X.shape[0]):
#     dist = []
#     for j in range(k):
#         dist.append(distance(X[i],clusters[j]['center']))
#     pred.append(np.argmin(dist))
#  return pred

# clusters = assign_clusters(X,clusters)
# clusters = update_clusters(X,clusters)
# pred = pred_cluster(X,clusters)

# plt.scatter(X[:,0],X[:,1],c = pred)
# for i in clusters:
#  center = clusters[i]['center']
#  plt.scatter(center[0],center[1],marker = '^',c = 'red')
# plt.show()


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Plot settings
plt.rc('figure', autolayout=True)
plt.rc('image', cmap='magma')

# Define kernel (edge detection)
kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
], dtype=tf.float32)

# Load image
image = tf.io.read_file('ganesh.jpg')
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[300, 300])

# Plot original image
img = tf.squeeze(image).numpy()
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original Gray Scale Image')
plt.show()

# Reformat image
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)

# Reformat kernel
kernel = tf.reshape(kernel, [3, 3, 1, 1])

# Convolution
image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=[1, 1, 1, 1],   # corrected
    padding='SAME'
)

# Activation (ReLU)
image_detect = tf.nn.relu(image_filter)

# Pooling
image_condense = tf.nn.pool(
    input=image_detect,
    window_shape=(2, 2),
    pooling_type='MAX',

    
    strides=(2, 2),
    padding='SAME'
)

# Plot results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(tf.squeeze(image_filter).numpy(), cmap='gray')
plt.axis('off')
plt.title('Convolution')

plt.subplot(1, 3, 2)
plt.imshow(tf.squeeze(image_detect).numpy(), cmap='gray')
plt.axis('off')
plt.title('Activation (ReLU)')

plt.subplot(1, 3, 3)
plt.imshow(tf.squeeze(image_condense).numpy(), cmap='gray')
plt.axis('off')
plt.title('Pooling')

plt.show()