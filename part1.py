from asyncore import file_dispatcher
import matplotlib.pyplot as plt
import cv2
import cv2 as cv
import os
import itertools
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import sys

orb = cv2.ORB_create(nfeatures=1000)

def find_matches_v0(ip1, ip2, thresh = 0.8):
    
    points_matched = 0
    # you can increase nfeatures to adjust how many features to detect 
    orb = cv2.ORB_create(nfeatures=1000)

    #Get Descriptors for image 1
    img1 = cv2.imread(ip1, cv2.IMREAD_GRAYSCALE)

    # detect features 
    (keypoints1, descriptors1) = orb.detectAndCompute(img1, None)

    #Get Descriptors for image 2
    img2 = cv2.imread(ip2, cv2.IMREAD_GRAYSCALE)

    # detect features 
    (keypoints2, descriptors2) = orb.detectAndCompute(img2, None)

    for d1ind, d1 in enumerate(descriptors1):
    
        descriptor_distances = []
        for d2 in descriptors2:
            descriptor_distances.append(cv2.norm( d1, d2, cv2.NORM_HAMMING))

        min_index = descriptor_distances.index(min(descriptor_distances))
        closest_d = descriptor_distances[min_index]
        descriptor_distances[min_index] = 10**10
        min_index2 = descriptor_distances.index(min(descriptor_distances))
        second_closest_d = descriptor_distances[min_index2]
        closest_point_index = min_index
        sec_closest_point_index = min_index2

        ratio = closest_d/second_closest_d

        if ratio < thresh:
          points_matched += 1

          k1 = keypoints2[closest_point_index]
          k2 = keypoints2[sec_closest_point_index]

          for j in range(-10, 10):
              img1[int(keypoints1[d1ind].pt[1])+j, int(keypoints1[d1ind].pt[0])+j] = 0 
              img1[int(keypoints1[d1ind].pt[1])-j, int(keypoints1[d1ind].pt[0])+j] = 255 

              img2[int(k1.pt[1])+j, int(k1.pt[0])+j] = 0 
              img2[int(k1.pt[1])-j, int(k1.pt[0])+j] = 255 

    return points_matched

#Count The Number Of Feature Point Matches Between 2 images
def find_matches(ip1,ip2, thresh = 0.75):
  #Get Descriptors for image 1
  img1 = cv2.imread(ip1, cv2.IMREAD_GRAYSCALE)

  # detect features 
  (keypoints1, descriptors1) = orb.detectAndCompute(img1, None)

  #Get Descriptors for image 2
  img2 = cv2.imread(ip2, cv2.IMREAD_GRAYSCALE)

  # detect features 
  (keypoints2, descriptors2) = orb.detectAndCompute(img2, None)


#############################################################################
# The following article was referred to for writing the below lines of code.
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

  bf = cv.BFMatcher(cv.NORM_HAMMING)
  
  matches = bf.knnMatch(descriptors1,descriptors2,k=2)
  # Check Ratio Against Threshold
  filtered_matches = []
  for m,n in matches:
      if m.distance < thresh*n.distance:
          filtered_matches.append([m])

# The following article was referred to for writing the above lines of code.
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
#############################################################################

  # Optional Code To Plot Matches Plot Matches
  # img3 = cv.drawMatchesKnn(img1,keypoints1,img2,keypoints2,filtered_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
  # plt.imshow(img3),plt.show()
  
  return len(filtered_matches)


#Create Distance Matrix

thresh = 0.8

def create_distance_matrix(images_list, thresh =thresh):
    distances_matrix = np.zeros((len(images_list), len(images_list)))
    for ii in range(len(images_list)-1):
      print(ii)
      for ii2 in range(len(images_list)-1):
        if ii2 != ii:
            matches = find_matches(rp+images_list[ii],rp+images_list[ii2], thresh)
            distances_matrix[ii,ii2] = matches

    return distances_matrix



def cluster_images(images_list, distances_matrix, k=10):
    #Transform similarity measure to distance measure
    x = distances_matrix.max()
    x = int(x)
    distances_matrix =  x - distances_matrix

#############################################################################
# The following article was referred to for writing the below lines of code.
# https://stackoverflow.com/questions/47321133/sklearn-hierarchical-agglomerative-clustering-using-similarity-matrix

    #Cluster the images
    model = AgglomerativeClustering(affinity='precomputed', n_clusters=k, linkage='complete').fit(distances_matrix)

# The following article was referred to for writing the above lines of code.
# https://stackoverflow.com/questions/47321133/sklearn-hierarchical-agglomerative-clustering-using-similarity-matrix
###############################################################
   
    return model


#Calculate Pairwise Clustering Accuracy
def calculate_accuracy(model, images_list):
    #Calculate Accuracy
    image_pairs = list(itertools.combinations(iter(images_list),2))

    tp = 0
    tn = 0
    for i1,i2 in image_pairs:
      if i1 != i2:
    #check if same cluster or not
        if i1.split('_')[0] == i2.split('_')[0]:
          if model.labels_[images_list.index(i1)] == model.labels_[images_list.index(i2)]:
            tp += 1

        else:
          if model.labels_[images_list.index(i1)] != model.labels_[images_list.index(i2)]:
            tn += 1

    acc = (tp+tn)/len(image_pairs)
    print('Accuracy:', acc)
    return acc


#Write Results To Output txt file
def write_outputfile(op, model, images_list):
    # Write Clusters to output file
    cluster_lists = {}
    for i in range(10):
      cluster_lists[i] = []
      for ind, j in enumerate(model.labels_):
        if j == i:
            cluster_lists[i].append(images_list[ind])

    for i in cluster_lists:
        cluster_lists[i] = ' '.join(cluster_lists[i])

    cluster_lines = cluster_lists.values()
    cluster_lines = list(cluster_lines)

    op = 'output.txt'

    with open(op, 'w') as f:
        f.write('\n'.join(cluster_lines))


if __name__ == '__main__':
    # Get the form image path
    print(sys.argv)
    k = sys.argv[1]
    k = int(k)
    op = sys.argv[-1]
    images_list = sys.argv[2:-1]
    print(k, op)
    print(images_list)

    distances_matrix = create_distance_matrix(images_list)
    model = cluster_images(images_list, distances_matrix, k)
    acc = calculate_accuracy(model, images_list)
    write_outputfile(op, model, images_list)