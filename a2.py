from asyncore import file_dispatcher
import matplotlib.pyplot as plt
import cv2
import cv2 as cv
import os
import itertools
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import sys
from PIL import Image
from PIL import Image, ImageDraw
from utilities import find_top_matches
from part2 import find_transformation, find_translation, find_rigid, find_affine, find_projection, transform_image
import random


rp = ''

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


def transform_image(im, transform):
    
    im_array = np.asarray(im)
    height = im_array.shape[0]
    width = im_array.shape[1]

    new_im_array = np.zeros(im_array.shape)

    
    in_transform = np.linalg.inv(transform)

    for x in range(width):
        for y in range(height):

            # calculate coordinates
            new_hocoord = np.dot(in_transform, np.array([x, y, 1]))
            new_coord = (int(new_hocoord[0]/new_hocoord[2]), int(new_hocoord[1]/new_hocoord[2]))

            # check if new coordinate is out of bounds
            if(new_coord[0] >= width or new_coord[0] < 0 or new_coord[1] >= height or new_coord[1] < 0):
                continue

            # transform pixel to new coordinate
            new_im_array[y][x] = im_array[new_coord[1]][new_coord[0]]

    new_im_array = new_im_array.astype('uint8')
    new_im = Image.fromarray(new_im_array)

    return new_im

def find_transformation(n, coords):
    if n == 1:
        transform = find_translation(coords)
    if n == 2:
        transform = find_rigid(coords)
    if n == 3:
        transform = find_affine(coords)
    if n == 4:
        transform = find_projection(coords)
    
    return transform

def find_translation(coords):
    coords = coords[:2]
    t_x = coords[1][0] - coords[0][0]
    t_y = coords[1][1] - coords[0][1]
    
    return np.array([[1, 0, t_x], 
                     [0, 1, t_y], 
                     [0, 0,   1]]) 

def find_rigid(coords):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    x4, y4 = coords[3]
    
    A = np.array([[x1, -y1, 1, 0],
                  [x1,  y1, 0, 1],
                  [x3, -y3, 1, 0],
                  [x3,  y3, 0, 1]])
    
    b = np.array([x2, y2, x4, y4])
    
    try:
        x = np.linalg.solve(A, b)
    except:
        return None
    
    return np.array([[x[0], -x[1], x[2]],
                     [x[1],  x[0], x[3]],
                     [   0,    0,   1]])

def find_affine(coords):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    x4, y4 = coords[3]
    x5, y5 = coords[4]
    x6, y6 = coords[5]
    
    A = np.array([[x1, y1, 1, 0, 0, 0], 
                  [x3, y3, 1, 0, 0, 0], 
                  [x5, y5, 1, 0, 0, 0], 
                  [0, 0, 0, x1, y1, 1], 
                  [0, 0, 0, x3, y3, 1], 
                  [0, 0, 0, x5, y5, 1]])
    
    b = np.array([x2, x4, x6, y2, y4, y6])
    
    try:
        x = np.linalg.solve(A, b)
    except:
        return None
    
    return np.array([[x[0], x[1], x[2]], 
                     [x[3], x[4], x[5]], 
                     [   0,    0,   1]])

def find_projection(coords):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    x4, y4 = coords[3]
    x5, y5 = coords[4]
    x6, y6 = coords[5]
    x7, y7 = coords[6]
    x8, y8 = coords[7]
    
    A = np.array([[x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2], 
                  [0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2],
                  [x3, y3, 1, 0, 0, 0, -x3*x4, -y3*x4],
                  [0, 0, 0, x3, y3, 1, -x3*y4, -y3*y4],
                  [x5, y5, 1, 0, 0, 0, -x5*x6, -y5*x6],
                  [0, 0, 0, x5, y5, 1, -x5*y6, -y5*y6],
                  [x7, y7, 1, 0, 0, 0, -x7*x8, -y7*x8],
                  [0, 0, 0, x7, y7, 1, -x7*y8, -y7*y8]])
    
    b = np.array([x2, y2, x4, y4, x6, y6, x8, y8])
    
    try:
        x = np.linalg.solve(A, b)
    except:
        return None
    
    
    return np.array([[x[0], x[1], x[2]], 
                     [x[3], x[4], x[5]], 
                     [x[6], x[7],   1]])
    
################
# Part 3

# k = number iterations
# l = size of random subset
# n = the type of transformation
def RANSAC(matches, k, l, threshold, n=4):
    matches = np.array(matches)
    subset = np.random.choice(matches.shape[0], (l, 4))
    
    # precompute subset transformation matrices
    subset_trans = []
    for m in range(l):
        comp = subset[m]
        
        # Get coordinates in the form [(x1, y1), (x1', y1'), (x2, y2), (x2', y2')...]
        comp_coords = np.array([matches[comp[0]][0], matches[comp[0]][1], 
                                matches[comp[1]][0], matches[comp[1]][1],
                                matches[comp[2]][0], matches[comp[2]][1], 
                                matches[comp[3]][0], matches[comp[3]][1]])
        
        translation = find_transformation(n, comp_coords)
        
        # Sometimes the matrix doesn't have an inverse
        if translation is not None:
            subset_trans += [translation]
    
    best_hyp = 0
    best_err = 0
    for i in range(k):
        err = 0
        
        # Generate  a random hypothesis
        hyp_ind    = np.random.choice(matches.shape[0], 4)
        hyp_coords = np.array([matches[hyp_ind[0]][0], matches[hyp_ind[0]][1], 
                               matches[hyp_ind[1]][0], matches[hyp_ind[1]][1],
                               matches[hyp_ind[2]][0], matches[hyp_ind[2]][1], 
                               matches[hyp_ind[3]][0], matches[hyp_ind[3]][1]])
        
        # Find projection matrix
        #hyp = find_projection(hyp_coords)
        #hyp = find_translation(hyp_coords)
        hyp = find_transformation(n, hyp_coords)
        
        # If the hypothesis matrix doesn't have an inverse, skip it
        if(hyp is None):
            continue
        
        for j in range(len(subset_trans)):
            if np.square(np.linalg.norm(hyp - subset_trans[j])) < threshold:
                err += 1
        
        if(err > best_err):
            best_hyp = hyp
            best_err = err
    
    print(best_err)
    return best_hyp


if __name__ == '__main__':
    # Get the form image path
    print(sys.argv)
    r_args = sys.argv[1:]

    if sys.argv[1] == 'part1':
        k = r_args[1]
        k = int(k)
        op = r_args[-1]
        images_list = r_args[2:-1]
        # print(k, op)
        # print(images_list)

        distances_matrix = create_distance_matrix(images_list)
        model = cluster_images(images_list, distances_matrix, k)
        acc = calculate_accuracy(model, images_list)
        write_outputfile(op, model, images_list)

    elif sys.argv[1] == 'part2':
        im = Image.open(r_args[2])
        n = int(r_args[1])
        coords = [[int(x) for x in c.split(',')] for c in r_args[5:]]
        
        transform = find_transformation(n, coords)
        new_im = transform_image(im, transform)
        
        print(transform)
        new_im.save(r_args[4])

    elif sys.argv[1] == 'part3':
        im1_path = r_args[1]
        im2_path = r_args[2]
        
        im = Image.open(im1_path)

        matches = np.array(find_top_matches(im1_path, im2_path, 50))
        transformation = RANSAC(matches, 1000, 10000, 10000, n=4)
        
        transform_image(im, transformation).save(r_args[3])