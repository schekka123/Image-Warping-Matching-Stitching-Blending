from PIL import Image, ImageDraw
import numpy as np
from utilities import find_top_matches
from part2 import find_transformation, find_translation, find_rigid, find_affine, find_projection, transform_image
import random
import sys

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

if __name__ == "__main__":
    im1_path = sys.argv[1]
    im2_path = sys.argv[2]
    
    im = Image.open(im1_path)

    matches = np.array(find_top_matches(im1_path, im2_path, 50))
    transformation = RANSAC(matches, 1000, 10000, 10000, n=4)
    
    transform_image(im, transformation).save(sys.argv[3])