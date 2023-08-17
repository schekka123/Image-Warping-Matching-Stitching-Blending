from PIL import Image
import numpy as np
import sys

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

if __name__ == "__main__":
    
    im = Image.open(sys.argv[2])
    n = int(sys.argv[1])
    coords = [[int(x) for x in c.split(',')] for c in sys.argv[5:]]
    
    transform = find_transformation(n, coords)
    new_im = transform_image(im, transform)
    
    print(transform)
    new_im.save(sys.argv[4])