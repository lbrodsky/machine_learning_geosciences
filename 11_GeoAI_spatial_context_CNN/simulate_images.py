# Simulate images with shapes: ['square', 'triangle', 'circle']  

import numpy as np
np.random.seed(42) 
import random
import matplotlib.pyplot as plt

object_shapes = ['square',  'triangle', 'circle'] 

def get_random_location(width, height, zoom=1.0):
    x = int(width * random.uniform(0.1, 0.9))
    y = int(height * random.uniform(0.1, 0.9))

    size = int(min(width, height) * 0.1 * zoom) # random.uniform(0.06, 0.12)

    return (x, y, size)
    
def add_filled_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

    return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, yy > y - s, yy < y + s]))


def add_triangle(arr, x, y, size):
    s = int(size / 2)

    triangle = np.tril(np.ones((size, size), dtype=bool))

    arr[x-s:x-s+triangle.shape[0],y-s:y-s+triangle.shape[1]] = triangle

    return arr


def add_circle(arr, x, y, size, fill=False):
    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]
    circle = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    new_arr = np.logical_or(arr, np.logical_and(circle < size, circle >= size * 0.7 if not fill else True))

    return new_arr

def logical_and(arrays):
    new_array = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        new_array = np.logical_and(new_array, a)

    return new_array


def array_to_colorimg(array, color_ix=0):
    colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56)])
    color = colors[color_ix]
    colorimg = np.ones((3, array.shape[0], array.shape[1]), dtype=np.float32) * 255
    
    i = 0 
    for i in range(len(color)): 
        colorimg[i, :, :] *= (array > 0) * color[i] + 1
        i+=1

    return colorimg.astype(np.uint8)


def generate_img_and_mask(height, width, noise, object_shape):
    shape = (height, width)
    object_shapes = ['square', 'triangle', 'circle'] 

    # Create input image
    arr = np.zeros(shape) 

    if object_shape == 'square': 
        square_location = get_random_location(*shape, zoom=3)
        arr = add_filled_square(arr, *square_location)  
        masks = np.asarray([
            add_filled_square(np.zeros(shape, dtype=bool), *square_location) 
            ]).astype(np.float32)

    elif object_shape == 'triangle': 
        triangle_location = get_random_location(*shape, zoom=2)
        arr = add_triangle(arr, *triangle_location)        
        masks = np.asarray([
            add_triangle(np.zeros(shape, dtype=bool), *triangle_location)
            ]).astype(np.float32)

    elif object_shape == 'circle': 
        circle_location = get_random_location(*shape, zoom=2)
        arr = add_circle(arr, *circle_location)
        masks = np.asarray([
            add_circle(np.zeros(shape, dtype=bool), *circle_location, fill=True)
            ]).astype(np.float32)

    # arr_color = array_to_colorimg(masks[0,:,:] * 1, color_ix=0)
    
    object_ix = object_shapes.index(object_shape)
    
    if noise is None: 
        noise = 0.0
    
    image = masks + (np.random.random_sample(shape) * noise)
    
    return  image, masks, object_ix 


def generate_image_shapes(height, width, count, noise):
    """Simulate images with 3 shapes at random position
       Return: X, Y, Y_label 
    """
    # randomize object selection 
    object_shapes = ['square', 'triangle', 'circle'] 
    rnd_idx = np.random.randint(0,len(object_shapes), count)
    x, y, y_label = zip(*[generate_img_and_mask(height, width, noise, object_shape=object_shapes[rnd_idx[i]]) for i in range(0, count)])

    X = np.asarray(x, dtype=np.float32) # * 255
    Y = np.asarray(y, dtype=np.float32)
    Y_label = np.asarray(y_label) # , dtype=np.float32

    return X, Y, Y_label


