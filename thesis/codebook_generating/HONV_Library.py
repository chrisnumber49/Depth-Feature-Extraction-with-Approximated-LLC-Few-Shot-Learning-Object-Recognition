import cv2
import numpy as np
import numba
import math
#from google.colab import drive
#drive.mount('/content/drive')
#img = cv2.imread('drive/My Drive/master/thesis/code/feature/person_037.png', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('person_037.png', cv2.IMREAD_GRAYSCALE)
#img2 = cv2.imread('nothing.jpg', cv2.IMREAD_GRAYSCALE)
#cv2.imshow('Image', img)
#cv2.imshow('Image2', img2)
#cv2.imwrite("Image-test.jpg", img2)
#cv2.waitKey(0)
#img.shape

@numba.jit
def compute_gradient(img):
    #compute gradient x,y
    gradient_values_x = 0.5*(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1))
    gradient_values_y = 0.5*(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1))
    
    #azimuth
    azimuth = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
    
    #zenith
    zenith = np.zeros((gradient_values_x.shape[0], gradient_values_x.shape[1]))
    for i in range(zenith.shape[0]):
        for j in range(zenith.shape[1]):
            zenith[i][j] = math.atan(np.sqrt((gradient_values_x[i][j]**2) + (gradient_values_y[i][j]**2)))
            zenith[i][j] = math.degrees(zenith[i][j])
    return azimuth, zenith

@numba.jit
def cell_gradient(azimuth, zenith, azimuth_size, zenith_size):
    azimuth_unit = 360 / azimuth_size
    zenith_unit = 90 / (zenith_size-1)
    
    #2d histogram
    orientation_centers = np.zeros((azimuth_size, zenith_size))
    for k in range(azimuth.shape[0]):
        for l in range(azimuth.shape[1]):
            #in every pixel
            gradient_azimuth = azimuth[k][l]
            gradient_zenith = zenith[k][l]
            
            #see whitch 4 bin this pixel belongs to
            min_azimuth = int(gradient_azimuth / azimuth_unit) % azimuth_size
            max_azimuth = (min_azimuth + 1) % azimuth_size
            min_zenith = int(gradient_zenith / zenith_unit) % zenith_size
            max_zenith = (min_zenith + 1) % zenith_size
            
            #first distribution
            ang_azimuth = gradient_azimuth % azimuth_unit
            dstb_min = (1 - (ang_azimuth / azimuth_unit))
            dstb_max = (ang_azimuth / azimuth_unit)
            
            #second distribution
            ang_zenith = gradient_zenith % zenith_unit
            orientation_centers[min_azimuth][min_zenith] += (dstb_min * (1 - (ang_zenith / zenith_unit)))
            orientation_centers[min_azimuth][max_zenith] += (dstb_min * (ang_zenith / zenith_unit))
            orientation_centers[max_azimuth][min_zenith] += (dstb_max * (1 - (ang_zenith / zenith_unit)))
            orientation_centers[max_azimuth][max_zenith] += (dstb_max * (ang_zenith / zenith_unit))
    return orientation_centers

@numba.jit
def concatenate_gradient(height, width, azimuth, zenith):
    cell_size = 8
    azimuth_size = 8
    zenith_size = 6

    #concatenate feature
    cell_gradient_matrix = np.zeros((int(height / cell_size), int(width / cell_size), azimuth_size, zenith_size))
    cell_gradient_vector = np.zeros((int(height / cell_size), int(width / cell_size), azimuth_size*zenith_size))
    
    #in every cell  
    for i in range(cell_gradient_matrix.shape[0]):
        for j in range(cell_gradient_matrix.shape[1]):

            cell_azimuth = azimuth[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
            cell_zenith = zenith[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]

            cell_gradient_matrix[i][j] = cell_gradient(cell_azimuth, cell_zenith, azimuth_size, zenith_size)
            
            #normalize
            mag = lambda vector: np.sqrt(np.sum(np.square(vector)))
            normalize = lambda vector, magnitude: vector / magnitude
            magnitude = mag(cell_gradient_matrix[i][j])
            if magnitude != 0:
                cell_gradient_matrix[i][j] = normalize(cell_gradient_matrix[i][j], magnitude)
            
            #feature reshape to 1d vector, 8*6 > 48
            cell_gradient_vector[i][j] = np.reshape(cell_gradient_matrix[i][j], (azimuth_size*zenith_size), order='C')
    return cell_gradient_vector

@numba.jit
def img_Vector(img):
    #resize
    width = (int(img.shape[1]/32)+1)*32
    height = (int(img.shape[0]/32)+1)*32
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    azimuth, zenith = compute_gradient(img)
    #print (azimuth.shape)
    #print (azimuth[416:424,128:136])
    #print(zenith[416:424,128:136])
    image_vector = concatenate_gradient(height, width, azimuth, zenith)
    X_shape = image_vector.shape[0]
    Y_shape = image_vector.shape[1]
    vector_size = image_vector.shape[2]
    #whole img feature reshape into N*48 array
    image_vector = np.reshape(image_vector, (X_shape*Y_shape, vector_size), order='C')
    return image_vector       #LLC stage return X_shape, Y_shape and image_vector2

