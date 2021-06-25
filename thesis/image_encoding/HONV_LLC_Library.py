import numpy as np
import numpy.linalg
import numpy.matlib
import time
import os
import cv2
import numba
import math


@numba.jit
def compute_gradient(img):
    gradient_values_x = 0.5*(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1))
    gradient_values_y = 0.5*(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1))

    azimuth = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)

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
    
    orientation_centers = np.zeros((azimuth_size, zenith_size))
    for k in range(azimuth.shape[0]):
        for l in range(azimuth.shape[1]):
            gradient_azimuth = azimuth[k][l]
            gradient_zenith = zenith[k][l]
            
            min_azimuth = int(gradient_azimuth / azimuth_unit) % azimuth_size
            max_azimuth = (min_azimuth + 1) % azimuth_size
            min_zenith = int(gradient_zenith / zenith_unit) % zenith_size
            max_zenith = (min_zenith + 1) % zenith_size
            
            ang_azimuth = gradient_azimuth % azimuth_unit
            dstb_min = (1 - (ang_azimuth / azimuth_unit))
            dstb_max = (ang_azimuth / azimuth_unit)
            
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

    cell_gradient_matrix = np.zeros((int(height / cell_size), int(width / cell_size), azimuth_size, zenith_size))
    cell_gradient_vector = np.zeros((int(height / cell_size), int(width / cell_size), azimuth_size*zenith_size))
    
    for i in range(cell_gradient_matrix.shape[0]):
        for j in range(cell_gradient_matrix.shape[1]):

            cell_azimuth = azimuth[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
            cell_zenith = zenith[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]

            cell_gradient_matrix[i][j] = cell_gradient(cell_azimuth, cell_zenith, azimuth_size, zenith_size)

            mag = lambda vector: np.sqrt(np.sum(np.square(vector)))
            normalize = lambda vector, magnitude: vector / magnitude
            magnitude = mag(cell_gradient_matrix[i][j])
            if magnitude != 0:
                cell_gradient_matrix[i][j] = normalize(cell_gradient_matrix[i][j], magnitude)
                
            cell_gradient_vector[i][j] = np.reshape(cell_gradient_matrix[i][j], (azimuth_size*zenith_size), order='C')
    return cell_gradient_vector

@numba.jit
def img_Vector(img):
    height, width = img.shape
    azimuth, zenith = compute_gradient(img)
    #print (azimuth.shape)
    #print (azimuth[416:424,128:136])
    #print(zenith[416:424,128:136])
    image_vector = concatenate_gradient(height, width, azimuth, zenith)
    X_shape = image_vector.shape[0]
    Y_shape = image_vector.shape[1]
    vector_size = image_vector.shape[2]
    image_vector = np.reshape(image_vector, (X_shape*Y_shape, vector_size), order='C')
    return image_vector, X_shape, Y_shape     #LLC stage return X_shape, Y_shape and image_vector2

def LLC_coding_appr(B,X,knn):
    beta = 1e-4

    nframe=X.shape[0]
    nbase=B.shape[0]
    

    #find k nearest neighbors
    XX = np.sum(np.multiply(X,X),axis=1) #sum of X^2 size = 1*N
    BB = np.sum(np.multiply(B,B),axis=1) #sum of B^2 size = 1*M
    #(X-B)^2 = X^2-2XB+B^2, size = N*M
    D = np.matrix(np.transpose(numpy.matlib.repmat(XX,nbase,1))) - 2*np.matrix(X)*np.transpose(np.matrix(B)) + np.matrix(np.matlib.repmat(BB,nframe,1))
    IDX = np.zeros(shape=(nframe, knn)) #IDX is Knn nearest index of B for each X
    for i in range(0,nframe):
        d = D[i,]  #size = 1*M
        idx = np.ravel(np.argsort(d)) #sort M for each N, the most similar to the less 
        IDX[i,] = idx[:knn] #pick the Knn most similar to N of M
    IDX = IDX.astype(int)


    #llc approximation coding
    II = np.identity(knn)
    Coeff = np.zeros(shape=(nframe, nbase)) #size of C(output) is N*M
    for i in range(0,nframe):
        idx = IDX[i,]
        z = np.zeros(shape=(knn, B.shape[1]))
        for j in range(0,knn):
            z[j,] = B[idx[j],] - X[i,] #z is B-X in Knn index => size = Knn * D(feature)  
        C = np.matrix(z)*np.transpose(np.matrix(z)) #local covariance of z, size of C = Knn * Knn
        C = C + np.matrix(II)*beta*np.trace(C) #c the covariance matrix shows difference between X and B
        w = numpy.linalg.solve(C,np.ones((knn,1)))#w are solved answers of each index of B, bigger z(B-X) smaller w, that means the more different the less score
        w = w/sum(w) #sum(w)=1
        for j in range(0,knn):
            Coeff[i,idx[j]] = w[j,]        
    #print("Coeff = ")
    #print(Coeff.shape)
    return Coeff

def LLC_pooling(B,X,pyramid,knn,X_x,X_y):

    # llc coding
    llc_codes = LLC_coding_appr(B,X,knn)  #size = N * M  
    nframe_x = X_x #set 80
    nframe_y = X_y #set 60
    dSize = B.shape[0]
    llc_codes = numpy.reshape(llc_codes, (nframe_x, nframe_y, dSize), order='C')
    
    pBins = np.array(pyramid)*np.array(pyramid) #spatial bins on each level = 1,4,16
    tBins = sum(pBins) #=21
    beta = np.matrix(np.zeros((tBins,dSize)))  #size = 21 * M, the final SPM feature of an image

    bId = -1
    pLevels = len(pyramid) #spatial levels = 3
    for i in range(pLevels): #3 time
        nBins = pyramid[i] #1, 2, 4	
        # find to which spatial bin each local descriptor belongs
        for j in range(nBins):
            for k in range(nBins):
                bId = bId + 1 #bId = 0, 1~4, 5~20
                
                fea_num_x = int(nframe_x/nBins) #80, 40, 20
                fea_num_y = int(nframe_y/nBins) #60, 30, 15
                
                llc = llc_codes[j*fea_num_x:(j+1)*fea_num_x, k*fea_num_y:(k+1)*fea_num_y]
                grid_llc = numpy.reshape(llc, (fea_num_x*fea_num_y, dSize), order='C')
                grid_code = np.amax(grid_llc,axis=0) #size of grid_code = 1*M
                beta[bId,:] =  grid_code #get the biggest M in each grid of beta
    
    beta = np.ravel(beta)  #size from 21 * M finally become 1 * 21M 
    beta = beta/math.sqrt(sum(np.square(beta)))  #Normalization
    return beta


def Img_code(img, codebook):
    pyramid=[1, 2, 4]
    knn=5
    
    width = (int(img.shape[1]/32)+1)*32
    height = (int(img.shape[0]/32)+1)*32
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    image_features, X_FeaShape, Y_FeaShape = img_Vector(img)
    
    code = LLC_pooling(codebook,image_features,pyramid,knn,X_FeaShape,Y_FeaShape)
    return code
