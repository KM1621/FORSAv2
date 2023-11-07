# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 19:34:06 2023

@author: Huruy FORSA util file
"""
import numpy as np
from fxpmath import Fxp
from numpy import random
from numpy import *

#Sorting full matrix
# Accepts transposed weight matrix and reorders rows to get minmum switching possible
# Returns new matrix with low switching between rows
def count_switching(mat1, word_size=8, frac_size=6):
    height = np.size(mat1, 0)
    width = np.size(mat1, 1)
    sw_vector = np.zeros(height-1)
    for row in range(height - 1): 
        temp0=0
        for col in range(width):
            row0 = Fxp(mat1[row][col], True, word_size, frac_size)
            row1 = Fxp(mat1[row + 1 ][col], True, word_size, frac_size)
            temp0 = temp0 + (row0 ^ row1).bin().count('1')
        sw_vector[row] =  temp0      
    return sw_vector

def sortFullMatrix(mat_in, word_size=8, frac_size=6):
    mat1 = mat_in
    height = np.size(mat1, 0)
    width = np.size(mat1, 1)
    switchingact = np.zeros(np.size(mat1, 0))
#     original_index=[]
    original_index = np.asanyarray([i for i in range(height)])
    sw_mat = []
    for row in range(height - 2): # Iterate until total rows - 2 or 3
#         print('starting with row :', row)
        sw_vector = np.zeros(height)
        sw_row = []
        for ii in range(row+1, height): # Iterate until total rows - 2 or 3
#             print('comparing switching between row :', row, 'and row', ii)
            temp0=0
            for col in range(width):
                row0 = Fxp(mat1[row][col], True, word_size, frac_size)
                row1 = Fxp(mat1[ii ][col], True, word_size, frac_size)
                temp0 = temp0 + (row0 ^ row1).bin().count('1')
            sw_vector[ii] =  temp0
            sw_row.append(sw_vector[ii])
#         print('Switching vector', sw_vector)
#         new_order = np.argsort(sw_vector)
        sw_mat.append(sw_row)
        new_row = np.argmin(sw_vector[row + 1:]) + row + 1  # Adding 1 to cancel the offset from removing (row + 1)th index
#         print('matrix before sorting')
#         print(mat1)
        # switch row i+1 and row i+2
        
        mat1[[row+1, new_row]] = mat1[[new_row, row+1]]
        original_index[[row+1, new_row]] = original_index[[new_row, row+1]]
#         print('original_index after being sorted:', original_index)
#         print('matrix after sorting')
#         print(mat1)
    return mat1, original_index # Returning original_index to rearrange activation maps

# weights = [
#     [60, 40, 50],
#     [90, 70, 80],
#     [20, 10, 20]
# ]

# random.seed(1)
# weights = random.randint(0, 100, (4, 4))
# print(weights)
# random.seed(1)
# weights = random.randint(0, 100, (4, 4))
# print(weights)

# # print(np.asarray(weights).shape)
# matrix_out = sortFullMatrix(np.asarray(weights))

rnd_from = -10
rnd_to = 10
def randMatMult(array_sizey, array_sizex, array_sizek):
    array_sizek = 5
    myArray = OSSystolicArray(array_sizey, array_sizex, array_sizek)

#     random.seed(4)
    activations = random.randint(rnd_from, rnd_to, (array_sizey, array_sizek))
#     print('Printing activations')
#     print(activations.tolist())
    weights = random.randint(rnd_from, rnd_to, (array_sizek, array_sizex))
#     print('Printing weights')
#     print(weights.tolist())

    myArray.fill_activations(activations.tolist())
#     myArray.fill_activations(activations)

    # myArray.fill_weights(np.transpose(weights.tolist()))
    myArray.fill_weights(weights.tolist())
    # myArray.fill_weights(weights)

    res_before, total_sw_before = myArray.run()
#     print('total_sw_before is :', total_sw_before)
#     print('res :', res)
#     print('activations.shape', activations.shape)
#     print('weights.shape', weights.shape)

#     print('result without sorted weights matrix')
    print(np.transpose(res_before).shape)
#     print('Expected without sorted weights matrix')
#     print(np.matmul(activations, weights))
    assert (np.transpose(res_before) == np.matmul(activations, weights)).all()
    print('Systolic array matches numpy matmul')

    # ########################################################### Sorting weights
    myArray = OSSystolicArray(array_sizey, array_sizex, array_sizek)

#     random.seed(0)
#     activations = random.randint(rnd_from, rnd_to, (array_size, array_size))
#     print('Printing activations')
#     print(activations.tolist())
#     weights = random.randint(rnd_from, rnd_to, (array_size, array_size))
#     print('Printing new sorted weights')
#     print(weights.tolist())
    weights, _ = sortFullMatrix(np.asarray(weights))
#     weights = np.transpose(sortFullMatrix(np.transpose(np.asarray(weights))))
#     print(weights)
    myArray.fill_activations(activations.tolist())
#     myArray.fill_activations(np.transpose(activations))

    # myArray.fill_weights(np.transpose(weights.tolist()))
    myArray.fill_weights(weights.tolist())
    # myArray.fill_weights(weights)

    res_after, total_sw_after = myArray.run()
#     print('total_sw_after is :', total_sw_after)
#     print('res :', res)

    # print(np.transpose(res))
#     print(np.matmul(activations, weights))
    assert (np.transpose(res_after) == np.matmul(activations, weights)).all()
#     assert (res == np.matmul(activations, weights)).all()
    print('Systolic array matches numpy matmul')
    return total_sw_before, total_sw_after

def unfold_in_array(x_in, filter_dim):
    out_dimx = filter_dim * filter_dim # out column dimension
    out_dimy = (x_in.shape[-1] - (filter_dim - 1)) * (x_in.shape[-2] - (filter_dim - 1)) # Out row dimension
    x_out = np.zeros((out_dimy, out_dimx))
    out_row = 0
    for row in range(x_in.shape[-2] - (filter_dim - 1)):      #  along the row dimension
        for col in range(x_in.shape[-1] - (filter_dim - 1)):  #  along the column dimension
            x_out[out_row, :] = x_in[0,0,row:row + filter_dim,col:col + filter_dim].flatten()
#             if row == 14 and col == 15:
#                 print('row 14 col 15 :', x_in[0,0,row:row + filter_dim,col:col + filter_dim].flatten())
#                 print('out_row :', out_row)
            out_row = out_row + 1
#     print(out_row)
    return x_out

def diff_letters(a,b):
    return sum ( a[i] != b[i] for i in range(len(a)) )

    
def sortFullMatrixbinary(list_in): #Eg: 32x200 bits filter sorting
    list_len = len(list_in)
    
    switchingact = np.zeros(list_len)
    original_index=[]
    original_index = np.asanyarray([i for i in range(list_len)])
    sw_mat = []
    for row in range(list_len - 1): # Iterate until total rows - 2 or 3
#         print('starting with row :', row)
        sw_vector = np.zeros(list_len)
        switchingBefore = 200
        sw_row = []
        for ii in range(row+1, list_len): # Iterate until total rows - 2 or 3
            sw_vector[ii] =  diff_letters(list_in[row], list_in[ii])  
            sw_row.append(sw_vector[ii])
#         print('Switching vector', sw_vector)
#         new_order = np.argsort(sw_vector)
        new_row = np.argmin(sw_vector[row + 1:]) + row + 1  # Adding 1 to cancel the offset from removing (row + 1)th index
        sw_mat.append(sw_row)
#         print('matrix before sorting')
#         print(mat1)
        # switch row i+1 and row i+2
        
#         list_in[[row+1, new_row]] = list_in[[new_row, row+1]]
        list_in[row+1], list_in[new_row] = list_in[new_row], list_in[row+1]
        original_index[[row+1, new_row]] = original_index[[new_row, row+1]]
#         print('original_index after being sorted:', original_index)
#         print('matrix after sorting')
#         print(mat1)
    return list_in, original_index, sw_vector, sw_mat # Returning original_index to rearrange activation maps++++

def sortFullMatrixbinary2(list_in): #Eg: 32x200 bits filter sorting
    list_len = len(list_in)
    
    switchingact = np.zeros(list_len)
    original_index = []
    original_index = np.asanyarray([i for i in range(list_len)])
    
    for iteration in range(1, list_len): # Iterate 32 times
        sw_vector = np.zeros(list_len)
        for ii in range(list_len-2): # Iterate until total rows - 2 or 3
            temp0 =  diff_letters(list_in[ii],list_in[ii+1])  
            temp1 =  diff_letters(list_in[ii],list_in[ii+2])  
            sw_vector[ii] = temp0
            # Check if temp0 is greater than temp1, if so swap ii+1 with ii+2
            if(temp0 > temp1):
                list_in[ii+1], list_in[ii+2] = list_in[ii+2], list_in[ii+1]
                original_index[[ii+1, ii+2]] = original_index[[ii+2, ii+1]]
                sw_vector[ii] = temp1
    return list_in, original_index, sw_vector # Returning original_index to rearrange activation maps
'''
#################### Sorting nfilter_reshaped and saving the binary of oringal and sorted
# read txt file rowise
fileNameBits = "./FORSA_SRAM/Filter_original_sram.txt"
f = open(fileNameBits, "r")
bin_line = f.readline()
bits_previous = bin_line
# bits_previous = int(bin_line, base=2)
print(bits_previous)
f.close()

#####2
count = 0
temp0 = 0
list_rows = []
with open(fileNameBits, "r") as openfileobject:
    for bin_line in openfileobject:
        ### Count switching between successive lines
        list_rows.append(bin_line)
        temp0 = diff_letters(bits_previous,bin_line)
        bits_previous = bin_line
        print(temp0)
#         print('switching activities at filter no ',count,'is', temp0)

        count+=1
# print(count)
f.close()
# Use a function to sort the list_rows
list_rows_sorted, sorted_index, sw_vector, sw_mat = sortFullMatrixbinary(list_rows)
print('switching vector of the sorted filter')
print(sw_mat)
# compare switching of sorted with original weight matrix for each row
# firstRow = list_rows_sorted[0]
# count=0
# print('switching activity of the sorted filter')
# for rows in list_rows_sorted:
#     temp0 = diff_letters(firstRow,rows)
#     print(temp0)
#     count=+1
#     firstRow = list_rows_sorted[count]


#save sorted list into a txt file
fileNameBits = "./FORSA_SRAM/Filter_sorted_sram_new.txt"
f = open(fileNameBits, "a")
list_len = len(list_rows_sorted)
for kk in range(list_len):
    f.write(list_rows_sorted[kk])
f.close()


#%% Experiment
# First layer parameters
nfilters = load('nfilters.npy') # filter
print('nfilters shape', nfilters.shape)
x_image = load('x_image.npy') # input (first layer input (image))
# print('x_image', x_image[0, 0, 14:14+5, 15:15+5])
print('x_image shape', x_image.shape)
x_image_unfolded = unfold_in_array(x_image, nfilters.shape[-1])
# print('x_image_unfolded:', x_image_unfolded[14*24+15, :])
print('x_image_unfolded shape :', x_image_unfolded.shape)
x_act = load('x_act.npy') # First layer output (6 activation maps)
print('x_act shape', x_act.shape)

# To be used in MATLAB
import scipy.io
scipy.io.savemat('nfilter_reshaped.mat', {'nfilter_reshaped': nfilter_reshaped})

scipy.io.savemat('nfilters.mat', {'nfilters': nfilters})
scipy.io.savemat('x_image.mat', {'x_image': x_image})
scipy.io.savemat('x_act.mat', {'x_act': x_act})

nfilter_reshaped = nfilters[:, 0, :, :]
print(nfilter_reshaped[0:2, :, :])
# new_arr = arr.reshape(*arr.shape[:2], -1, *arr.shape[-2:])
nfilter_reshaped = nfilters.reshape(*nfilters.shape[:2], -1)[:, 0, :]
# .reshape((6, 25))
print(nfilter_reshaped[0:2, :])
# print('x_image_unfolded shape:', x_image_unfolded.shape)

array_sizey = 576
array_sizex = 32
array_sizek = 25
import scipy.io


myArray = OSSystolicArray(array_sizey, array_sizex, array_sizek)

activations = x_image_unfolded  # random.randint(rnd_from, rnd_to, (array_sizey, array_sizek))
weights = np.transpose(nfilter_reshaped) # random.randint(rnd_from, rnd_to, (array_sizek, array_sizex))

myArray.fill_activations(activations.tolist())
myArray.fill_weights(weights.tolist())

res_before, total_sw_before = myArray.run()
scipy.io.savemat('res_before.mat', {'res_before': res_before})

# print(np.matmul(activations, weights)[100:105, 0:5])
# print(np.transpose(np.asanyarray(res_before))[100:105, 0:5])

assert (np.transpose(np.asanyarray(res_before)) == np.matmul(activations, weights)).all()
print('Systolic array matches numpy matmul')

# ########################################################### Sorting weights
myArray = OSSystolicArray(array_sizey, array_sizex, array_sizek)
weights, new_index = sortFullMatrix(np.asarray(weights))
activations = activations[:, new_index] # Sorting activations to get the original result
myArray.fill_activations(activations.tolist())
myArray.fill_weights(weights.tolist())

res_after, total_sw_after = myArray.run()
# assert (np.transpose(res_after) == np.matmul(activations, weights)).all() # Comparing with builtin function
assert (np.transpose(res_after) == np.transpose(np.asanyarray(res_before))).all() # Comparing with the original output activation map
print('Systolic array matches numpy matmul')

No_exp = 2
# array_size = 4
for ii in range(No_exp):
    total_sw_before, total_sw_after = randMatMult(array_sizey, array_sizex, array_sizek)
    print('total_sw_before :', total_sw_before)
    print('total_sw_after :', total_sw_after)
    
 '''