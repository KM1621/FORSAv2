# Definition of functions used for sorting and calculating switching activities
import torch
import numpy as np
word_size = 8
frac_size = 6
from fxpmath import Fxp


## Modifying the sorting function
# First convert the matrices to binary
def mat_2_bin(mat1):
    height = np.size(mat1, 0)
    width = np.size(mat1, 1)
    mat_bin = []
    for row in range(height): # Iterate until total rows - 2 or 3
#         print('working on row no :', row)
        row_vec = []
        for col in range(width):
            cur_element = Fxp(mat1[row][col], True, word_size, frac_size)
            row_vec.append(cur_element.bin())
        mat_bin.append(row_vec)
    return mat_bin

# print(b1_conv_wt_bin)

def sortFullBinMatrix_V2(mat_in):
    mat1 = mat_in
    height = np.size(mat1, 0)
    width = np.size(mat1, 1)
    original_index=[]
    original_index = np.asanyarray([i for i in range(height)])
    for row in range(height - 2): # Iterate until total rows - 2 or 3
#         print('working on row no :', row)
        sw_vector = np.zeros(height)
        for ii in range(row+1, height): #
            temp0=0
            for col in range(width):
                temp0 = temp0 + (mat1[row][col] ^ mat1[ii ][col]).bin().count('1')
            sw_vector[ii] =  temp0
#             print(sw_vector)
#         print(sw_vector)
        new_row = np.argmin(sw_vector[row + 1:]) + row + 1  # Adding 1 to cancel the offset from removing (row + 1)th index
        mat1[row+1], mat1[new_row] = mat1[new_row], mat1[row+1]
        original_index[[row+1, new_row]] = original_index[[new_row, row+1]]
    return mat1, original_index # Returning original_index to rearrange activation maps

# New sorting function (Faster)
######## Faster method to calcuate switching activity
def association(I,II):
    B = ((I.reshape(-1,1) & (2**np.arange(word_size))) != 0).astype(int)
    A = ((II.reshape(-1,1) & (2**np.arange(word_size))) != 0).astype(int)
    return np.logical_xor(A[:,::-1],B[:,::-1]).astype(np.int8) # for the binary values, the binding or assossiation is xor operation.

def calc_sw_act_V2(mat1):
    height = np.size(mat1, 0)
    sw_vector = np.zeros(height-1)
    for row in range(height - 1):
        sw_vec = association(mat1[row], mat1[row+1])
        sw_vector[row] =  np.sum(sw_vec)
    return sw_vector 

def calc_sw_act(mat1):
    height = np.size(mat1, 0)
    width = np.size(mat1, 1)
    sw_vector = np.zeros(height-1)
    for row in range(height - 1): # Iterate until total rows - 1
            temp0=0
            for col in range(width):
                row0 = Fxp(mat1[row][col], True, word_size, frac_size)
                row1 = Fxp(mat1[row + 1 ][col], True, word_size, frac_size)
                temp0 = temp0 + (row0 ^ row1).bin().count('1')
            sw_vector[row] =  temp0
    return sw_vector # Returning original_index to rearrange activation maps

######################## Comparing both functions
def sortFullMatrix_V2(mat_in): #Using full row as one vector
    dim_shape = np.count_nonzero(mat_in.shape)-1
    mat1 =   torch.flatten(mat_in.data.cpu(), -1*dim_shape).numpy()
    height = np.size(mat1, 0)
    mat1 = (mat1 * (2**frac_size)).astype(int)
    original_index=[]
    original_index = np.asanyarray([i for i in range(height)])
    for row in range(height - 2): # Iterate until total rows - 2 or 3
#         print('working on row no :', row)
        sw_vector = np.zeros(height)
        for ii in range(row+1, height): # mat1[row][col] ^ mat1[ii ][col]
            sw_vector[ii] =  np.sum(association(mat1[row][:], mat1[ii][:]))
        new_row = np.argmin(sw_vector[row + 1:]) + row + 1  # Adding 1 to cancel the offset from removing (row + 1)th index
        mat1[[row+1, new_row]] = mat1[[new_row, row+1]]
        original_index[[row+1, new_row]] = original_index[[new_row, row+1]]
    return mat1, original_index # Returning original_index to rearrange activation maps

######################## Comparing sorted and original matrix
def compare_sw_sort(mat_1, original_index, flat_start, isconv=True, printsumm=False): # Using full row as one vector
    if isconv==True:
        b1_conv_wt_mat = torch.flatten(mat_1, flat_start).cpu().detach().numpy()
    else:
        b1_conv_wt_mat = mat_1.cpu().detach().numpy()
    fixed_point_matrix = (b1_conv_wt_mat * (2**frac_size)).astype(int)
    fixed_point_matrix_original = fixed_point_matrix.copy()
    fixed_point_matrix_sorted = fixed_point_matrix[original_index]#[:] #########sortFullMatrix_V2##############
    sw_vector = calc_sw_act_V2(fixed_point_matrix_original)
    ################### after sorting
    sw_vector_sorted = calc_sw_act_V2(fixed_point_matrix_sorted)
    if printsumm:
        print('Switching before sorting', np.sum(sw_vector))
        print('Switching after sorting', np.sum(sw_vector_sorted))
        print('Percentage of switching reduction', (np.sum(sw_vector) - np.sum(sw_vector_sorted))*100/np.sum(sw_vector), '%')
    sw_redc_rate = (np.sum(sw_vector) - np.sum(sw_vector_sorted))*100/np.sum(sw_vector)
    return mat_1[original_index], sw_redc_rate, np.sum(sw_vector), np.sum(sw_vector_sorted) # Returning original_index to rearrange activation maps



import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_dist(filter_mat):
    dimension = 0
    dim_shape = np.count_nonzero(filter_mat.shape)
    print(dim_shape)
    data_along_dimension = torch.flatten(filter_mat.cpu().detach(), -1*dim_shape) #data[:, dimension]#.numpy()
    kde = gaussian_kde(data_along_dimension.numpy())
    x = np.linspace(data_along_dimension.numpy().min(), data_along_dimension.numpy().max(), np.sum(filter_mat.shape))
    y = kde(x)
    plt.plot(x, y, label='Distribution')
    plt.fill_between(x, y, alpha=0.2)  # Fill the area under the curve
    plt.title(f'Distribution Along Dimension {dimension}')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
