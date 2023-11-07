''' This is to study the effect of sorting and re-arranging matrices in percentage of switching activity reduction. It generates heatmap to visualize the percentage of reduction for different matrix dimensions '''
import numpy as np
from fxpmath import Fxp
import matplotlib.pyplot as plt
import torch
import argparse

mat_W_list   = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
mat_H_A_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024] # A is west
mat_H_B = 4 # B is north
word_size  = 32
frac_size = 0
# 
xlabs = [i for i in range(len(mat_W_list))]
ylabs = [i for i in range(len(mat_H_A_list))]
parser = argparse.ArgumentParser(description='Matrix multiplication switching activity stats with and without sorting')
parser.add_argument('--n_exp', default=1, type=int, metavar='N', help='number of experiments for a given experiment')

##### add more arguments here
args = parser.parse_args()


def main():

    global args, best_prec
    
    number_expPsize = args.n_exp
    sw_heatmap = np.zeros((len(mat_W_list), len(mat_H_A_list)))
    sw_heatmap_A = np.zeros((len(mat_W_list), len(mat_H_A_list)))
    for idx_B, mat_W in enumerate(mat_W_list):
        print('idx_B', idx_B)
        for idx_A, mat_H_A in enumerate(mat_H_A_list):
#             print('idx_A', idx_A)
            avg = np.zeros(number_expPsize)
            avg_A = np.zeros(number_expPsize)
            for i in range(number_expPsize):
                A = np.random.randint(low=0, high=10000, size=(mat_H_A, mat_W), dtype=np.int64)
                B = np.random.randint(low=0, high=10000, size=(mat_W, mat_H_B), dtype=np.int64)
                B_original = B.copy()
                A_original = A.copy()
#                 B_bin = mat_2_bin(B)
                B_sorted, original_index = sortFullMatrix_V2(B)
#                 B_sorted = B[original_index, :]
                A_rearranged = A[:, original_index]
                assert (np.dot(A_rearranged, B_sorted) == np.dot(A_original,B_original)).all()
                sw_vector = calc_sw_act_V2(B_original)
                sw_vector_sorted = calc_sw_act_V2(B_sorted)
                sw_act = np.sum(sw_vector)
                sw_act_sorted = np.sum(sw_vector_sorted)
                avg[i] = (sw_act - sw_act_sorted)/sw_act
#                 print("Switching activity after rearranging mat 1", avg[i])
                sw_vector = calc_sw_act_V2(A_original.transpose())
                sw_act = np.sum(sw_vector)
                sw_vector_rearranged = calc_sw_act_V2(A_rearranged.transpose())
                sw_act_sorted = np.sum(sw_vector_rearranged)
#                 sw_red_rate_A = (sw_act - sw_act_sorted)/sw_act
                avg_A[i] = (sw_act - sw_act_sorted)/sw_act
#                 print("Switching activity after rearranging mat 2", sw_red_rate_A)
            sw_heatmap[idx_B, idx_A] = np.mean(avg)
            sw_heatmap_A[idx_B, idx_A] = np.mean(avg_A)
    # Assuming you have multiple NumPy arrays for each heatmap plot
    sw_heatmap1 = sw_heatmap
    sw_heatmap2 = sw_heatmap_A
    print(sw_heatmap)
    vmin = np.max(sw_heatmap)
    vmax = np.min(sw_heatmap)
    # Example custom tick labels for both x-axis and y-axis
    xlabs = mat_W_list
    ylabs = mat_H_A_list

    # Create the subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Adjust the number of subplots as needed
#     cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [x, y, width, height] position
    titles = ['Switching activity of sorted matrix', 'Switching activity of rearranged matrix']
    # Iterate over each subplot and set custom tick labels, labels, and titles
    for i, sw_heatmap in enumerate([sw_heatmap1, sw_heatmap2]):
        ax = axes[i]
        im = ax.imshow(sw_heatmap, cmap='coolwarm', interpolation='nearest', vmin=vmin, vmax=vmax)

        # Set custom tick labels for the X and Y axes
        ax.set_xticks(np.arange(len(xlabs)))
        ax.set_yticks(np.arange(len(ylabs)))
        ax.set_xticklabels(xlabs, fontsize=14)
        ax.set_yticklabels(ylabs, fontsize=14)

        # Add a colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('', fontsize=18)
        # Modify the x-axis and y-axis labels with your desired text for each subplot
#         ax.set_xlabel(f"X-axis Label {i+1}")
#         ax.set_ylabel(f"Y-axis Label {i+1}")
        ax.set_xlabel("Matrix A height", fontsize=18)
        ax.set_ylabel("Matrix B width", fontsize=18)

        # Set a title for each subplot
#         ax.set_title(f"Heatmap {i+1}")
        ax.set_title(titles[i], fontsize=18)
    # Adjust the layout and spacing between subplots
    plt.tight_layout()

#     cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height] position
#     plt.colorbar(im, cax=cbar_ax)

    # Save the figure as an image
    plt.savefig("multiple_heatmaps.png")

    # Show the figure with multiple subplots
#     plt.show()

#     plt.imshow(sw_heatmap, cmap='coolwarm', interpolation='nearest')
#     print(sw_heatmap.dtype)
# #     ax.set_xticks(np.arange(len(xlabs)), labels = xlabs)
# #     ax.set_yticks(np.arange(len(ylabs)), labels = ylabs)
#     plt.colorbar()  # Add a colorbar
#     plt.title("Matrix Heatmap")
#     plt.xlabel("X-axis labels")
#     plt.ylabel("Y-axis labels")
#     plt.savefig("heatmap_"+str(mat_H_B)+".png")
#     plt.show()

                
## Modifying the sorting function
# New sorting function (Faster)
######## Faster method to calcuate switching activity
def mat_2_bin(mat1):
    height = np.size(mat1, 0)
    width = np.size(mat1, 1)
    mat_bin = []
    for row in range(height): # Iterate until total rows - 2 or 3
#         print('working on row no :', row)
        row_vec = []
        for col in range(width):
            cur_element = Fxp(mat1[row][col], True, word_size, frac_size)
            row_vec.append(cur_element)
        mat_bin.append(row_vec)
    return mat_bin

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
    return sw_vector # Returning original_index to rearrange activation maps

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
    mat1 = mat_in
    height = np.size(mat1, 0)
    original_index=[]
    original_index = np.asanyarray([i for i in range(height)])
    for row in range(height - 2): # Iterate until total rows - 2 or 3
#         print('working on row no :', row)
        sw_vector = np.zeros(height)
        for ii in range(row+1, height): # mat1[row][col] ^ mat1[ii ][col]
            sw_vector[ii] =  np.sum(association(mat1[row][:], mat1[ii][:]))
#         print(sw_vector)
        new_row = np.argmin(sw_vector[row + 1:]) + row + 1  # Adding 1 to cancel the offset from removing (row + 1)th index
        mat1[[row+1, new_row]] = mat1[[new_row, row+1]]
        original_index[[row+1, new_row]] = original_index[[new_row, row+1]]
    return mat1, original_index # Returning original_index to rearrange activation maps

######################## Comparing sorted and original matrix
def compare_sw(mat_1, original_index, flat_start): #Using full row as one vector
    b1_conv_wt_mat = torch.flatten(mat_1, flat_start).detach().numpy()
    fixed_point_matrix = (b1_conv_wt_mat * (2**frac_size)).astype(int)
    fixed_point_matrix_original = fixed_point_matrix.copy()
    fixed_point_matrix_sorted = fixed_point_matrix[original_index][:] #########sortFullMatrix_V2##############
    sw_vector = calc_sw_act_V2(fixed_point_matrix_original)
    ################### after sorting
    sw_vector_sorted = calc_sw_act_V2(fixed_point_matrix_sorted)
    print('Switching before sorting', np.sum(sw_vector))
    print('Switching after sorting', np.sum(sw_vector_sorted))
    print('Percentage of reduction', (np.sum(sw_vector) - np.sum(sw_vector_sorted))*100/np.sum(sw_vector), '%')
    sw_redc_rate = (np.sum(sw_vector) - np.sum(sw_vector_sorted))*100/np.sum(sw_vector)
    return mat_1[original_index], sw_redc_rate # Returning original_index to rearrange activation maps
if __name__=='__main__':
    main()