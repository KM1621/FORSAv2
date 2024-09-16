# FORSAv2: Sorting filters of a pre-traned DNN model for reducing switching activity

Filter reordering technique for reducing dynamic power consumption of a pre-trained model inference List of notebooks to reproduce the results presented in our paper.
![Filter sorting in a convolution layer and using the index obtained from sorting to re-order filters of the a succeeding
layer along its channel](https://github.com/KM1621/FORSAv2/blob/main/Figures/Fig4.gif)

![Filter sorting in a fully connected layer and using the index obtained from sorting to re-order filters of the a succeeding
layer](https://github.com/KM1621/FORSAv2/blob/main/Figures/Fig5.gif)

![Demonstrating how sorting and reordering preserves output feature map matrix](https://github.com/KM1621/FORSAv2/blob/main/Figures/Fig3.gif)
# List of scripts and notebooks used in FORSAv2 paper
1. Sorting a binary matrix based on bit flips between successive rows of a matrix. [Interconnect post alyout simulation](https://github.com/KM1621/FORSAv2/blob/main/WireModels/Sort_list_wiremodel.ipynb)
   ![Bit flips between successive rows of a Matrix before
and after sorting](https://github.com/KM1621/FORSAv2/blob/main/Figures/Fig2.gif)
3. Filter sorting in a convolution layer and Fully connected layer. [Filter sorting CNN](https://github.com/KM1621/FORSAv2/blob/main/main_LeNet_FORSA.py)
4. Batch script to reproduce the Switching activity of all layers before and after sorting for different bit-width. [Switching activity of all layers](https://github.com/KM1621/FORSAv2/blob/main/FORSA_batch.bat)
5. Heatmap for switching activity of sorted and re-arranged matrices A and B from Equation 2. [Generate heatmap](https://github.com/KM1621/FORSAv2/blob/main/mat_mult_heatmap.py) 
6. Switching activity reduction for popular DNN models. [Switching activity reduction for different DNN models](https://github.com/KM1621/FORSAv2/tree/main/PyTorch-Models)
   - VGG
   - AlexNet
   - GoogleNet
   - SqueezeNet
   - MobileNet
   - Resnet
   - LeNet
7. Effect of matrix dimension on switching activity reduction. [Matrix dimension](https://github.com/KM1621/FORSAv2/blob/main/sw_binary_mat.ipynb) 
