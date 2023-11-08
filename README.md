# FORSAv2: Sorting filters of a pre-traned DNN model for reducing switching activity
Filter reordering technique for reducing dynamic power consumption of a pre-trained model inference List of notebooks to reproduce the results presented in our paper
# List of scripts and notebooks used in FORSAv2 paper
1. Sorting a binary matrix based on bit flips between successive rows of a matrix [Interconnect post alyout simulation](https://github.com/KM1621/FORSAv2/blob/main/WireModels/Sort_list_wiremodel.ipynb)
2. Filter sorting in a convolution layer and Fully connected layer [Filter sorting CNN](https://github.com/KM1621/FORSAv2/blob/main/main_LeNet_FORSA.py)
3. Batch script to reproduce the Switching activity of all layers before and after sorting for different bit-width [Switching activity of all layers](https://github.com/KM1621/FORSAv2/blob/main/FORSA_batch.bat)
4. Heatmap for switching activity of sorted and re-arranged matrices A and B from Equation 2 [Generate heatmap](https://github.com/KM1621/FORSAv2/blob/main/mat_mult_heatmap.py) 
5. Switching activity reduction for popular DNN models [Switching activity reduction for different DNN models](https://github.com/KM1621/FORSAv2/tree/main/PyTorch-Models)
   - VGG
   - AlexNet
   - GoogleNet
   - SqueezeNet
   - MobileNet
   - Resnet
   - LeNet
6. Effect of matrix dimension on switching activity reduction [Generate heatmap](https://github.com/KM1621/FORSAv2/blob/main/mat_mult_heatmap.py) 
