# Lung-Cancer-Segmentation-CBCT

# Localization Module

process(): Implements localization algorithm 1

process2(): Implements localization algorithm 2

process_all(): Call process() or process2() for all scans

# Segmentation Module

main.py: For feature extraction and classifier training purposes

deploy_main.py: Classification of test images using a saved model

# False-Positive Suppression

hierarchical_clustering.py: Implements hierarchical clustering on segmentation output map

# Illustrations of the results

<img src="https://raw.githubusercontent.com/brcsomnath/Lung-Cancer-Segmentation-CBCT/master/Results/output_groundtruth.png" width="400"> <img src="https://raw.githubusercontent.com/brcsomnath/Lung-Cancer-Segmentation-CBCT/master/Results/output_prediction.png" width="400"> <br>

<img src="https://raw.githubusercontent.com/brcsomnath/Lung-Cancer-Segmentation-CBCT/master/Results/output_groundtruth0.png" width="400"> <img src="https://raw.githubusercontent.com/brcsomnath/Lung-Cancer-Segmentation-CBCT/master/Results/output_prediction0.png" width="400"> <br>

<img src="https://raw.githubusercontent.com/brcsomnath/Lung-Cancer-Segmentation-CBCT/master/Results/output_groundtruth1.png" width="400"> <img src="https://raw.githubusercontent.com/brcsomnath/Lung-Cancer-Segmentation-CBCT/master/Results/output_prediction1.png" width="400"> <br>

<img src="https://raw.githubusercontent.com/brcsomnath/Lung-Cancer-Segmentation-CBCT/master/Results/output_groundtruth3.png" width="400"> <img src="https://raw.githubusercontent.com/brcsomnath/Lung-Cancer-Segmentation-CBCT/master/Results/output_prediction3.png" width="400"> <br>
