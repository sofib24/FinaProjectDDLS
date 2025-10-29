Project Plan — Cell Type Classification in Tumour–Stroma Spheroids (MIP-based, Pre-trained CNNs)


Summary
 This project builds an automated cell-type classifier (tumor, fibroblast, endothelial) using control samples from the Tumour–Stroma Spheroid 3D Dataset (Piccinini & Diosdi, 2024). The model input is a grayscale spheroid image and the model should predict which parts of the spheroid are composed of which cell types - tumor cells, fibroblasts or endothelial cells. 

Step 1
First, we use Cellpose to segment each channel and save the masks in data/Segmentations/channels_cellpose. 

Step 2
Next, we create grayscale image of the original merged images. These can be found in data/processed/images. This serves as the input into the classification U-net. 

Step 3 
Finally, train a U net. The input images are in data/processed/images. From the innput image, I want to determine what part of the spheroid is composed of endothelial cells, what part is tumors cell and what part is fibroblasts. The labeled images to use for training and defining the ground truth are in data/Segmentations/channels_cellpose. For each imagem there is a separate mask where files named ch002 are endothelial cells, ch003 are tumor cells and ch004 are fibroblasts. Create visualizations of the accuracy, loss function and an example of the predicted areas versus the ground truth. 

Step4
The full U-Net model file can be found at folder step3_unet_model, the framework is PyTorch model, and the input images are found in data/processed/images ant they are 32-bit. Now I want to create an app where I upload an image that is 8-bit, then the image is passed to the model and returns a png where you see the original image and the predicted masks as in visualize_predictions.py. Furthermore show the ground truth masks (8bit, in data/Segmentations/8bit) and the dice plot. Use fastapi for the backend server. And for the frontend, I want to create a single page web app using CSS framework tailwind CSS.

