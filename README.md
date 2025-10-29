# U-Net Segmentation Viewer for Tumour-Stroma Spheroids

## Project Summary

This project focuses on cell type classification (tumor, fibroblast, endothelial) within Tumour-Stroma Spheroids using a U-Net model. The project utilizes a U-Net architecture with a pre-trained ResNet34 encoder. The use of a pre-trained encoder (like ResNet34, typically trained on a large dataset like ImageNet) is a common and effective strategy in deep learning, especially when working with limited datasets. This transfer learning approach helps in:

-   **Faster Convergence:** The model starts with good initial weights, reducing training time.
-   **Improved Performance:** Pre-trained features are often robust and generalize well, leading to better segmentation accuracy, especially with smaller datasets.
-   **Reduced Overfitting:** By using learned features, the model is less likely to overfit to the specific training data.

It includes data preparation, model training, prediction visualization, and a web application to interactively view the results.

## Key Design Choices

Throughout this project, several design and technical choices were made to optimize for efficiency, performance, and maintainability:

-   **Maximum Intensity Projection (MIP) Data:** The project utilizes Maximum Intensity Projection (MIP) images as input. This technique is crucial for 3D biological imaging data, as it effectively reduces the dimensionality of volumetric data into a 2D representation while preserving the brightest signals, which often correspond to cellular structures.
-   **Cellpose for Ground Truth Segmentation:** Cellpose, a deep learning-based segmentation algorithm, was chosen for generating the initial ground truth masks. Its robust performance in segmenting various cell types and structures provides a strong foundation for the U-Net model's training.
-   **FastAPI for Backend Development:** FastAPI was selected for building the web application's backend due to its high performance (comparable to Node.js and Go), modern Python features (type hints), and excellent developer experience.
-   **Tailwind CSS for Frontend Styling:** For the frontend, Tailwind CSS was employed. This utility-first CSS framework enables rapid UI development and highly customizable designs, ensuring a clean and responsive user interface.

## Folder Structure

-   `data/`: Contains raw and processed image data.
    -   `MIP/`: Raw Maximum Intensity Projection (MIP) images.
    -   `processed/images/`: Grayscale processed images used as input for the U-Net model.
    -   `Segmentations/channels_cellpose/`: Original Cellpose-generated masks (ground truth).
    -   `Segmentations/8bit/`: 8-bit versions of Cellpose-generated masks, suitable for web display.
-   `step3_unet_model/`: Contains the U-Net model definition, training script, and prediction visualization tools.

### Training and Validation

The U-Net model is trained using the `train.py` script. The training process involves:

-   **Dataset Preparation:** The `SpheroidDataset` (defined in `dataset.py`) loads processed images from `data/processed/images` and corresponding ground truth masks from `data/Segmentations/channels_cellpose`. The dataset is split into three parts: a **training set** (70%) for model learning, a **validation set** (15%) for monitoring performance and hyperparameter tuning during training, and a dedicated **test set** (15%) for final, unbiased evaluation of the model's generalization capabilities on unseen data. Data augmentation techniques (defined in `train.py`) are applied to the training set to increase the diversity of the training data and improve model generalization.
    *Note: It has been observed that the dataset may contain a limited number of fibroblast cells, which could impact the model's ability to accurately segment this cell type. To mitigate this, a pre-trained UNet model with a ResNet34 encoder is utilized, leveraging transfer learning to improve performance and generalization even with imbalanced or limited data.*
-   **Model Architecture:** The model utilizes a `UNetWithResnet34Encoder` (defined in `pretrained_model.py`), which is a U-Net architecture leveraging a pre-trained ResNet34 as its encoder for robust feature extraction.
-   **Loss Function and Optimizer:** (Assuming standard practices) The model is typically trained using a combination of a pixel-wise binary cross-entropy loss and a Dice loss, optimized using Adam or a similar optimizer.
-   **Validation:** During training, the model's performance on the validation set is monitored to track progress, detect overfitting, and determine the best model checkpoints.
-   **Testing:** After training is complete, the model is evaluated on the independent test set to provide an unbiased assessment of its performance on completely unseen data.
-   **Metrics:** Key metrics such as accuracy, Dice coefficient, and loss are tracked and visualized (e.g., `loss_plot.png`, `dice_plot.png`) to assess the model's learning progress and segmentation quality.

    -   `dataset.py`: Defines the dataset loading for the U-Net.
    -   `model.py`: Defines the U-Net model architecture.
    -   `pretrained_model.py`: Likely contains the U-Net model with a pre-trained encoder.
    -   `train.py`: Script for training the U-Net model.
    -   `unet_spheroid.pth`: The trained U-Net model weights.
-   `webapp/`: Contains the FastAPI web application for visualizing results.
    -   `app.py`: The FastAPI backend application.
    -   `static/`: Static files for the web app (e.g., CSS, JS, if any).
    -   `templates/`: HTML templates for the web app.
        -   `index.html`: The main single-page application interface.

-   `visualize_predictions.py`: Script to generate visualizations of U-Net predictions.
-   `convert_masks_to_8bit.py`: Utility script to convert existing masks to 8-bit PNG format.
-   `view_ground_truth.py`: Utility script to view ground truth masks (saves to a file).
-   `README.md`: This file.

## Scripts Explanation and Outputs

### Example Output Images in Root Directory

Several `.png` files are present directly in the project's root directory (`FinalProject/`). These are example output images generated during various stages of development and data exploration:

-   `A_96h_Image001_comparison.png`, `B_48h_Image001_comparison.png`, `B_48h_Image005_comparison.png`, `B_96h_Image001_comparison.png`, `C_96h_Image001_comparison.png`: These images are examples demonstrating the successful conversion of original merged images to grayscale, often showing a side-by-side comparison.
-   `accuracy_plot.png`, `class_accuracy_plot_endothelial.png`, `class_accuracy_plot_fibroblasts.png`, `class_accuracy_plot_tumor.png`, `dice_plot.png`, `class_dice_plot_endothelial.png`, `class_dice_plot_fibroblasts.png`, `class_dice_plot_tumor.png`, `loss_plot.png`: These are plots generated during model training and evaluation, showcasing metrics like accuracy, Dice score, and loss over epochs.
-   `data_visualization_0.png` to `data_visualization_4.png`: These images were likely produced during initial data exploration or visualization steps to understand the dataset.
-   `plot.png`: A general plot, likely generated during data exploration or initial testing.
-   `prediction_visualization.png`: An older or single example of a prediction visualization.
-   `ground_truth_masks.png`: The output from running the `view_ground_truth.py` script, showing an example of the three ground truth masks.

### `CellposeSegmentation.ipynb`

This Jupyter Notebook serves as a comprehensive guide and execution environment for Cellpose-based segmentation. It covers the following key aspects:

-   **Cellpose Testing:** This section likely contains initial tests and explorations of Cellpose parameters on a sample image to find optimal settings.
-   **Actual Segmentation:** This is the core part of the notebook where Cellpose is applied to a larger set of images (or all images) to generate the segmentation masks. It includes preprocessing steps and the application of the Cellpose model.
-   **Mask Verification:** This section is dedicated to visually verifying the quality of the generated Cellpose masks by displaying them alongside their original images. This helps in assessing the accuracy and effectiveness of the segmentation.

### `convert_masks_to_8bit.py`

This utility script takes existing PNG mask files (e.g., from `data/Segmentations/channels_cellpose`) and converts them to an 8-bit PNG format, scaling pixel values to the 0-255 range. This is crucial for proper display in web browsers.

**Outputs:**
-   `data/Segmentations/8bit/`: 8-bit versions of the Cellpose masks (e.g., `A_24h_Image001_ch002_mask.png`).

### `step3_unet_model/visualize_predictions.py`

This script loads the trained U-Net model (`unet_spheroid.pth`), performs predictions on a subset of validation images, and generates visualization plots. It displays the original image and the three predicted masks (Endothelial, Tumor, Fibroblasts).

**Outputs:**
-   `step3_unet_model/prediction_visualization_X.png`: PNG images showing the original image and the predicted masks for `X` examples.

### `view_ground_truth.py`

A simple utility script to load and display the ground truth masks for a specified image ID. It saves the combined plot to a file.

**Outputs:**
-   `ground_truth_masks.png`: A PNG image showing the three ground truth masks for the specified example image.

### `webapp/main.py` (FastAPI Backend)

This is the backend for the web application. It provides API endpoints to:
-   Serve static files (images, CSS, JS).
-   Handle image uploads from the user.
-   Perform U-Net model inference on the uploaded image to generate predicted masks.
-   Load corresponding ground truth masks (if available).
-   Calculate Dice scores for each cell type.
-   Generate a visualization combining the original image, predicted masks, and ground truth masks.
-   Return the visualization, Dice scores, and the training loss plot to the frontend.

### `webapp/templates/index.html` (Frontend)

This is the single-page web application interface. It uses HTML, Tailwind CSS, and JavaScript to:
-   Provide a form for users to upload an 8-bit image.
-   Display a loading indicator during processing.
-   Show the prediction results, including the original image, predicted masks, ground truth masks, and calculated Dice scores.
-   Display the training loss plot (`loss_plot.png`).

## How to Run the Web Application

1.  **Start the FastAPI application:**
    Navigate to the project's root directory and execute the provided script:
    ```bash
    ./run_uvicorn_and_log.sh
    ```
    This command will start the FastAPI server using Uvicorn. The `--reload` flag means the server will automatically restart if code changes are detected. The web application dynamically performs model inference and visualization upon image upload, so there's no need to pre-run separate scripts for predictions or 8-bit mask conversion.

2.  **Access the application:** Open your web browser and navigate to `http://localhost:8000` (or the appropriate public URL if using a tunneling service like `ngrok`).

3.  **Image Upload and Prediction:** Upload an 8-bit image through the web interface. The application will process it, run the U-Net model, and display the original image, predicted masks, ground truth masks (if available), Dice scores, and the training loss plot.

    *Note: The prediction threshold for binarizing masks is currently set to `0.05` to include more cells, even those with lower probability.*
