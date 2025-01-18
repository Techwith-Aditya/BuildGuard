# AI for Construction Safety 

## Overview
This project leverages **AI-powered computer vision** to ensure safety compliance on construction sites. The model is trained to identify individuals as:
- **Safe** (wearing a helmet and following safety protocols)
- **Unsafe** (not wearing a helmet or violating safety norms)

By automating the detection process, this project aims to enhance workplace safety and reduce accidents on construction sites.

---

## Features
- **Helmet Detection**: Identifies workers wearing or not wearing helmets.
- **Safety Classification**: Classifies individuals as safe or unsafe.
- **Performance Visualization**: Includes confusion matrix, PR curves, and F1-score analysis.
- **Custom Dataset**: Model trained on a custom dataset collected from real-world construction sites.

---

## Technology Stack
- **Framework**: [Google Colab](https://colab.research.google.com/)
- **Model**: YOLOv11 (You Only Look Once)
- **Language**: Python
- **Libraries**: TensorFlow, PyTorch, OpenCV, Matplotlib

---

## How It Works
1. **Data Preparation**: Dataset annotated with classes (`Safe`, `Unsafe`, `Helmet`, `No Helmet`).
2. **Model Training**: YOLOv11 trained on labeled data for safety compliance detection.
3. **Inference**: Model deployed to detect and classify safety violations in real-time.
4. **Evaluation**: Performance measured using F1-scores, precision-recall curves, and visualizations.

---

## Results
- **Accuracy**: High precision in detecting helmet use and safety violations.  
- **PR Curve**: Strong performance on custom datasets.  
- **Visualizations**:  
  - **Confusion Matrix**: Shows classification performance.  
  - **F1 Curve**: Displays F1-score vs confidence thresholds.  
  - **Predictions**: Samples from training and validation.  
  - **Correlogram**: Label correlation in the dataset.

---

## How to Use
1. Clone this repository:  
   ```bash
   git clone https://github.com/Techwith-Aditya/AI_For_Construction_Safety.git
   
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt

3. Prepare the Dataset:
   - Place your dataset in the data folder (or specify its location).
   - Ensure the dataset is labeled with the following classes: Safe, Unsafe, Helmet, No Helmet.

4. Train the Model:
   - Open the training notebook and execute the cells to train the YOLOv11 model:
     ```bash
     python Model_YOLOv11_Training.ipynb
   - Customize the training parameters if needed (e.g., epochs, learning rate).

5. Test the Model:
   - Run inference on test images/videos to check the model's performance:
     ```bash
     python inference.py --image path/to/image.jpg
     python inference.py --video path/to/video.mp4
   - Replace path/to/image.jpg or path/to/video.mp4 with the actual path.

6. Evaluate the Model:
   - Evaluate the model's performance using precision, recall, and F1-score:
     ```bash
     python evaluate.py

7. Visualize Results:
   - Navigate to the Images Folder to see training and validation predictions.
   - View plots of precision-recall curves and other performance metrics.


