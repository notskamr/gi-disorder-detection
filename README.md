# Implementation of a Lightweight Convolutional Neural Network for the Detection of Gastrointestinal Disorders

This repository contains the code and utilities for detecting gastrointestinal (GI) disorders using various Convolutional Neural Network (CNN) architectures. The project focuses on classifying colonoscopy images into three categories: Normal, Polyps, and Ulcerative Colitis.

## Repository Contents

- `models.ipynb`: The main Jupyter notebook containing the implementation of various CNN models for GI disorder detection.
- `lazy-predict.ipynb`: A Jupyter notebook using the LazyPredict library for quick model comparison and analysis.
- `graphing.py`: A Python module with utility functions for creating graphs and visualizations.
- `utils.py`: A Python module with quality-of-life utility functions for data processing and model training.

## Setup and Requirements

1. Clone this repository:
   ```
   git clone https://github.com/notskamr/gi-disorder-detection.git
   cd gi-disorder-detection
   ```

2. Ensure you have the following main dependencies:
   - TensorFlow
   - NumPy
   - Pandas
   - Matplotlib
   - Scikit-learn
   - LazyPredict

## Dataset

The project uses a pooled variant of the [HyperKvasir dataset](https://datasets.simula.no/hyper-kvasir/). This pooled version can be found on [OSF](https://osf.io/7maz5/). To use your own dataset:

1. Organize your images in the following structure:
   ```
   data_dir/
   ├── normal/
   ├── polyps/
   └── ulcerative-colitis/
   ```

2. Update the `DATA_DIR` variable in `models.ipynb` and `lazy-predict.ipynb` to point to your dataset location.

## Usage

1. Start by running the `lazy-predict.ipynb` notebook to get a quick comparison of various machine learning models:
   ```
   jupyter notebook lazy-predict.ipynb
   ```

2. For the main analysis and CNN model training, open and run `models.ipynb`:
   ```
   jupyter notebook models.ipynb
   ```

3. The `models.ipynb` notebook will:
   - Load and preprocess the dataset
   - Define and train various CNN architectures (VGG16, EfficientNet variants, MobileNetV3)
   - Evaluate model performance
   - Generate visualizations and confusion matrices

4. Results, including model checkpoints and evaluation figures, will be saved in the `./models/{MAX_EPOCHS}e/` directory, where `{MAX_EPOCHS}` is the maximum number of epochs specified in the notebook (default is 50).

## Customization

- To add new models or modify existing ones, define the models and add them to the `models` list of dictionaries in `models.ipynb`.
- Adjust hyperparameters such as `BATCH_SIZE`, `IMG_SIZE`, and `MAX_EPOCHS` in `models.ipynb` to experiment with different training configurations.
- Modify the `graphing.py` and `utils.py` files to add or change utility functions as needed.

## Acknowledgments

- The HyperKvasir dataset for providing the initial data for this research.
- The TensorFlow team for their excellent deep learning framework.
- The developers of LazyPredict for their useful rapid model prototyping tool.

## Contact

For any questions or feedback, please open an issue in this repository or contact me at [contact@vsahni.me].