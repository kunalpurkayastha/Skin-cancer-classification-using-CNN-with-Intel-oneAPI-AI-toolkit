# Skin cancer classification using CNN with Intel oneAPI AI toolkit


## Installation of Anaconda and Intel oneAPI AI toolkits

Download Anaconda to run the files in Intel optimized Python and Intel optimized Tensorflow [here](https://www.anaconda.com/products/distribution) .

After installing Anaconda, open the CMD.exe prompt from the Anaconda and run the following commands

Install Intel optimized python in conda
```bash
conda install -c intel intelpython3_full
```

Create a virtual environment with Intel optimized Python
```bash
conda create -n <your_environment_name> intelpython3_full python=3.8
```

Activate virtual environment
```bash
conda activate <your_environment_name>
```

Add intel channel
```bash
conda config --add channels intel
```

Install Intel optimized Tensorflow
```bash
conda install intel-tensorflow
```
or
```bash
pip install intel-tensorflow
```

In case any duplicacy error occurs, use the below command to avoid it
```bash
set KMP_DUPLICATE_LIB_OK=TRUE
```

## Installation of Python dependencies to be able to train and test the classification model

```bash
conda install matplotlib
```

```bash
pip install keras_preprocessing
```

```bash
pip install Pillow
```

```bash
pip install opencv-python
```

```bash
pip install numpy
```

```bash
pip install scikit-image
```


## Usage

Get into the directory where you saved the dataset and codes
```bash
cd <cloned_Directory_name>
```

Run the file to train and save the model
```bash
python Train.py
```

Run this file to test images using the saved model
```bash
python Test.py
```