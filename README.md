# Iris image processing to apply Machine Learning algorithms

The purpose of this project is to provide an alternative method to validate the theory of iridology, an alternative medicine techinque that claims to determine information about the health of a person just by analizing the iris.
The proposed method consists in extracting segments of iris from eye images, making predictions using a CNN model and then compare the results against the iridology's one.

The CNN model is trained on segments of iris divided in two categories: segments with known problems related to that particular region of the iris (which is related to a particular region of the body according to the iridology's chart) and segments without problems.

If the model is accurate enough we can compare the predictions with the results of an iridology analisys on the same segments. If most of the results differ then we can say that the iridology (for this particular region of the eye) is a false theory.
If the entire process is repeated for different parts of the iris than we can say that the iridology is a fake theory.

The project provides a set of scripts to extract the iris segments from the eye images, train the machine learning model and make predictions. The databased used for the tests is the CASIA db and the file [config.ini](config.ini) is used to store all the important parameters.

## Installation

Clone the repository.

```bash
git clone https://github.com/nicoloalbergoni/iridology-CNN
```

Inside the project directory create a Virtual Environment.

```bash
cd iridology-CNN\
virtualenv venv
```

Then you just need to activate the environment and install all the required packages listed in [requirements.txt](requirements.txt).

#### Linux

```bash
source /venv/bin/activate
pip install -r requirements.txt
```

#### Windows

```bash
cd /venv/Scripts/
activate
pip install -r requirements.txt
```

## Usage

Place the eye images related to problems inside the **DB_PROBS** subfolder contained in the **DATA_IMAGE** folder wheres the normal labeled images in the **DB_NORMAL** folder.

Then to start the preprocess task simply run the [preprocess.py](preprocess.py) script.

```bash
python preprocess.py
```

This script performs several operations on the images like: filtering, iris recognition algorithms, segmentation, cropping and resizing, then it saves the extracted segments in a folder called **TEMP_SEG**.
Those segments are rady to be used for the next phase which is training.

To train a model execute the [train.py](train.py) script.

```bash
python train.py
```

The scripts outputs the related `.model` file and saves it the **MODELS** folder.

To make predictions copy the desired `.model` file into the **PREDICTIONS** directory and add the images on which you want to make predictions to the **DATA_TO_PREDICT** subfolder.
Then run the [predict.py](predict.py) script.

```bash
python predict.py
```

The script outputs to the console the predictions results.

## Note about the configuration file

The [config.ini](config.ini) file contains all the required parameters for the project to work; however the parameters are set to be working right with the selected set of images (CASIA DB). 

If you wish you can change the images to work with but you will probably have to also change the value of the parameters in the configuration file to obtain good results. If you don't find the right set of parameters for your images than some images may be discarder due to bad recognition of the iris.
