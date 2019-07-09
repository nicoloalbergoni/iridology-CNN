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

### Linux

```bash
source /venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bash
\venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
