# Topic Modeler
Creating topic distribution from documents 

# How it Works
The model creats a matrix of counts of the words in the documents using the [CountVectorizer](), the matrix __DWC__. The matric __DWC__ is used to generate the [Pointwise Mutual Information](https://en.wikipedia.org/wiki/Pointwise_mutual_information) values then the values are passed through a [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) to normalize between (0, 1).


# Setup

## Prerequisite
Make sure you have the following installed, set up and ready to go
- Python 3.6+
- Python Virtualenv (virtualenv/pipenv/poetry)
- Nodejs (This is usefull for the jupyter lab extensions and widgets)

1. install all python packages and their dependencies
```
    pip install -r requirements.txt
```

2. set up jupyter extentions
```
    jupyter nbextension enable --py widgetsnbextension
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

3. start the lab
```
    jupyter lab
```
