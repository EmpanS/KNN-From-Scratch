# KNN-From-Scratch
### A numpy implementation of the lazy-learner KNN.

## Content
This project contains three files:
1. Conda environment (.yml file): [knn.yml](https://github.com/EmpanS/KNN-From-Scratch/blob/master/knn.yml)
2. The Class KNN (.py file), contains the KNN-algorithm: [knn.py](https://github.com/EmpanS/KNN-From-Scratch/blob/master/knn.py)
3. An example notebook, containing two toy examples on how to use the class: [example.ipynb](https://github.com/EmpanS/KNN-From-Scratch/blob/master/example.ipynb)

The Conda environment is only necessary if one wants to run the jupyter notebook. To use the class [KNN](https://github.com/EmpanS/KNN-From-Scratch/blob/master/knn.py), one only need a numpy installation. The class was built using numpy-version 1.18.1.

## How to use
1. Clone the repository.
2. Create the conda enviroment from the environment file called kMeans.yml by running:
```console
$ conda env create -f knn.yml
```
3. Then activate the enviroment:
```console
$ conda activate knn
```
4. Now, you can either go through the example iPython notebook or play with the KNN class by simply:
```python
from knn import KNN
```

## Lessons Learned
In this small project I got to implement the KNN-algorithm from scratch. It was suprisingly little code. I got to enhance my skills in numpy and learned some tricks to avoid performance issues due to for-loops. For example, when calculating the distance from one point to all other points in the model, I used only numpy arrays instead of looping and calculating the euclidean distance one by one.

Any comments, suggestions or feedback is heavily appreciated. Thanks and happy ML!

Emil Sandström
