# pyMBC3
MBC3 in Python

Works with Python version 3.8.5

Required Dependencies:

Numpy (1.19.1)

Scipy (1.5.2)

Matplotlib (3.3.0)

Openpxyl (3.0.4)

Xlsxwriter (3.0.2)

## How to use pyMBC3

cd dir_with_linearization_files

__python3.8 path_to_pyMBC3.py input.txt__

Sample input.txt files are provided for the two cases provided in the examples.

The above command will create a folder "results" with Campbell data in xlsx format and CampbellDiagram.png file

Ex: 
unzip 5MW_OC4Semi.zip

cd 5MW_OC4Semi

__python3.8 ../pyMBC3.py input.txt__
