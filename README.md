# IPSC_Model
Code for the model to predict the effect of various factors on the Manufacturing of Induced Pluripotent Stem Cells

### The Model:
The 'Model' class is a fairly standard multi-layered perceptron model with three hidden layers. It takes the input from the 'data.csv' class, processes it, and then applies three percetron layers. The activation function is just the sigmoid (Model.nonlin()).

### The Data: 
The 'Scraper' class is a scraper that, when given the link to an Amazonia! Gene Microarray list (any of the links here: http://amazonia.transcriptome.eu/myAmaZonia.php?section=list), moves down the tree structure and dumps the signal strength for each gene ('genes.csv') in a specific cell into a csv file ('data.csv')

This data is then one-hot encoded in the 'model.init_data()' function and sliced into input (the gene abbreviation, the cell, and the p-value) and output (the signal of the gene in that cell) for prediction training.

It employs the 'csv' and 'Beautiful Soup' Libraries
