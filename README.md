# Generic classifier pipeline

### This project contains: <p>

* functions to load and transform data from a database or a csv file
* visualization tools 
* standard exploratory data analysis and preprocessing methodologies
* grid search via cross validation for the model and related hyperparameters selection 
* unit testing framework

## Data extraction package:

The package named 'dataset_elt' provides a simple way to load the required dataset both from a Onesait ontology or from csv file. <p>
If the data extraction is from a Onesait ontology, the user must enter:<p>

* iot_client_host: server URL where the ontology is stored
* iot_client_name: client name whith granted access  
* iot_client_token: valid token

Otherwise, if the data is stored locally as a file:

* dataset_location: storage location of the file including the file name

## Dataset visualizations

The package called 'dataset_plots' has several utilities to plot the extracted/loaded dataset, as part of the exploratory analysis. Data visualization is a useful tool to get a first overview about the data before implementing the pipeline changes. Visualizations must be taken carefully, since it may even arise confusions if deeper analysis is not carried out.

## Dataset EDA

As already mentioned, we can implement an exploratory analysis on the dataset; with this, we can automate the selection of attributes by means of found correlations and other aspects. An example of a Pandas profile report can be found at '\dataset_EDA\eda_reports' folder. With this useful and interactive report in HTML format, we can check descriptive statistics values, correlation matrixes and even automate the selected features based on a given correlation threshold: 

![Alt text](/readme_files/EDA_2_opt.png "EDA example 1")
![Alt text](/readme_files/EDA_opt.png "EDA example 2")

## Dataset preprocessing

With the 'dataset_preprocessing' package, we can look for missing values, impute them with the desired strategy, look for outliers...

## Modelling

Once we have prepared the dataset to be modeled, with the 'modelling' package we can implement a grid search cross validation strategy for the model and hyperparameter selection. The user of the package can enter a list of possible models, with the corresponding possible hyperparameters to try with.
As a result, we obtain a table (actually, a pandas dataframe) with best combination of hyperparameters for each one of the trained models.
The selected models must be part of the scikit-learn library, although the same logic can be applied to other frameworks.

## Validation

Finally, the best model is finally selected based on a given desired metric among the best models returned by the Modelling step. 
An example of the cross validation metrics result, for each model, is as follows, showing a mean_test_recall of 0.98 for the best classifier:

![Alt text](/readme_files/cross_validation_results_opt.jpg "Validation results")

For this example, since there is a high unbalance between rain/no rain data, and considering that a weather predictor should be accurate predicting rain, we have chosen 'recall' as the main validation metric focusing on detecting true positives. More info on recall meaning: https://en.wikipedia.org/wiki/Precision_and_recall

## Testing framework

As with any software project, testing is compulsory for delivering a reliable product; with frameworks like unittest, py.test... we can implement as many code and logic tests as we wish:

![Alt text](/readme_files/tests_opt.jpg "Tests")