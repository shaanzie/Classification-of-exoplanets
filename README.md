# Classification-of-exoplanets
Machine Learning techniques for multi-class classification problems with specific focus to exoplanet characteristics.

## Problem Statement

Explore the efficacy of machine learning (ML) in characterizing exoplanets into different classes. The source of
the data used in this work is University of Puerto Rico's Planetary Habitability Laboratory's Exoplanets Catalog
(PHL-EC). Perform a detailed analysis of the structure of the data and propose methods that can be used to
effectively categorize new exoplanet samples. Contributions are two-fold; elaborate on the results obtained
by using ML algorithms by stating the accuracy of each method used and propose a paradigm to automate the task
of exoplanet classification for relevant outcomes. 

## Dataset Cleaning

Almost all the 64 features of the dataset were used to predict the output classes for the final model. These features were first preprocessed and then based on the model, were changed according to the requirement. For example, for the Gaussian and Bernoulli Naive Bayes models, the features were first normalised and fit onto a Gaussian and Bernoulli curve to give accurate results.

For the analysis, 5 output classes were considered.
1. Atmosphere
2. Zone
3. Habitable
4. Mass
5. Composition

The features in itself had no correlation among each other, so we assumed all features to be independent of each other. The output labels had no correlation amongst themselves either, and hence were treated to be independent classes, and the models tried to figure out a meaningful weighted ensemble of these features to try and predict these classes.

Before training the models, the dataset was cleaned.
1. All the NaN values were filled with 0
2. The categorical class names were encoded to integers for ease of classification
3. The features which did not contribute to the classification were removed, the Kepler name for instance.

## Results

| Model                   	| Mass   	| Atmosphere 	| Composition 	| Zone   	| Habitable 	|
|-------------------------	|--------	|------------	|-------------	|--------	|-----------	|
| K Means                 	| -      	| -          	| -           	| -      	| 0.9849    	|
| Decision Trees          	| 0.3170 	| 0.6221     	| 0.5270      	| 0.0835 	| -         	|
| Bernoulli Naive Bayes   	| 0.7817 	| 0.6954     	| 0.9854      	| 0.9683 	| 0.9832    	|
| Gaussian Naive Bayes    	| 0.6645 	| 0.6761     	| 0.6864      	| 0.9496 	| 0.9651    	|
| Support Vector Machines 	| 0.9832 	| 0.9935     	| 0.9896      	| 0.9942 	| 0.9858    	|

The Support Vector Machine model seems to be the most optimal model that can be used in such a classification problem with this dataset, as it gave high accuracies for the predictable classes. 
There might be a case of overfitting not explored, but since the validation was created through a simple random sample, the model seems to be performing well. T
he model appears to be skewed towards a habitable class, because of how the dataset was created in the first place, so there were not enough samples to clearly distinguish between the non habitable and habitable class.
More examples of the non habitable class may have led to less skewness, but for the existing dataset, the model seems to be working.

