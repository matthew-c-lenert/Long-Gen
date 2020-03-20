# The Long-Gen Package
A library of functions for generating longitudinal data

## Package Definitions

This library creates has a suite of functions for generating synthetic longitudinal data (data that is self correlated over time). This package allows the user to finely control various temporal properties of the data, such as its stationarity, the type of relationship between the features and the outcome, the collinearity/auto-collinearity of the features, and others. The data is meant to resemble synchronous clinical data with differing numbers of measurements for a set of patients. This library uses a longitudinal random effects model as a data generating process. This is a hierarchical longitudinal regression equation with both global beta coefficients (fixed effects) and patient specific coefficients (random effects): https://en.wikipedia.org/wiki/Mixed_model.


## Package Classes, Functions, and Parameters Definitions

## Classes:


#### long_data_set(n=2000,num_measurements=25,collinearity_bucket="low-low",trend_bucket="linear",sampling_bucket="random",sampling_function=None,b_colin=0.13,beta_var=1,time_importance_factor=5,sigma_e=0.05,num_features=2,num_extraneous_variables=0,link_fn="identity",num_piecewise_breaks=0,random_effects=["intercept","time","trend-time"],coefficient_values={})

This initializes the class with the desired parameters. It gets the class ready to create data.

Parameters:
n: the number of unique patients in the data set (integer). Default = 2000.

num_measurements: the average number of measurements per patients. The number of measurement for a specific patient is drawn from a Pareto distribution with median roughly equal to the integer specified here (integer). Default = 25.

collinearity_bucket: the level of correlation between different features and the level of autocorrelation of a feature with itself over time. Features are different draws from a Gaussian Process. Feature values are determined by the timing of the sample. The buckets represent different parameter sets for the Gaussian Process. Default is "low-low". Please specify one of the following buckets as a string:
   1. "low-low" : low collinearity(0.1-0.4), low autocorrelation (0.1-0.4)
   2. "low-moderate" : low collinearity(0.1-0.4), moderate autocorrelation (0.4-0.7)
   3. "low-high" : low collinearity(0.1-0.4), high autocorrelation (0.7-0.9)
   4. "moderate-high" : moderate collinearity (0.4-0.6), high autocorrelation (0.7-0.99)
   5. "high-high" : high collinearity (0.6-0.9), high autocorrelation (0.7-0.99)

trend_bucket: The type of trend over time in the outcome variable. Default is "linear". Please specify one of the following buckets as a string:
   1. "linear" : a linearly increasing or decreasing trend over time
   2. "quadratic" : a non linear trend (2nd order polynomial) over time
   3. "seasonal" : a sinusoidal trend with a linear increase or decrease over time

sampling_bucket: the type of sampling scheme used to determine the timing of measurements. Default is "random". Please specify one of the following buckets:
   1. "equal" : the timing of measurements are equally spaced across the time interval (0,1). You can use the transform function discussed later to change the time interval to one of your choosing.
   2. "random" : the timing of measurements are uniformly randomly drawn across the interval from (0,1).
   3. "not-random": the timing of measurements start off equally spaced across the time interval (0,0.5). Any abnormal observation (greater than 1 or less than -1) in the feature set will cause the sampling frequency to increase proportional to the extremeness of the feature value until all features return to the normal range. If there are multiple values out of normal range, then the sampling frequency is increased proportional to the most extreme value. The increase in sampling frequency occurs after the abnormal observation(s). The number of abnormal features are correlated with the total number of measurements and the length of the sampling period. For example, if all measurements are abnormal the sampling period extends to (0,1).
   4. "custom-no-features" : specify your own function to determine the timing of samples. This function should take the number of measurements (integer) as a parameter and should output a numpy array of numeric sample times of size equal to the parameter. Features will be created via Gaussian Process.
   5. "custom-feature-values" : specify your own function to determine the timing of samples. This function should take two parameters: 1st: the number of measurements, 2nd a dictionary of numpy arrays with the feature values. This function should output a numpy array of size equal to the number of measurements. Features will be created via a correlated multivariate normal distribution (no autocorrelation).

sampling_function: if using a custom sampling function please specify it here. Default is None. Please remember to also set the sampling_bucket parameter to either "custom-no-features" or "custom-feature-values".

b_colin: this parameter determines how collinear patient specific effects (random effects) are. For example, setting this number to one would mean that the patient specific intercept and time slope would be perfectly correlated (if the slope is positive, the intercept will also be positive). Specify a number in (0,1). The deault is 0.13.

beta_var: set the magnitude of feature and time coefficients. Larger numbers will result in larger feature and time effects. These effects are drawn from a normal distribution centered at 0. Please choose a number greater than 0. The default is 1.

b_var: set the magnitude of the personalized offsets for the intercept and time coefficients. Larger numbers will result in larger inter-subject variability. These effects are drawn from a normal distribution centered at 0. Please choose a number greater than 0. The default is 1.

time_importance_factor: Determine how relatively important the time coefficient should be compared to feature effects. Values above 1 will mean time should be more important and values less than 1 mean time should be less important. One will likely want to use this parameter if they are doing variable transforms later on. Please choose a number greater than 0. Default is 5.

sigma_e: set the amount of observation error for each measurement of the outcome. Larger values mean larger amounts of unobserved measurement error. Smaller values mean more precise measurements. The distribution of the error is determined by the link function. We use the canonical error distributions for each link function and hold the variance of the error constant for each patient. Specify a number in [0,1). Default is 0.05.

num_features: the number of numeric features you wish to generate. Features will be autocorrelated if the sampling scheme is "equal", "random", or "custom-no-features". All features start off as real valued, but you can transform them with a function of your choosing (details below). Please specify a non-negative integer. The default is 2.

num_extraneous_variables: the number of numeric variables that have no effect on the outcome, but are also measured with the features and the outcome. Extraneous variables will be autocorrelated if the sampling scheme is "equal", "random", or "custom-no-features". All extraneous variables start off as real valued, but you can transform them with a function of your choosing (details below). Please specify a non-negative integer. The default is 0.

link_fn: the type of relationship the features have with the outcome, it also determines the distribution of the error. In a standard regression equation this would be the identity function making the relationship linear. Default is "identity". Please choose one of the following:
   1. "identity" : no transformation, normally distributed error.
   2. "log" : an exponential relationship between the features and the outcomes, poisson distributed error.
   3. "logit" : an expit relationship between the features and the outcome. The outcome is binary (0 or 1), but y_prob represents the true probability of y. The error is binomially distributed.
   4. "inverse" : a different flavor of exponential relationship between the features and the outcomes, gamma distributed error.

num_piecewise_breaks: the number of times the global (fixed effect) coefficients of the model shift. Instead of having just one model over the whole time interval, you can introduce global shifts that happen at specific time points. These shifts might represent the progression of disease or the instability of a process. Please specify a non-negative integer. The default is 0.

random_effects: features that have patient specific values. These patient specific (random) effects determine the correlation structure of the outcome and create inter-subject variability. It is recommended that you at least specify ["intercept","time"] as random effects to cause the outcome to be correlated over time. Non-linear time components can have patient specific effects by adding "trend-time" to the list. You can give features a patient specific effect by adding them to the list like so: ["intercept","time","x1"]. Please specify a list of features and the intercept. Patient specific effects are normally distributed and centered at 0 as per standard assumptions of the random effects model. Default is ["intercept","time","trend-time"].

coefficient_values: you can specify a dictionary of coefficient values to create a precise model. The dictionary should have the feature name as the key and the numeric coefficient as the value. You must give all features including "time", "trend-time", and "intercept" a value. This parameter should not be used without careful study of the underlying code. This parameter is not supported with num_piecewise_breaks > 0. The default is {}.

time_breaks: you can specify where you wish piecewise breaks to occur in the interval (0,1). If time_breaks is left empty while the num_piecewise_breaks > 0, then the location of the piecewise breaks will be selected at random over the (0,1) interval. If the list is not empty, then the list size must be equal to the number of piecewise breaks. The default is [].

#### create_data_set()

This function creates a data set based on the initialized parameters in long format and stores that data set as a Pandas data frame in the data_frame attribute of the class.

#### export_to_csv(path_name,file_name)

This function exports the data frame saved in the data_frame attribute to a csv. It also creates a separate csv that describes the parameters used to create the data.

Parameters:

path_name: the directory where the csv's should be saved

file_name: a descriptive name of the data set. Note that the data_frame csv will have "data_" prepended to this descriptive name and the model parameter csv will have "params_" prepended to this descriptive name.

#### transform_variable_feature(column_names,transformation_function)

This function applies a custom function to transform a numeric feature or extraneous variable. This function does all the tedious work of unapplying the error and the link function as to update the values of the feature, extraneous variable, and, if needed, the outcome. This function will create new variables/features/outcome with "new_" prepended to the feature/variable name changed. If you want to apply multiple transformations, make sure you replace the outcome variable "y" with "new_y" from the first transformation before applying the second. I would strongly discourage applying transformations without use of this function, please refer to code to see why. This function is supported with num_piecewise_breaks > 0.

Parameters:

column_names: a list of the column names you want to transform. These can be features (including "time") or extraneous variables. Just add them to a list like so: ["x1","x2","time"].

transformation_function: this function should take a numpy array as an input and should return a numpy array of the same size. You may apply whatever numerical transformation you wish so long as the result is in the real numbers and an array of equal size to the input is returned.


## Citing this Package
Please cite this package if used in research applications. Please cite with the following: Matthew C Lenert, Jeffrey Blume, Thomas Lasko, Michael Matheny, Asli Weitkamp, Colin G Walsh. "Deep Aion Project: Longitudinal Data Generator". https://github.com/matthew-c-lenert/Long-Gen.
