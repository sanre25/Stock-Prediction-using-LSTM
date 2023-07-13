# Information regarding files

- **Model_param.sav**: Contains model parameters, i.e., the trained model.
- **210922_Sandeep**: Contains a Python script in which the `predict_func` is modified and used for evaluation as per the guidelines.
- **sample_input**: Same as given in the project.
- **sample_close**: Same as given in the project.
- **Train_Model_Jupy**: Contains a Jupyter Python notebook file in which the model is trained.
- **Test_Model_Jupy**: Contains a Jupyter Python notebook file for testing.

# Libraries Used

- numpy
- pandas (for dataframe)
- scikit-learn (StandardScaler, MinMaxScaler)
- keras (Sequential, Dense, LSTM)
- matplotlib.pyplot
- joblib (save the model to disk)

# Guidelines

- To train the model, you need to load the `STOCK_INDEX.csv` file in the working directory.
- `Model_param` should be in the working directory to check predictions for new test data.
- `Test_Model_Jupy` notebook only contains the testing part to check the working of `predict_func()` and `evaluate()` functions.
