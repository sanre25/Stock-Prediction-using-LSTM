import numpy as np
import pandas as pd


def evaluate():
    # Input the csv file
    """
    Sample evaluation function
    Don't modify this function
    """
    df = pd.read_csv('sample_input.csv')
     
    actual_close = np.loadtxt('sample_close.txt')
    
    pred_close = predict_func(df)
    
    # Calculation of squared_error
    actual_close = np.array(actual_close)
    pred_close = np.array(pred_close)
    mean_square_error = np.mean(np.square(actual_close-pred_close))


    pred_prev = [df['Close'].iloc[-1]]
    pred_prev.append(pred_close[0])
    pred_curr = pred_close
    
    actual_prev = [df['Close'].iloc[-1]]
    actual_prev.append(actual_close[0])
    actual_curr = actual_close

    # Calculation of directional_accuracy
    pred_dir = np.array(pred_curr)-np.array(pred_prev)
    actual_dir = np.array(actual_curr)-np.array(actual_prev)
    dir_accuracy = np.mean((pred_dir*actual_dir)>0)*100

    print(f'Mean Square Error: {mean_square_error:.6f}\nDirectional Accuracy: {dir_accuracy:.1f}')
    

def predict_func(data):
    """
    Modify this function to predict closing prices for next 2 samples.
    Take care of null values in the sample_input.csv file which are listed as NAN in the dataframe passed to you 
    Args:
        data (pandas Dataframe): contains the 50 continuous time series values for a stock index

    Returns:
        list (2 values): your prediction for closing price of next 2 samples
    """
    import numpy as np
    import pandas as pd 
    import joblib
    
    model = joblib.load('Model_param.sav')
    
    
    
    data = data.interpolate() 
    FullData=data[['Close']].values
    
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    sc=MinMaxScaler()
    DataScaler = sc.fit(FullData)
    
    Last50Days = np.array(data['Close'])
    # Normalizing the data just like we did for training the model
    Last50Days=sc.transform(Last50Days.reshape(-1,1))

    # Changing the shape of the data to 3D
    # Choosing TimeSteps as 50 because we have used the same for training
    NumSamples=1
    TimeSteps=50
    NumFeatures=1
    Last50Days=Last50Days.reshape(NumSamples,TimeSteps,NumFeatures)

    #############################

    # Making predictions on data
    predicted_Price = model.predict(Last50Days)
    predicted_Price = sc.inverse_transform(predicted_Price)
    
    return np.array(predicted_Price[0][0:2])
    

if __name__== "__main__":
    evaluate()