Project: Stock Price Trend Prediction with LSTM

Description:
This project implements a Streamlit web application that predicts stock price trends using a Long Short-Term Memory (LSTM) neural network model. It fetches historical stock data, processes it, and uses a pre-trained LSTM model to forecast future price movements.

Setup Instructions:

Ensure you have Python installed (Python 3.8+ is recommended).

Navigate to the root directory of this project in your terminal.

Install the required Python libraries. These are listed in the 'requirements.txt' file.
It's crucial that the 'requirements.txt' file is in the same directory as 'app.py'.
Run the following command:
pip install -r requirements.txt

Model File:
This application requires a pre-trained Keras LSTM model. The model file 'lstm_stock_prediction_model.keras' must be present in the same directory as 'app.py'. Ensure this file is committed to your repository if deploying to a cloud service like Streamlit Cloud.
Note: If you encounter issues loading the model, it might be due to a TensorFlow/Keras version mismatch. Re-save your model with the TensorFlow version specified in 'requirements.txt' if necessary.

How to Run the Application:

After installing the dependencies and ensuring the model file is in place, run the Streamlit application from your terminal:
streamlit run app.py

This command will open the application in your web browser.

Key Libraries Used:

Streamlit: For building the interactive web application.

yfinance: For fetching historical stock data.

numpy: For numerical operations.

pandas: For data manipulation and analysis.

pandas_ta: For technical analysis indicators (if used in your app).

tensorflow: For building and loading the LSTM model.

keras: High-level API for neural networks, integrated with TensorFlow.
