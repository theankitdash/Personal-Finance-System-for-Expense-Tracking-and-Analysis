import sys
import json
from collections import defaultdict
from dateutil import parser
from datetime import timedelta
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

def analyze_financial_data(aggregated_data, detailed_data):
    # Aggregate Data Analysis
    month_category_data = defaultdict(lambda: defaultdict(float))
    for item in aggregated_data:
        try:
            month_category_data[item['month']][item['category']] += item['total_amount']
        except (KeyError, TypeError):
            print(f"Skipping invalid aggregated item: {item}")

    # Detailed Data Analysis
    for item in detailed_data:
        try:
            date = parser.parse(item['date']) + timedelta(days=1)
            month = date.strftime('%b-%Y')
            month_category_data[month][item['category']] += item['amount']
        except (ValueError, KeyError, TypeError):
            print(f"Skipping invalid detailed item: {item}")

    # Generate report lines
    report_lines = []
    for month, categories in sorted(month_category_data.items()):
        report_lines.append(f"Month: {month}")
        for category, amount in sorted(categories.items()):
            report_lines.append(f"  Category: {category}, Amount: INR {amount:.2f}")
        report_lines.append("")  # Blank line for separation
    
    total_amount = sum(amount for categories in month_category_data.values() for amount in categories.values())
    report_lines.append(f"Total Amount Spent: INR {total_amount:.2f}")
    
    return "\n".join(report_lines), month_category_data

def prepare_data(month_category_data):
    # Flatten the month_category_data into a list of amounts
    data = []
    for month, categories in sorted(month_category_data.items()):
        for category, amount in sorted(categories.items()):
            data.append(amount)

    # Convert to numpy array and reshape for LSTM
    data = np.array(data).reshape(-1, 1)
    
    # Scale data to the range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    
    return data, scaler

def create_prediction_model(data):
    # Define the model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(data.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def main():
    if len(sys.argv) > 1:
        try:
            input_data = json.loads(sys.argv[1])
            aggregated_data = input_data.get('aggregatedData', [])
            detailed_data = input_data.get('detailedData', [])
            
            report, month_category_data = analyze_financial_data(aggregated_data, detailed_data)
            print(report)
            
            # Prepare data for TensorFlow model
            prepared_data, scaler = prepare_data(month_category_data)
            
            # Create and train the model
            model = create_prediction_model(prepared_data)
            
            # Reshape data for LSTM input
            prepared_data = np.reshape(prepared_data, (prepared_data.shape[0], 1, 1))
            
            # Train the model
            model.fit(prepared_data, prepared_data, batch_size=1, epochs=1)
            
            # Make a prediction (example, modify according to your needs)
            predicted_expenses = model.predict(prepared_data)
            predicted_expenses = scaler.inverse_transform(predicted_expenses)
            
            print("Predicted Expenses: ", predicted_expenses)
            
        except json.JSONDecodeError:
            print("Invalid JSON data provided.")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("No data provided.")

if __name__ == "__main__":
    main()
