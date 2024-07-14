import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def financial_analysis(data):
    # Parse JSON data
    try:
        expenses_data = json.loads(data)
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {e}"
    
    # Extract unique categories and their total amounts
    categories = [entry['category'] for entry in expenses_data]
    amounts = np.array([entry['total_amount'] for entry in expenses_data]).reshape(-1, 1)
    
    # Create input features for TensorFlow
    X = np.arange(len(categories)).reshape(-1, 1)
    y = amounts

    # Build a simple neural network model
    model = Sequential([
        Dense(10, activation='relu', input_shape=(1,)),
        Dense(1)
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=200, verbose=0)

    # Predict future expenses for each category
    future_expenses = model.predict(X).flatten()

    # Calculate total and average amounts
    total_amount = np.sum(amounts)
    average_amount = np.mean(amounts)
    
    # Provide suggestions based on predicted expenses
    suggestions = []
    for category, actual, predicted in zip(categories, amounts.flatten(), future_expenses):
        if predicted > average_amount:
            suggestions.append(f"Consider reducing expenses in {category} (Predicted: {predicted:.2f}, Actual: {actual:.2f})")

    # Construct the analysis result in text format
    analysis_result = f"Total Amount: {total_amount:.2f}\n"
    analysis_result += f"Average Amount: {average_amount:.2f}\n\n"
    analysis_result += "Predicted Future Expenses:\n"
    for category, predicted in zip(categories, future_expenses):
        analysis_result += f"  {category}: {predicted:.2f}\n"
    analysis_result += "\nSuggestions:\n"
    if suggestions:
        analysis_result += "\nSuggestions:\n"
        for suggestion in suggestions:
            analysis_result += f"  - {suggestion}\n"
    else:
        analysis_result += "\nNo suggestions for reducing expenses."

    return analysis_result

if __name__ == "__main__":
    # Read financial data from command line argument
    if len(sys.argv) > 1:
        data = sys.argv[1]
        print(f"Received data: {data}")
    else:
        print("Error: No JSON data provided.")
        print("Usage: python script.py '<json_data>'")
        sys.exit(1)

    # Perform financial analysis
    result = financial_analysis(data)

    # Print the result
    print(result)
