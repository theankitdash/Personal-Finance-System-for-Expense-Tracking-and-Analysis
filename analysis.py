import sys
import json
import mysql.connector
from mysql.connector import Error

def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',  
            user='root',     
            password='Chiku@4009', 
            database='finance-tracker'  
        )
        
        if connection.is_connected():
            print("Successfully connected to the database")
            return connection

    except Error as e:
        print(f"Error: {e}")
        return None

def close_connection(connection):
    if connection.is_connected():
        connection.close()
        print("Connection closed")


conn = create_connection()

def fetch_data_from_database(connection):
    try:
        cursor = connection.cursor(dictionary=True)
        query = "SELECT category, amount, date FROM expenses WHERE date >= ? AND date <= ?"  # Adjust the query as needed
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        return rows
    except Error as e:
        print(f"Error fetching data from database: {e}")
        return []

if len(sys.argv) > 1:
    try:
        data = json.loads(sys.argv[1])
        print("Data from Python: ", data)
    except json.JSONDecodeError:
            print("Invalid JSON data provided.")
    except Exception as e:
        print(f"An error occurred: {e}")




close_connection(conn)
