import requests
from bs4 import BeautifulSoup

# Function to scrape data from the website
def scrape_website(url):
    # Send an HTTP request to the URL
    response = requests.get(url)
    
    # Check if request was successful
    if response.status_code == 200:
        # Parse HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find and extract data of interest
        # Example: Extracting all <a> tags
        links = soup.find_all('a')
        
        # Return the extracted data
        return links
    else:
        print("Failed to fetch data from the website.")
        return None

# Example URL to scrape
url = 'https://example.com'

# Call the scrape_website function and get the result
data = scrape_website(url)

# Perform analysis on the extracted data
if data:
    # Example analysis: Count the number of links
    num_links = len(data)
    print("Number of links found:", num_links)
    # Further analysis can be done here
else:
    print("No data available for analysis.")
