document.addEventListener('DOMContentLoaded', function() {
    // Add event listener to the form submission
    document.getElementById('date-form').addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent default form submission behavior
        
        // Fetch and render expense data
        fetchExpensesData();
    });

    // Add event listener to the analyze button
    document.getElementById('analyze-btn').addEventListener('click', function() {
        // Fetch and update analysis results
        const expensesData = JSON.parse(document.getElementById('graph').dataset.expenses);
        fetchAndUpdateAnalysisResults(expensesData);
    });
});

let chartInstance = null;

function fetchExpensesData() {
  const fromDate = document.getElementById('from-date').value;
  const toDate = document.getElementById('to-date').value;

  // Fetch expenses data within the specified time period
  fetch(`/expensesData?fromDate=${fromDate}&toDate=${toDate}`)
      .then(response => response.json())
      .then(data => {
          // Process the data and create the bar graph
          renderPieChart(data);

          // Show the analyze button after rendering the graph
          document.getElementById('analyze-btn').style.display = 'inline-block';

          // Store the expenses data in the graph element's dataset for later use
          document.getElementById('graph').dataset.expenses = JSON.stringify(data);
      })
      .catch(error => {
          console.error('Error fetching expenses:', error);
      });
}

function renderPieChart(expenses) {
    const ctx = document.getElementById('graph').getContext('2d');

    // Destroy the previous chart instance if it exists
    if (chartInstance) {
        chartInstance.destroy();
    }

    // Create a new chart instance
    chartInstance = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: expenses.map(expense => expense.category),
            datasets: [{
                label: 'Total Spent',
                data: expenses.map(expense => expense.total_amount),
                backgroundColor: [
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(255, 159, 64, 0.2)'
                    // Add more colors if you have more categories
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)'
                    // Add more colors if you have more categories
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                yAxes: [{
                    display: false
                }]
            }
        }
    });
}

async function fetchAndUpdateAnalysisResults(expensesData) {
    try {
        const response = await fetch('/analyzeFinancialData', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ data: expensesData }) 
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.text();
        updateAnalysisResults(result);
    } catch (error) {
        console.error('Error fetching analysis results:', error);
    }
}


function updateAnalysisResults(data) {
    const analysisResultsDiv = document.getElementById('analysis-results');
    analysisResultsDiv.innerHTML = '';
    const resultParagraph = document.createElement('p');
    resultParagraph.textContent = data;
    analysisResultsDiv.appendChild(resultParagraph);

    // Show the analysis section
    document.getElementById('analysis-section').style.display = 'block';
}

  