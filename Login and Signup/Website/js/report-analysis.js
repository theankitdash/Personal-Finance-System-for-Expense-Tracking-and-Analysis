document.addEventListener('DOMContentLoaded', function() {
  // Add event listener to the form submission
  document.getElementById('date-form').addEventListener('submit', function(event) {
      event.preventDefault(); // Prevent default form submission behavior
      
      // Fetch and render expense data
      fetchExpensesData();
  });
});

function fetchExpensesData() {
  const fromDate = document.getElementById('from-date').value;
  const toDate = document.getElementById('to-date').value;

  // Fetch expenses data within the specified time period
  fetch(`/expensesData?fromDate=${fromDate}&toDate=${toDate}`)
      .then(response => response.json())
      .then(data => {
          // Process the data and create the bar graph
          renderPieChart(data);

          // Update analysis results
          updateAnalysisResults(data);
      })
      .catch(error => {
          console.error('Error fetching expenses:', error);
      });
}

function renderPieChart(expenses) {
    const ctx = document.getElementById('graph').getContext('2d');
    new Chart(ctx, {
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


function updateAnalysisResults(data) {
    const analysisResultsDiv = document.getElementById('analysis-results');
    // Clear previous results
    analysisResultsDiv.innerHTML = '';
    // Display analysis results
    const resultParagraph = document.createElement('p');
    if (typeof data === 'string') {
        // If data is a string, directly display it
        resultParagraph.textContent = data;
    } else if (typeof data === 'object') {
        // If data is an object, format it as a string and display
        resultParagraph.textContent = JSON.stringify(data);
    } else {
        // Handle other types of data
        resultParagraph.textContent = "Unknown data type received";
    }
    
    analysisResultsDiv.appendChild(resultParagraph);
}

  