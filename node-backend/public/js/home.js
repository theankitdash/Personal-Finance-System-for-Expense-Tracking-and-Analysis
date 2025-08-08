document.addEventListener('DOMContentLoaded', function() {
  
  // Fetch current user's details
  fetch('/personalDetails')
    .then(response => {
      if (!response.ok) {
        throw new Error('Failed to fetch current credentials');
      }
      return response.json();
    })
    .then(data => {
      // Display Name
      document.getElementById('name').textContent = data.personalDetails.name || '';

      // Fetch and render expense data for the current month
      fetchCurrentMonthExpenses();
    })
    .catch(error => {
      console.error('Error:', error);
      alert('An error occurred while fetching current credentials');
    });
});

  // Fetch current user's budget
  fetch('/getBudgetDetails')
  .then(response => {
      if (response.ok) {
          return response.json();
      } else {
          throw new Error('Failed to fetch current credentials');
      }
  })
  .then(data => {    
      //console.log(data);
      if (data.success && Array.isArray(data.budgets)) {
          // Initialize a dictionary to hold budget values
          const budgetDict = {};
          
          // Populate the dictionary with data from the budgets array
          data.budgets.forEach(budget => {
              budgetDict[budget.category] = budget.amount;
          });
          
          // Populate the input fields with fetched data
          document.getElementById('clothing-budget').value = budgetDict['Clothing'] || '';
          document.getElementById('entertainment-budget').value = budgetDict['Entertainment'] || '';
          document.getElementById('food-budget').value = budgetDict['Food'] || '';
          document.getElementById('housing-budget').value = budgetDict['Housing'] || '';
          document.getElementById('investment-budget').value = budgetDict['Investment'] || '';
          document.getElementById('medical-budget').value = budgetDict['Medical'] || '';
          document.getElementById('other-budget').value = budgetDict['Other'] || '';
          document.getElementById('transportation-budget').value = budgetDict['Transportation'] || '';
          document.getElementById('utilities-budget').value = budgetDict['Utilities'] || '';
      } else {
          console.error('Unexpected data format or failed request.');
      }
  })
  .catch(error => {
      console.error('Error:', error);
  });

  document.getElementById('save-budget-btn').addEventListener('click', function() {
      const budgetDetails = {
          'Clothing': document.getElementById('clothing-budget').value,
          'Entertainment': document.getElementById('entertainment-budget').value,
          'Food': document.getElementById('food-budget').value,
          'Housing': document.getElementById('housing-budget').value,
          'Investment': document.getElementById('investment-budget').value,
          'Medical': document.getElementById('medical-budget').value,
          'Other': document.getElementById('other-budget').value,
          'Transportation': document.getElementById('transportation-budget').value,
          'Utilities': document.getElementById('utilities-budget').value
      };

      // Create an array to hold the promises
      const promises = [];

      // Loop through the budgetDetails object and send each category's amount to the server
      for (const [category, amount] of Object.entries(budgetDetails)) {
              // Push each fetch request promise into the promises array
              promises.push(
                  // Send an AJAX POST request to the server
                  fetch('/saveBudgetDetails', {
                      method: 'POST',
                      headers: {
                          'Content-Type': 'application/json'
                      },
                      body: JSON.stringify({ category, amount })
                  })
                  .then(response => response.json())
                  .then(data => {
                      if (!data.success) {
                          throw new error(`Failed to save budget for ${category}: ${data.message}`);
                      }
                  })
                  .catch(error => {
                      console.error('Error saving budget details:', error);
                  })
              );
      }

      // Use Promise.all to handle all fetch requests
      Promise.all(promises)
          .then(() => {
              alert('Budget details saved successfully.');
          })
          .catch(error => {
              console.error('Error processing budget details:', error);
      });
  });


function fetchCurrentMonthExpenses() {
  fetch('/currentMonthExpenses')
    .then(response => response.json())
    .then(expensesData => {
      // Process the data and create the bar graph
      renderBarGraph(expensesData);
    })
    .catch(error => {
      console.error('Error fetching current month expenses:', error);
    });
}


function renderBarGraph(expenses) {
  // Extract category-wise data from expenses
  const categories = {};
  expenses.forEach(expense => {
    const category = expense.category;
    if (!categories[category]) categories[category] = 0;
    categories[category] += parseFloat(expense.amount);
  });

  // Prepare data for Chart.js
  const categoryLabels = Object.keys(categories);
  const categoryAmounts = Object.values(categories);

  // Define colors for the chart
  const colors = {
    background: [
      'rgba(255, 99, 132, 0.2)',
      'rgba(54, 162, 235, 0.2)',
      'rgba(255, 206, 86, 0.2)',
      'rgba(75, 192, 192, 0.2)',
      'rgba(153, 102, 255, 0.2)',
      'rgba(255, 159, 64, 0.2)',
      'rgba(199, 199, 199, 0.2)',
      'rgba(83, 102, 255, 0.2)',
      'rgba(255, 255, 99, 0.2)',
      'rgba(255, 105, 180, 0.2)'
    ],
    border: [
      'rgba(255, 99, 132, 1)',
      'rgba(54, 162, 235, 1)',
      'rgba(255, 206, 86, 1)',
      'rgba(75, 192, 192, 1)',
      'rgba(153, 102, 255, 1)',
      'rgba(255, 159, 64, 1)',
      'rgba(199, 199, 199, 1)',
      'rgba(83, 102, 255, 1)',
      'rgba(255, 255, 99, 1)',
     'rgba(255, 105, 180, 1)'
    ]
  };

  // Render the bar graph using Chart.js
  const ctx = document.getElementById('expenseChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: categoryLabels,
      datasets: [{
        data: categoryAmounts,
        backgroundColor: colors.background,
        borderColor: colors.border,
        borderWidth: 1
      }]
    },
    options: {
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            color: '#333', 
            font: {
              weight: 'bold' 
            }
          },
          grid: {
            color: 'rgba(0, 0, 0, 0.1)' 
          }
        },
        x: {
          ticks: {
            color: '#333', 
            font: {
              weight: 'bold' 
            }
          },
          grid: {
            color: 'rgba(0, 0, 0, 0.1)' 
          }
        }
      },
      plugins: {
        title: {
          display: true, 
          text: 'Current Month Expenses', 
          color: '#333', 
          font: {
            size: 18, 
            weight: 'bold' 
          }
        },
        legend: {
          display: false
        }
      },
      responsive: true,
      maintainAspectRatio: false
    }
  });
}