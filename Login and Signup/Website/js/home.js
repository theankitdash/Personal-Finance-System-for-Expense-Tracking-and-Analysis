document.addEventListener('DOMContentLoaded', function() {
  // Fetch current user's details
  fetch('/Details')
    .then(response => {
      if (response.ok) {
        return response.json();
      } else {
        throw new Error('Failed to fetch current credentials');
      }
    })
    .then(data => {
      // Display Name
      document.getElementById('name').textContent = data.name || '';

      // Fetch and render expense data for the current month
      fetchCurrentMonthExpenses();
    })
    .catch(error => {
      console.error('Error:', error);
      alert('An error occurred while fetching current credentials');
    });
});

function fetchCurrentMonthExpenses() {
  // Fetch expenses data for the current month
  fetch('/currentMonthExpenses')
    .then(response => response.json())
    .then(data => {
      // Process the data and create the bar graph
      renderBarGraph(data);
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
    if (!categories[category]) {
      categories[category] = 0;
    }
    categories[category] += parseFloat(expense.amount);
  });

  // Prepare data for Chart.js
  const categoryLabels = Object.keys(categories);
  const categoryAmounts = Object.values(categories);

  // Render the bar graph using Chart.js
  const ctx = document.getElementById('expenseChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: categoryLabels,
      datasets: [{
        label: 'Total Expenses',
        data: categoryAmounts,
        backgroundColor: [
          'rgba(255, 99, 132, 0.2)',
          'rgba(54, 162, 235, 0.2)',
          'rgba(255, 206, 86, 0.2)',
          'rgba(75, 192, 192, 0.2)',
          'rgba(153, 102, 255, 0.2)',
          'rgba(255, 159, 64, 0.2)'
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(153, 102, 255, 1)',
          'rgba(255, 159, 64, 1)'
        ],
        borderWidth: 1
      }]
    },
    options: {
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            color: '#333', // Light text color for dark background
            font: {
              weight: 'bold' // Make font bold
            }
          },
          grid: {
            color: 'rgba(0, 0, 0, 0.1)' // Dark grid lines for contrast
          }
        },
        x: {
          ticks: {
            color: '#333', // Dark text color for light background
            font: {
              weight: 'bold' // Make font bold
            }
          },
          grid: {
            color: 'rgba(0, 0, 0, 0.1)' // Dark grid lines for contrast
          }
        }
      },
      plugins: {
        legend: {
          labels: {
            color: '#333', // Dark text color for legend
            font: {
              weight: 'bold' // Make font bold
            }
          }
        }
      },
      responsive: true,
      maintainAspectRatio: false
    }
  });
}