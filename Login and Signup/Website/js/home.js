document.addEventListener('DOMContentLoaded', function() {

  fetch('/Details') // Endpoint to get the current user's Details
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
      
  })
  .catch(error => {
      console.error('Error:', error);
      alert('An error occurred while fetching current credentials');
  });

  // Create a sample chart using Chart.js
  const ctx = document.getElementById('expenseChart').getContext('2d');
  const expenseChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Rent', 'Groceries', 'Utilities', 'Entertainment', 'Transportation'],
      datasets: [{
        label: 'Monthly Expenses',
        data: [1000, 500, 300, 200, 400],
        backgroundColor: [
          'rgba(255, 99, 132, 0.5)',
          'rgba(54, 162, 235, 0.5)',
          'rgba(255, 206, 86, 0.5)',
          'rgba(75, 192, 192, 0.5)',
          'rgba(153, 102, 255, 0.5)'
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(153, 102, 255, 1)'
        ],
        borderWidth: 1
      }]
    },
    options: {
      scales: {
        yAxes: [{
          ticks: {
            beginAtZero: true
          }
        }]
      }
    }
  });
});
