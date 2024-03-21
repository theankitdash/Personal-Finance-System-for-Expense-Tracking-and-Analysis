// JavaScript for financial spending analysis

document.addEventListener('DOMContentLoaded', function() {
    // Sample data for the graph
    const labels = ['January', 'February', 'March', 'April', 'May', 'June'];
    const data = [1000, 1500, 1200, 1800, 2000, 1600];
  
    // Function to update the graph
    function updateGraph() {
      // Code to update the graph based on selected date range
      // For demonstration purposes, let's just redraw the same graph with sample data
      const ctx = document.getElementById('graph').getContext('2d');
      const graph = new Chart(ctx, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [{
            label: 'Monthly Spending',
            data: data,
            fill: false,
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1
          }]
        },
        options: {
          scales: {
            y: {
              beginAtZero: true
            }
          }
        }
      });
    }
  
    // Handle form submission
    const form = document.getElementById('date-form');
    form.addEventListener('submit', function(event) {
      event.preventDefault(); // Prevent default form submission behavior
      updateGraph(); // Update the graph
    });
  
    // Initial graph rendering
    updateGraph();
  });
  