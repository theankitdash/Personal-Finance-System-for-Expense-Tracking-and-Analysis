document.addEventListener('DOMContentLoaded', function() {
  
    fetch('/Details') // Endpoint to get the current user's phone number
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error('Failed to fetch current budget details');
            }
        })
        .then(budgetData => {
          document.getElementById('housingBudget').value = budgetData.housing;
          document.getElementById('transportationBudget').value = budgetData.transportation;
          document.getElementById('foodBudget').value = budgetData.food;
          document.getElementById('utilitiesBudget').value = budgetData.utilities;
          document.getElementById('clothingBudget').value = budgetData.clothing;
          document.getElementById('medicalBudget').value = budgetData.medical;
            
        })
        .catch(error => {
            console.error('Error fetching budget details:', error);
            alert('An error occurred while fetching current budget details');
        });

      // Add event listener to the form submission
      const saveBudgetBtn = document.getElementById('saveBudgetBtn');
      saveBudgetBtn.addEventListener('click', function() {
        // Gather updated budget data
        const updatedBudgetData = {
          housing: document.getElementById('housingBudget').value,
          transportation: document.getElementById('transportationBudget').value,
          food: document.getElementById('foodBudget').value,
          utilities: document.getElementById('utilitiesBudget').value,
          clothing: document.getElementById('clothingBudget').value,
          medical: document.getElementById('medicalBudget').value
        };

          // Send updated budget data to the server
          fetch('/saveBudget', {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json'
              },
              body: JSON.stringify(updatedBudgetData)
          })
          .then(response => {
              if (response.ok) {
                  return response.json();
              } else {
                  throw new Error('Failed to update Budget');
              }
          })
          .then(data => {
              // Handle successful saving of personal details
              console.log('Budget saved successfully:', data);
              alert('Budget saved successfully');
          })
          .catch(error => {
              // Handle error
              console.error('Error saving Budget:', error);
              alert('An error occurred while saving Budget');
          });
      });
});