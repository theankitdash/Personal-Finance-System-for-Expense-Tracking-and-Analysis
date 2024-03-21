// JavaScript for financial budget and expenses

document.addEventListener('DOMContentLoaded', function() {
    // Validate budget allocation form
    const budgetForm = document.querySelector('#budget-form');
    budgetForm.addEventListener('submit', function(event) {
      const budgetInputs = document.querySelectorAll('#budget-form input[type="number"]');
      let isValid = true;
      budgetInputs.forEach(function(input) {
        if (parseInt(input.value) < 0) {
          isValid = false;
          input.classList.add('error');
        } else {
          input.classList.remove('error');
        }
      });
      if (!isValid) {
        event.preventDefault(); // Prevent form submission if any input is invalid
        alert('Budget amount cannot be negative!');
      }
    });
  
    // Validate expense submission form
    const expenseForm = document.querySelector('#expense-form');
    expenseForm.addEventListener('submit', function(event) {
      const expenseAmountInput = document.querySelector('#expenseAmount');
      if (parseInt(expenseAmountInput.value) <= 0) {
        event.preventDefault(); // Prevent form submission if expense amount is not positive
        alert('Expense amount must be greater than zero!');
      }
    });
  });
  