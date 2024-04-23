document.addEventListener('DOMContentLoaded', function() {

    fetch('/Details') // Endpoint to get the current user's Details
    .then(response => {
        if (response.ok) {
            return response.json();
        } else {
            throw new Error('Failed to fetch current credentials');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while fetching current credentials');
    });

    // Add event listener to the form submission
    const saveExpenseBtn = document.getElementById('saveExpenseBtn');
    saveExpenseBtn.addEventListener('click', function(event) {
        event.preventDefault(); 

        const date = document.getElementById('expenseDate').value;
        const amount = document.getElementById('expenseAmount').value;
        const description = document.getElementById('expenseDescription').value;
        const category = document.getElementById('expenseCategory').value;

        // Send request to save Expenses
        fetch('/saveExpenses', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ date, amount, description, category })
        })
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error('Failed to save Expense');
            }
        })
        .then(data => {
            // Handle successful saving of Expenses
            console.log('Expense saved successfully:', data);
            alert('Expense saved successfully');
        })
        .catch(error => {
            // Handle error
            console.error('Error saving Expense:', error);
            alert('An error occurred while saving Expense');
        });
    });
});

