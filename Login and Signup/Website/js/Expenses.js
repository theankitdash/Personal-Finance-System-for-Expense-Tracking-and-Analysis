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
    .catch(error => {
        console.error('Error fetching user details:', error);
        alert('An error occurred while fetching user details');
    });

    // Add event listener to the form submission
    const saveExpenseBtn = document.getElementById('saveExpenseBtn');
    saveExpenseBtn.addEventListener('click', function(event) {
        event.preventDefault(); // Prevent default form submission behavior

        const date = document.getElementById('expenseDate').value;
        const amount = document.getElementById('expenseAmount').value;
        const description = document.getElementById('expenseDescription').value;
        const category = document.getElementById('expenseCategory').value;

        // Send request to save expense
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
                throw new Error('Failed to save expense');
            }
        })
        .then(data => {
            // Handle successful saving of expense
            console.log('Expense saved successfully:', data);
            alert('Expense saved successfully');

            // After saving the expense, update the expenses history
            updateExpensesHistory();
        })
        .catch(error => {
            // Handle error
            console.error('Error saving expense:', error);
            alert('An error occurred while saving expense');
        });
    });

    // Function to update expenses history
    function updateExpensesHistory() {
        // Fetch expenses history
        fetch('/expensesHistory')
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error('Failed to fetch expenses history');
            }
        })
        .then(expenses => {
            // Clear existing expenses list
            const expensesList = document.getElementById('expensesList');
            expensesList.innerHTML = '';

            // Populate expenses list with fetched data
            expenses.forEach(expense => {
                const li = document.createElement('li');
                li.textContent = `${expense.date} - ${expense.amount} - ${expense.category}`;
                expensesList.appendChild(li);
            });
        })
        .catch(error => {
            // Handle error
            console.error('Error fetching expenses history:', error);
            alert('An error occurred while fetching expenses history');
        });
    }

    // Initial update of expenses history when the page loads
    updateExpensesHistory();
});
