document.addEventListener('DOMContentLoaded', function() {

    // Function to format date as YYYY-MM-DD
    function formatDate(date) {
        if (!date) return ''; // Return empty string if date is not provided
        const d = new Date(date);
        const year = d.getFullYear();
        const month = String(d.getMonth() + 1).padStart(2, '0');
        const day = String(d.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    } 
    
    // Function to handle fetch errors
    function handleFetchError(error, message) {
        console.error(message, error);
        alert(`An error occurred: ${message}`);
    }

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
        handleFetchError(error, 'Error fetching user details');
    });

    // Add event listener to the form submission
    const saveExpenseBtn = document.getElementById('saveExpenseBtn');
    saveExpenseBtn.addEventListener('click', function(event) {
        event.preventDefault(); // Prevent default form submission behavior

        const date = document.getElementById('expenseDate').value;
        const amount = document.getElementById('expenseAmount').value;
        const description = document.getElementById('expenseDescription').value;
        const category = document.getElementById('expenseCategory').value;

        // Validate input values (optional)
        if (!date || !amount || !description || !category) {
            alert('Please fill out all fields');
            return;
        }

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
            handleFetchError(error, 'Error saving expense');
        });
    });

    // Function to update expenses history
    function updateExpensesHistory() {
        // Fetch expenses history based on selected category
        const selectedCategory = document.getElementById('selectCategory').value;
        const url = `/expensesHistory?category=${selectedCategory}`;

        fetch(url)
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error('Failed to fetch expenses history');
            }
        })
        .then(expenses => {
            // Clear existing expenses list
            const expensesTableBody = document.getElementById('expensesTableBody');
            expensesTableBody.innerHTML = '';

            // Populate expenses table with fetched data
            expenses.forEach(expense => {
                const formattedDate = formatDate(expense.date);
                const row = createExpenseRow(expense, formattedDate);

                // Append row to the table body
                expensesTableBody.appendChild(row);

            });
        })
        .catch(error => {
            // Handle error
            handleFetchError(error, 'Error fetching expenses history');
        });
    }

    // Create table row for each expense
    function createExpenseRow(expense, formattedDate) {
        const row = document.createElement('tr');

        // Create cells for each attribute
        const dateCell = document.createElement('td');
        dateCell.textContent = formattedDate;
        row.appendChild(dateCell);

        const amountCell = document.createElement('td');
        amountCell.textContent = expense.amount;
        row.appendChild(amountCell);

        const descriptionCell = document.createElement('td');
        descriptionCell.textContent = expense.description;
        row.appendChild(descriptionCell);

        //Action button
        const actionCell = document.createElement('td');
        const removeButton = document.createElement('button');
        removeButton.textContent = 'Remove';
        removeButton.addEventListener('click', () => removeExpense(expense.id));
        actionCell.appendChild(removeButton);
        row.appendChild(actionCell);

        return row;
    }

    // Function to remove expense
    function removeExpense(id) {
        fetch(`/expenses/${id}`, {
            method: 'DELETE'
        })
        .then(response => {
            if (response.ok) {
                // Refresh expenses history after removing expense
                updateExpensesHistory();
                alert('Expense removed successfully');
            } else {
                throw new Error('Failed to remove expense');
            }
        })
        .catch(error => {
            // Handle error
            handleFetchError(error, 'Error removing expense');
        });
    }

    // Add event listener to the category dropdown menu
    const selectCategory = document.getElementById('selectCategory');
    selectCategory.addEventListener('change', updateExpensesHistory);

    // Initial update of expenses history when the page loads
    updateExpensesHistory();

});
