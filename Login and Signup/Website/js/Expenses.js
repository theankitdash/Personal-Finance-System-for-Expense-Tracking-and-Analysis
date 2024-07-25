document.addEventListener('DOMContentLoaded', function() {
    // Function to format date as YYYY-MM-DD
    function formatDate(dateString) {
        if (!dateString) return ''; // Return empty string if date is not provided
        const date = new Date(dateString);
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
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
            console.log('Expense saved successfully:', data);
            alert('Expense saved successfully');
            fetchAllExpenses(); // Refresh expenses history after saving
        })
        .catch(error => {
            handleFetchError(error, 'Error saving expense');
        });
    });

    // Fetch all expenses history
    function fetchAllExpenses() {
        fetch('/expensesHistory?filter=all&value=')
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error('Failed to fetch expenses history');
            }
        })
        .then(expenses => {
            const expensesTableBody = document.getElementById('expensesTableBody');
            expensesTableBody.innerHTML = '';

            expenses.forEach(expense => {
                const formattedDate = formatDate(expense.date);
                const row = createExpenseRow(expense, formattedDate);
                expensesTableBody.appendChild(row);
            });
        })
        .catch(error => {
            handleFetchError(error, 'Error fetching expenses history');
        });
    }

    // Fetch unique options for a given filter type
    function fetchUniqueOptions(filterType) {
        return fetch(`/uniqueOptions?filter=${filterType}`)
            .then(response => {
                if (response.ok) {
                    return response.json();
                } else {
                    throw new Error('Failed to fetch unique options');
                }
            });
    }

    // Update filter options
    function updateFilterOptions() {
        const filterType = document.getElementById('selectFilter').value;
        const selectOption = document.getElementById('selectOption');
        const dateInput = document.getElementById('dateInput');

        selectOption.innerHTML = '<option value="">Select an option</option>'; // Default option
        selectOption.style.display = 'none'; // Hide by default
        dateInput.style.display = 'none'; // Hide by default

        if (filterType === 'all') {
            fetchAllExpenses();
            return;
        }

        if (filterType === 'date') {
            dateInput.style.display = 'block';
            return;
        }

        selectOption.style.display = 'block';
        fetchUniqueOptions(filterType)
        .then(uniqueOptions => {
            uniqueOptions.forEach(option => {
                const opt = document.createElement('option');
                opt.value = option;
                opt.textContent = option;
                selectOption.appendChild(opt);
            });
        })
        .catch(error => {
            handleFetchError(error, 'Error fetching unique options');
        });
    }

    // Update expenses history based on selected filter
    function updateExpensesHistory() {
        const filterType = document.getElementById('selectFilter').value;
        const selectOption = document.getElementById('selectOption').value;
        const dateInput = document.getElementById('dateInput').value;

        const params = new URLSearchParams();
        params.append('filter', filterType);
        params.append('value', filterType === 'date' ? dateInput : selectOption);

        const url = `/expensesHistory?${params.toString()}`;

        fetch(url)
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error('Failed to fetch expenses history');
            }
        })
        .then(expenses => {
            const expensesTableBody = document.getElementById('expensesTableBody');
            expensesTableBody.innerHTML = '';

            if (expenses.length === 0) {
                const noRecordsMessage = document.createElement('tr');
                noRecordsMessage.innerHTML = '<td colspan="5">No records found for this selection</td>';
                expensesTableBody.appendChild(noRecordsMessage);
                return;
            }

            expenses.forEach(expense => {
                const formattedDate = formatDate(expense.date);
                const row = createExpenseRow(expense, formattedDate);
                expensesTableBody.appendChild(row);
            });
        })
        .catch(error => {
            handleFetchError(error, 'Error fetching expenses history');
        });
    }

    // Create table row for each expense
    function createExpenseRow(expense, formattedDate) {
        const row = document.createElement('tr');

        const dateCell = document.createElement('td');
        dateCell.textContent = formattedDate;
        row.appendChild(dateCell);

        const amountCell = document.createElement('td');
        amountCell.textContent = expense.amount;
        row.appendChild(amountCell);

        const descriptionCell = document.createElement('td');
        descriptionCell.textContent = expense.description;
        row.appendChild(descriptionCell);

        const categoryCell = document.createElement('td');
        categoryCell.textContent = expense.category;
        row.appendChild(categoryCell);

        const actionCell = document.createElement('td');
        const removeButton = document.createElement('button');
        removeButton.textContent = 'Remove';
        removeButton.setAttribute('aria-label', 'Remove expense');
        removeButton.addEventListener('click', () => removeExpense(expense.id));
        actionCell.appendChild(removeButton);
        row.appendChild(actionCell);

        return row;
    }

    // Remove expense
    function removeExpense(id) {
        fetch(`/expenses/${id}`, {
            method: 'DELETE'
        })
        .then(response => {
            if (response.ok) {
                updateExpensesHistory();
                alert('Expense removed successfully');
            } else {
                throw new Error('Failed to remove expense');
            }
        })
        .catch(error => {
            handleFetchError(error, 'Error removing expense');
        });
    }

    // Event listeners
    document.getElementById('selectFilter').addEventListener('change', updateFilterOptions);
    document.getElementById('selectOption').addEventListener('change', updateExpensesHistory);
    document.getElementById('dateInput').addEventListener('change', updateExpensesHistory);

    // Initial setup
    updateFilterOptions();
});
