document.addEventListener('DOMContentLoaded', function() {

    // Fetch all expenses history
    async function fetchAllExpenses() {
        try {
            const response = await fetch('/expensesHistory?filter=all&value=');
            if (!response.ok) {
                throw new Error('Failed to fetch expenses history');
            }
            const expenses = await response.json();
            const expensesTableBody = document.getElementById('expensesTableBody');
            expensesTableBody.innerHTML = '';

            expenses.forEach(expense => {
                const formattedDate = formatDate(expense.date);
                const row = createExpenseRow(expense, formattedDate);
                expensesTableBody.appendChild(row);
            });
        } catch (error) {
            handleFetchError(error, 'Error fetching expenses history');
        }
    }

    // Fetch unique options for a given filter type
    async function fetchUniqueOptions(filterType) {
        try {
            const response = await fetch(`/uniqueOptions?filter=${filterType}`);
            if (!response.ok) {
                throw new Error('Failed to fetch unique options');
            }
            return await response.json();
        } catch (error) {
            handleFetchError(error, 'Error fetching unique options');
        }
    }
    
    // Function to fetch categories
    async function fetchCategories() {
        try {
            const response = await fetch('/categories');
            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    const categories = data.categories;
                    const categorySelect = document.getElementById('expenseCategory');

                    // Clear any existing options
                    categorySelect.innerHTML = '<option value="" disabled selected>Select category</option>';

                    // Populate categories in the dropdown
                    categories.forEach(category => {
                        const option = document.createElement('option');
                        option.value = category;
                        option.textContent = category;
                        categorySelect.appendChild(option);
                    });
                } else {
                    throw new Error('Failed to fetch categories');
                }
            } else {
                throw new Error('Failed to fetch categories from the server');
            }
        } catch (error) {
            console.error('Error fetching categories:', error);
            alert('Error fetching categories. Please try again later.');
        }
    }
    
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

    // Add event listener to the form submission
    const saveExpenseBtn = document.getElementById('saveExpenseBtn');
    
    function handleSaveOrUpdateButtonClick(event) {
        event.preventDefault(); // Prevent default form submission behavior

        const date = document.getElementById('expenseDate').value;
        const amount = document.getElementById('expenseAmount').value;
        const description = document.getElementById('expenseDescription').value;
        const category = document.getElementById('expenseCategory').value;
        const expenseId = saveExpenseBtn.getAttribute('data-expense-id'); // Retrieve the expense ID

        // Validate input values
        if (!date || !amount || !description || !category) {
            alert('Please fill out all fields');
            return;
        }

        const url = expenseId ? `/updateExpenses/${expenseId}` : '/saveExpenses';
        const method = expenseId ? 'PUT' : 'POST';
        const expenseData = { date, amount, description, category };

        console.log(`Sending ${method} request to ${url} with data:`, expenseData);

        // Send request to save or update expense
        fetch(url, {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(expenseData)
        })
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                return response.json().then(errorData => {
                    throw new Error(`Failed to ${expenseId ? 'update' : 'save'} expense: ${errorData.message}`);
                });
            }
        })
        .then(data => {
            console.log(`${expenseId ? 'Expense updated' : 'Expense saved'} successfully:`, data);
            alert(`${expenseId ? 'Expense updated' : 'Expense saved'} successfully`);
            fetchAllExpenses(); // Refresh expenses history after saving/updating

            // Reset form and button state
            resetForm();
        })
        .catch(error => {
            handleFetchError(error, `Error ${expenseId ? 'updating' : 'saving'} expense`);
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
    async function updateExpensesHistory() {
        const filterType = document.getElementById('selectFilter').value;
        const selectOption = document.getElementById('selectOption').value;
        const dateInput = document.getElementById('dateInput').value;

        const params = new URLSearchParams();
        params.append('filter', filterType);
        params.append('value', filterType === 'date' ? dateInput : selectOption);

        const url = `/expensesHistory?${params.toString()}`;

        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error('Failed to fetch expenses history');
            }
            const expenses = await response.json();
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
        } catch (error) {
            handleFetchError(error, 'Error fetching expenses history');
        }
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

        //Action button
        const actionCell = document.createElement('td');
        
        const removeButton = document.createElement('button');
        removeButton.textContent = 'Remove';
        removeButton.setAttribute('aria-label', 'Remove expense');
        removeButton.addEventListener('click', () => removeExpense(expense.id));
        actionCell.appendChild(removeButton);

        const modifyButton = document.createElement('button');
        modifyButton.textContent = 'Modify';
        modifyButton.setAttribute('aria-label', 'Modify expense');
        modifyButton.addEventListener('click', () => modifyExpense(expense));
        actionCell.appendChild(modifyButton);

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

    // Modify expense
    function modifyExpense(expense) {
        // Populate the form fields with the expense details
        document.getElementById('expenseDate').value = formatDate(expense.date);
        document.getElementById('expenseAmount').value = expense.amount;
        document.getElementById('expenseDescription').value = expense.description;
        document.getElementById('expenseCategory').value = expense.category;

        // Set the expense ID on the save button
        saveExpenseBtn.setAttribute('data-expense-id', expense.id);

        // Scroll to the form (optional)
        document.getElementById('expenseForm').scrollIntoView();

        // Update save button to indicate modification
        saveExpenseBtn.textContent = 'Update Expense';

        // Remove any existing event listener on the save button
        saveExpenseBtn.removeEventListener('click', handleSaveOrUpdateButtonClick);

        // Add new event listener to handle update
        saveExpenseBtn.addEventListener('click', handleSaveOrUpdateButtonClick);
    }

    // Reset form and button state
    function resetForm() {
        document.getElementById('expenseDate').value = '';
        document.getElementById('expenseAmount').value = '';
        document.getElementById('expenseDescription').value = '';
        document.getElementById('expenseCategory').value = '';

        saveExpenseBtn.removeAttribute('data-expense-id'); // Remove the data-expense-id attribute
        saveExpenseBtn.textContent = 'Save Expense';
        saveExpenseBtn.removeEventListener('click', handleSaveOrUpdateButtonClick);
        saveExpenseBtn.addEventListener('click', handleSaveOrUpdateButtonClick);

        // Reset filter and dropdown
        document.getElementById('selectFilter').value = 'all'; // Set filter to 'all'
        updateFilterOptions(); // Update filter options to reset the dropdown
    }

    // Event listeners
    document.getElementById('selectFilter').addEventListener('change', updateFilterOptions);
    document.getElementById('selectOption').addEventListener('change', updateExpensesHistory);
    document.getElementById('dateInput').addEventListener('change', updateExpensesHistory);

    // Initial setup
    updateFilterOptions();
    fetchCategories();
    saveExpenseBtn.addEventListener('click', handleSaveOrUpdateButtonClick); // Ensure the event listener is added here
    fetchAllExpenses(); // Fetch all expenses on initial setup
});
