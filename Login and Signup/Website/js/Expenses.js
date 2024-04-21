function filterExpenses() {
    var selectElement = document.getElementById("selectCategory");
    var selectedCategory = selectElement.value;
    var expenses = getExpenses(); // Assume getExpenses() retrieves expenses data
    
    var filteredExpenses = expenses.filter(function(expense) {
        return selectedCategory === "all" || expense.category === selectedCategory;
    });

    displayExpenses(filteredExpenses);
}

function displayExpenses(expenses) {
    var expensesList = document.getElementById("expensesList");
    expensesList.innerHTML = "";

    expenses.forEach(function(expense) {
        var li = document.createElement("li");
        li.textContent = expense.date + " - " + expense.amount + " - " + expense.description;
        expensesList.appendChild(li);
    });
}

// Function to fetch expenses data, replace with actual implementation
function getExpenses() {
    // Replace this with actual data retrieval logic
    return [
        { date: "2024-04-21", amount: 50, description: "Groceries", category: "Food" },
        { date: "2024-04-20", amount: 100, description: "Gas", category: "Transportation" },
        // Add more expenses data here
    ];
}

// Initially display all expenses
filterExpenses();
