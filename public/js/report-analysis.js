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
    .then(data => {    
        console.log(data);
        if (data.success && Array.isArray(data.budgets)) {
            // Initialize a dictionary to hold budget values
            const budgetDict = {};
            
            // Populate the dictionary with data from the budgets array
            data.budgets.forEach(budget => {
                budgetDict[budget.category] = budget.amount;
            });
            
            // Populate the input fields with fetched data
            document.getElementById('clothing-budget').value = budgetDict['Clothing'] || '';
            document.getElementById('food-budget').value = budgetDict['Food'] || '';
            document.getElementById('housing-budget').value = budgetDict['Housing'] || '';
            document.getElementById('investment-budget').value = budgetDict['Investment'] || '';
            document.getElementById('medical-budget').value = budgetDict['Medical'] || '';
            document.getElementById('other-budget').value = budgetDict['Other'] || '';
            document.getElementById('transportation-budget').value = budgetDict['Transportation'] || '';
            document.getElementById('utilities-budget').value = budgetDict['Utilities'] || '';
        } else {
            console.error('Unexpected data format or failed request.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });

    document.getElementById('save-budget-btn').addEventListener('click', function() {
        const budgetDetails = {
            'Clothing': document.getElementById('clothing-budget').value,
            'Food': document.getElementById('food-budget').value,
            'Housing': document.getElementById('housing-budget').value,
            'Investment': document.getElementById('investment-budget').value,
            'Medical': document.getElementById('medical-budget').value,
            'Other': document.getElementById('other-budget').value,
            'Transportation': document.getElementById('transportation-budget').value,
            'Utilities': document.getElementById('utilities-budget').value
        };

        let alertShown = false;  // Flag to track if alert has been shown

        // Create an array to hold the promises
        const promises = [];

        // Loop through the budgetDetails object and send each category's amount to the server
        for (const [category, amount] of Object.entries(budgetDetails)) {
            if (amount) {
                // Push each fetch request promise into the promises array
                promises.push(
                    // Send an AJAX POST request to the server
                    fetch('/saveBudgetDetails', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ category, amount })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (!data.success) {
                            throw new error(`Failed to save budget for ${category}: ${data.message}`);
                        }
                    })
                    .catch(error => {
                        console.error('Error saving budget details:', error);
                    })
                );
            }
        }

        // Use Promise.all to handle all fetch requests
        Promise.all(promises).then(() => {
            if (!alertShown) {
                alert('Budget details saved successfully.');
                alertShown = true;
            }
        }).catch(error => {
            console.error('Error processing budget details:', error);
        });
    });

    const dateForm = document.getElementById('date-form');
    const fromDateInput = document.getElementById('from-date');
    const toDateInput = document.getElementById('to-date');
    const loadingIndicator = document.getElementById('loading-indicator');

    fromDateInput.addEventListener('change', handleDateChange);
    toDateInput.addEventListener('change', handleDateChange);

    dateForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        const fromDate = fromDateInput.value;
        const toDate = toDateInput.value;

        if (fromDate && toDate) {
            loadingIndicator.style.display = 'flex';
            await AnalyzeExpenses();
            loadingIndicator.style.display = 'none';
        } else {
            alert('Please select valid dates.');
        }
    });
});

let chartInstance = null;  // Ensure chartInstance is defined in the global scope

async function handleDateChange() {
    const fromDate = document.getElementById('from-date').value;
    const toDate = document.getElementById('to-date').value;

    if (fromDate && toDate) {
        const loadingIndicator = document.getElementById('loading-indicator');
        loadingIndicator.style.display = 'flex';
        await fetchExpensesData(fromDate, toDate);
        loadingIndicator.style.display = 'none';
    }
}

async function fetchExpensesData(fromDate, toDate) {
    try {
        const response = await fetch(`/expensesData?fromDate=${fromDate}&toDate=${toDate}`);
        if (!response.ok) {
            throw new Error('Failed to fetch expenses data');
        }

        const data = await response.json();

        if (data.aggregatedData.length > 0) {
            renderPieChart(data.aggregatedData);
            document.getElementById('graph').dataset.aggregatedData = JSON.stringify(data.aggregatedData);
        } else {
            alert('No data available for the selected date range.');
        }
    } catch (error) {
        console.error('Error fetching expenses data:', error);
        alert('An error occurred while fetching expenses data.');
    }
}

async function AnalyzeExpenses() {
    try {
        const aggregatedData = JSON.parse(document.getElementById('graph').dataset.aggregatedData || '[]');

        if (aggregatedData.length === 0) {
            alert('No aggregated data available for analysis.');
            return;
        }

        const response = await fetch('/analyzeFinancialData', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ aggregatedData: aggregatedData
             })
        });

        if (!response.ok) {
            throw new Error('Failed to fetch analysis results');
        }

        const result = await response.text();
        updateAnalysisResults(result);
    } catch (error) {
        console.error('Error fetching analysis results:', error);
        alert('An error occurred while fetching analysis results.');
    }
}

function aggregateData(rawData) {
    const aggregatedData = {};

    rawData.forEach(expense => {
        const { category, amount } = expense;
        if (aggregatedData[category]) {
            aggregatedData[category] += amount;  
        } else {
            aggregatedData[category] = amount;  
        }
    });

    return Object.keys(aggregatedData).map(category => ({
        category,
        total_amount: aggregatedData[category]  
    }));
}

function renderPieChart(rawData) {
    const aggregatedData = aggregateData(rawData);
    const ctx = document.getElementById('graph').getContext('2d');

    if (chartInstance) {
        chartInstance.destroy();
    }

    chartInstance = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: aggregatedData.map(expense => expense.category),
            datasets: [{
                label: 'Total Spent',
                data: aggregatedData.map(expense => expense.total_amount),
                backgroundColor: getBackgroundColor(aggregatedData.length),
                borderColor: getBorderColor(aggregatedData.length),
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                yAxes: [{
                    display: false
                }]
            }
        }
    });
}

function getBackgroundColor(length) {
    return [
        'rgba(255, 99, 132, 0.2)',
        'rgba(54, 162, 235, 0.2)',
        'rgba(255, 206, 86, 0.2)',
        'rgba(75, 192, 192, 0.2)',
        'rgba(153, 102, 255, 0.2)',
        'rgba(255, 159, 64, 0.2)',
        'rgba(199, 199, 199, 0.2)',
        'rgba(83, 102, 255, 0.2)',
        'rgba(255, 255, 99, 0.2)'
    ].slice(0, length);
}

function getBorderColor(length) {
    return [
        'rgba(255, 99, 132, 1)',
        'rgba(54, 162, 235, 1)',
        'rgba(255, 206, 86, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(153, 102, 255, 1)',
        'rgba(255, 159, 64, 1)',
        'rgba(199, 199, 199, 1)',
        'rgba(83, 102, 255, 1)',
        'rgba(255, 255, 99, 1)'
    ].slice(0, length);
}

function updateAnalysisResults(data) {
    const analysisResultsDiv = document.getElementById('analysis-results');
    analysisResultsDiv.textContent = data;
    document.getElementById('analysis-section').style.display = 'block';
}
