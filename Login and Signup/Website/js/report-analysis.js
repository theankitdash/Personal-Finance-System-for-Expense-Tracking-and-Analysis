document.addEventListener('DOMContentLoaded', function() {
    const dateForm = document.getElementById('date-form');
    const fromDateInput = document.getElementById('from-date');
    const toDateInput = document.getElementById('to-date');

    fromDateInput.addEventListener('change', handleDateChange);
    toDateInput.addEventListener('change', handleDateChange);

    dateForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        const fromDate = fromDateInput.value;
        const toDate = toDateInput.value;

        if (fromDate && toDate) {
            await fetchAndAnalyzeExpenses(fromDate, toDate);
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
        await fetchExpensesData(fromDate, toDate);
    }
}

async function fetchExpensesData(fromDate, toDate) {
    try {
        const response = await fetch(`/expensesData?fromDate=${fromDate}&toDate=${toDate}`);
        if (!response.ok) {
            throw new Error('Failed to fetch expenses data');
        }

        const data = await response.json();
        console.log('Fetched Data:', data); // Debugging line

        if (data.aggregatedData.length > 0) {
            renderPieChart(data.aggregatedData);
        } else {
            alert('No data available for the selected date range.');
        }

        // Store detailed data in a hidden field or a global variable for analysis
        document.getElementById('graph').dataset.detailedExpenses = JSON.stringify(data.detailedData);
    } catch (error) {
        console.error('Error fetching expenses data:', error);
        alert('An error occurred while fetching expenses data.');
    }
}

async function fetchAndAnalyzeExpenses(fromDate, toDate) {
    const loadingIndicator = document.getElementById('loading-indicator');
    let loadingTimeout;

    try {
        const expensesData = JSON.parse(document.getElementById('graph').dataset.detailedExpenses || '[]');
        
        if (expensesData.length === 0) {
            alert('No detailed data available for analysis.');
            return;
        }

        // Show loading indicator if analysis takes more than 2 seconds
        loadingTimeout = setTimeout(() => {
            loadingIndicator.style.display = 'flex';
        }, 2000);

        const response = await fetch('/analyzeFinancialData', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ data: expensesData })
        });

        clearTimeout(loadingTimeout);
        loadingIndicator.style.display = 'none';

        if (!response.ok) {
            throw new Error('Failed to fetch analysis results');
        }

        const result = await response.text();
        updateAnalysisResults(result);
    } catch (error) {
        clearTimeout(loadingTimeout);
        loadingIndicator.style.display = 'none';
        console.error('Error fetching analysis results:', error);
        alert('An error occurred while fetching analysis results.');
    }
}

function renderPieChart(aggregatedData) {
    const ctx = document.getElementById('graph').getContext('2d');
    if (chartInstance) {
        chartInstance.destroy();
    }

    console.log('Aggregated Data for Chart:', aggregatedData); // Debugging line

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
        'rgba(255, 159, 64, 0.2)'
    ].slice(0, length);
}

function getBorderColor(length) {
    return [
        'rgba(255, 99, 132, 1)',
        'rgba(54, 162, 235, 1)',
        'rgba(255, 206, 86, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(153, 102, 255, 1)',
        'rgba(255, 159, 64, 1)'
    ].slice(0, length);
}

function updateAnalysisResults(data) {
    const analysisResultsDiv = document.getElementById('analysis-results');
    analysisResultsDiv.innerHTML = `<p>${data}</p>`;
    document.getElementById('analysis-section').style.display = 'block';
}
