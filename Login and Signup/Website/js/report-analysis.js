document.addEventListener('DOMContentLoaded', function() {
    
    const dateForm = document.getElementById('date-form');
    const analyzeBtn = document.getElementById('analyze-btn');

    dateForm.addEventListener('submit', function(event) {
        event.preventDefault();
        fetchExpensesData();
    });

    analyzeBtn.addEventListener('click', function() {
        const expensesData = JSON.parse(document.getElementById('graph').dataset.expenses);
        fetchAndUpdateAnalysisResults(expensesData);
    });
});

let chartInstance = null;

async function fetchExpensesData() {
    const fromDate = document.getElementById('from-date').value;
    const toDate = document.getElementById('to-date').value;

    try {
        const response = await fetch(`/expensesData?fromDate=${fromDate}&toDate=${toDate}`);
        if (!response.ok) {
            throw new Error('Failed to fetch expenses data');
        }

        const data = await response.json();
        renderPieChart(data);
        document.getElementById('analyze-btn').style.display = 'inline-block';
        document.getElementById('graph').dataset.expenses = JSON.stringify(data);
    } catch (error) {
        console.error('Error fetching expenses data:', error);
        alert('An error occurred while fetching expenses data');
    }
}

function renderPieChart(expenses) {
    const ctx = document.getElementById('graph').getContext('2d');
    if (chartInstance) {
        chartInstance.destroy();
    }

    chartInstance = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: expenses.map(expense => expense.category),
            datasets: [{
                label: 'Total Spent',
                data: expenses.map(expense => expense.total_amount),
                backgroundColor: getBackgroundColor(expenses.length),
                borderColor: getBorderColor(expenses.length),
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

async function fetchAndUpdateAnalysisResults(expensesData) {
    try {
        const response = await fetch('/analyzeFinancialData', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ data: expensesData })
        });

        if (!response.ok) {
            throw new Error('Failed to fetch analysis results');
        }

        const result = await response.text();
        updateAnalysisResults(result);
    } catch (error) {
        console.error('Error fetching analysis results:', error);
        alert('An error occurred while fetching analysis results');
    }
}

function updateAnalysisResults(data) {
    const analysisResultsDiv = document.getElementById('analysis-results');
    analysisResultsDiv.innerHTML = `<p>${data}</p>`;
    document.getElementById('analysis-section').style.display = 'block';
}

  