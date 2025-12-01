document.addEventListener('DOMContentLoaded', function() {

    const dateForm = document.getElementById('date-form');
    const fromDateInput = document.getElementById('from-date');
    const toDateInput = document.getElementById('to-date');
    const loadingIndicator = document.getElementById('loading-indicator');
    const submitBtn = document.getElementById('submit-btn');

    dateForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        const fromDate = fromDateInput.value;
        const toDate = toDateInput.value;

        // Validate dates
        if (!fromDate || !toDate) {
            alert('Please select both From and To dates.');
            return;
        }

        // Disable submit button to prevent double submit
        submitBtn.disabled = true;
        loadingIndicator.style.display = 'flex';

        try {
            await analyzeExpenses(fromDate, toDate);
        } catch (err) {
            console.error(err);
            alert('An error occurred while analyzing expenses.');
        } finally {
            loadingIndicator.style.display = 'none';
            submitBtn.disabled = false;
        }
    });
});

async function analyzeExpenses(fromDate, toDate) {
    try {
        const response = await fetch('/analyzeFinancialData', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            credentials: 'include',
            body: JSON.stringify({ fromDate, toDate })
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

function updateAnalysisResults(data) {
    const analysisResultsDiv = document.getElementById('analysis-results');
    analysisResultsDiv.textContent = data;
    document.getElementById('analysis-container').style.display = 'block';
}