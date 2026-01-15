document.addEventListener('DOMContentLoaded', function () {

    const dateForm = document.getElementById('date-form');
    const fromDateInput = document.getElementById('from-date');
    const toDateInput = document.getElementById('to-date');
    const loadingIndicator = document.getElementById('loading-indicator');
    const submitBtn = document.getElementById('submit-btn');

    dateForm.addEventListener('submit', async function (event) {
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
        const response = await fetchWithCredentials('/analyzeFinancialData', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            credentials: 'include',
            body: JSON.stringify({ fromDate, toDate })
        });

        if (!response.ok) {
            throw new Error('Failed to generate Excel Report');
        }

        // Convert response to blob
        const blob = await response.blob();

        // Create downloadable link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');

        a.href = url;
        a.download = `analysis_${fromDate}_to_${toDate}.xlsx`;
        a.style.display = 'none';

        document.body.appendChild(a);
        a.click();

        window.URL.revokeObjectURL(url);
        a.remove();

    } catch (error) {
        console.error('Error downloading Excel:', error);
        alert('An error occurred while downloading the Excel file.');
    }
}
