// Shared utility functions for frontend

/**
 * Display validation errors from backend in a user-friendly format
 */
function displayValidationErrors(errorData) {
    if (errorData.errors && Array.isArray(errorData.errors)) {
        const errorMessages = errorData.errors
            .map(err => `• ${err.param}: ${err.msg}`)
            .join('\n');
        alert(`Please fix the following:\n\n${errorMessages}`);
    } else {
        alert(errorData.message || 'An error occurred');
    }
}

/**
 * Handle fetch errors with proper error display
 */
async function handleFetchError(response, defaultMessage) {
    if (!response.ok) {
        try {
            const errorData = await response.json();
            displayValidationErrors(errorData);
        } catch {
            alert(defaultMessage || 'An error occurred');
        }
        return true;
    }
    return false;
}

/**
 * Format date as YYYY-MM-DD
 */
function formatDate(dateString) {
    if (!dateString) return '';
    const date = new Date(dateString);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

/**
 * Show loading state on a button
 */
function showLoading(buttonElement) {
    if (!buttonElement) return;
    buttonElement.disabled = true;
    buttonElement.dataset.originalText = buttonElement.textContent;
    buttonElement.textContent = 'Loading...';
}

/**
 * Hide loading state on a button
 */
function hideLoading(buttonElement) {
    if (!buttonElement) return;
    buttonElement.disabled = false;
    if (buttonElement.dataset.originalText) {
        buttonElement.textContent = buttonElement.dataset.originalText;
        delete buttonElement.dataset.originalText;
    }
}

/**
 * Generic error handler for console logging and user notification
 */
function handleError(error, context) {
    console.error(`Error in ${context}:`, error);
    alert(`An error occurred: ${error.message || 'Please try again later'}`);
}
