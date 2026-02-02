// Shared utility functions for frontend

/**
 * Wrapper for fetch that automatically includes credentials (cookies)
 * This ensures auth tokens and session cookies are sent with every request
 */
function fetchWithCredentials(url, options = {}) {
    return fetch(url, {
        ...options,
        credentials: 'include'  // Include cookies in all requests
    });
}

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
 * Works with both Response objects and Error objects
 */
async function handleFetchError(responseOrError, defaultMessage) {
    // Handle Error objects (from catch blocks)
    if (responseOrError instanceof Error) {
        console.error(defaultMessage, responseOrError);
        alert(`An error occurred: ${defaultMessage}`);
        return true;
    }

    // Handle Response objects
    if (!responseOrError.ok) {
        try {
            const errorData = await responseOrError.json();
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
 * Sanitize HTML to prevent XSS attacks
 */
function sanitizeHTML(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/**
 * Check if running in production environment
 */
function isProduction() {
    return window.location.hostname === 'the-financial-tracker.azurewebsites.net';
}

/**
 * Get the base URL based on environment
 */
function getBaseUrl() {
    return isProduction()
        ? 'https://the-financial-tracker.azurewebsites.net'
        : 'http://localhost:3000';
}

/**
 * Development-only logging (suppressed in production)
 */
function devLog(...args) {
    if (!isProduction()) {
        console.log(...args);
    }
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

