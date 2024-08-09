document.addEventListener('DOMContentLoaded', () => {
    let isLoginForm = true;

    // DOM elements
    const elements = {
        formTitle: document.getElementById('formTitle'),
        submitButton: document.getElementById('submitButton'),
        toggleFormText: document.getElementById('toggleFormText'),
        confirmPasswordLabel: document.getElementById('confirmPasswordLabel'),
        confirmPassword: document.getElementById('confirmPassword'),
        forgotPasswordText: document.getElementById('forgotPasswordText').querySelector('a'),
        forgotPasswordModal: document.getElementById('forgotPasswordModal'),
        closeModalButton: document.getElementById('closeModalButton'),
        verifyUserForm: document.getElementById('verifyUserForm'),
        resetPasswordForm: document.getElementById('resetPasswordForm'),
        step1: document.getElementById('step1'),
        step2: document.getElementById('step2')
    };

    // Utility functions
    const displayElement = (element, display = 'block') => element.style.display = display;
    const hideElement = (element) => element.style.display = 'none';

    const toggleForm = (event) => {
        event.preventDefault();
        isLoginForm = !isLoginForm;

        elements.formTitle.innerText = isLoginForm ? 'Login' : 'Register';
        elements.submitButton.innerText = isLoginForm ? 'Login' : 'Register';
        elements.toggleFormText.innerHTML = isLoginForm ?
            'Don\'t have an account? <a href="#">Register here</a>' :
            'Already have an account? <a href="#">Login here</a>';

        if (isLoginForm) {
            hideElement(elements.confirmPasswordLabel);
            hideElement(elements.confirmPassword);
        } else {
            displayElement(elements.confirmPasswordLabel);
            displayElement(elements.confirmPassword);
        }
    };

    const isStrongPassword = (password) => /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/.test(password);
    const isPhoneValid = (phone) => /^[0-9]{10}$/.test(phone);

    const authenticateUser = async (phone, password, action) => {
        const apiUrl = `/auth/${action}`;

        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json' 
                },
                body: JSON.stringify({ phone, password })
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.message || 'An error occurred');
            }

            return result;
        } catch (error) {
            console.error('Error:', error);
            throw new Error(error.message || 'An error occurred during authentication');
        }
    };

    const submitForm = async (event) => {
        event.preventDefault();

        const phone = document.getElementById('phone').value;
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirmPassword').value;

        if (!isPhoneValid(phone)) {
            alert('Please enter a valid phone number.');
            return;
        }

        if (!isStrongPassword(password)) {
            alert('Password must contain at least one uppercase letter, one lowercase letter, one digit, one special character, and be at least 8 characters long.');
            return;
        }

        if (!isLoginForm && password !== confirmPassword) {
            alert('Passwords do not match. Please check and try again.');
            return;
        }

        const action = isLoginForm ? 'login' : 'register';

        try {
            const response = await authenticateUser(phone, password, action);

            if (response.success) {
                alert(`${action.charAt(0).toUpperCase() + action.slice(1)} successful.`);
                window.location.href = isLoginForm ? 'home.html' : 'account-settings.html';
            } 
        } catch (error) {
            alert(error.message);
        }
    };

    const openForgotPasswordModal = (event) => {
        event.preventDefault();
        displayElement(elements.forgotPasswordModal);
    };

    const closeForgotPasswordModal = () => hideElement(elements.forgotPasswordModal);

    const verifyUser = async (event) => {
        event.preventDefault();

        const phone = document.getElementById('resetPhone').value;
        const dob = document.getElementById('dob').value;

        if (!isPhoneValid(phone)) {
            alert('Please enter a valid phone number.');
            return;
        }

        if (!dob) {
            alert('Please enter your date of birth.');
            return;
        }

        try {
            const response = await fetch('/auth/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ phone, dob })
            });

            const result = await response.json();
            if (result.success) {
                hideElement(elements.step1);
                displayElement(elements.step2);
            } else {
                alert('Verification failed. Please check your details and try again.');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
        }
    };

    const resetPassword = async (event) => {
        event.preventDefault();

        const phone = document.getElementById('resetPhone').value;
        const newPassword = document.getElementById('newPassword').value;

        if (!isStrongPassword(newPassword)) {
            alert('New password must contain at least one uppercase letter, one lowercase letter, one digit, one special character, and be at least 8 characters long.');
            return;
        }

        try {
            const response = await fetch('/auth/resetPassword', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ phone, newPassword })
            });

            const result = await response.json();
            if (result.success) {
                alert('Password reset successful.');
                closeForgotPasswordModal();
            } else {
                alert('Password reset failed. Please try again.');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
        }
    };

    // Event listeners
    elements.toggleFormText.addEventListener('click', toggleForm);
    elements.forgotPasswordText.addEventListener('click', openForgotPasswordModal);
    elements.closeModalButton.addEventListener('click', closeForgotPasswordModal);
    elements.verifyUserForm.addEventListener('submit', verifyUser);
    elements.resetPasswordForm.addEventListener('submit', resetPassword);
    elements.submitButton.addEventListener('click', submitForm);
});
