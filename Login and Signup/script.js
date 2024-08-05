let isLoginForm = true;

document.addEventListener('DOMContentLoaded', function() {
    const formTitle = document.getElementById('formTitle');
    const submitButton = document.getElementById('submitButton');
    const toggleFormText = document.getElementById('toggleFormText');
    const confirmPasswordLabel = document.getElementById('confirmPasswordLabel');
    const confirmPassword = document.getElementById('confirmPassword');
    const forgotPasswordText = document.getElementById('forgotPasswordText');
    const forgotPasswordModal = document.getElementById('forgotPasswordModal');
    const closeModalButton = document.getElementById('closeModalButton');
    const resetPasswordForm = document.getElementById('resetPasswordForm');
    const step1 = document.getElementById('step1');
    const step2 = document.getElementById('step2');

    // Add event listener for toggleFormText link clicks
    toggleFormText.addEventListener('click', toggleForm);
    forgotPasswordText.querySelector('a').addEventListener('click', openForgotPasswordModal);
    closeModalButton.addEventListener('click', closeForgotPasswordModal);

    // Add event listener for resetPasswordForm submit
    verifyUserForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const phone = document.getElementById('resetPhone').value;
        const dob = document.getElementById('dob').value;

        // Validate phone number and date of birth
        if (!isPhoneValid(phone)) {
            alert('Please enter a valid phone number.');
            return;
        }
        if (!dob) {
            alert('Please enter your date of birth.');
            return;
        }

        // Request to verify phone and DOB
        try {
            const response = await fetch('/auth/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ phone, dob }),
            });

            const result = await response.json();
            if (result.success) {
                step1.style.display = 'none';
                step2.style.display = 'block';
            } else {
                alert('Verification failed. Please check your details and try again.');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
        }
    });

    resetPasswordForm.addEventListener('submit', async (event) => {
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
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ phone, newPassword }),
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
    });

    function toggleForm(event) {
        event.preventDefault();
        isLoginForm = !isLoginForm;

        if (isLoginForm) {
            formTitle.innerText = 'Login';
            submitButton.innerText = 'Login';
            toggleFormText.innerHTML = 'Don\'t have an account? <a href="#">Register here</a>';
            confirmPasswordLabel.style.display = 'none';
            confirmPassword.style.display = 'none';
        } else {
            formTitle.innerText = 'Register';
            submitButton.innerText = 'Register';
            toggleFormText.innerHTML = 'Already have an account? <a href="#">Login here</a>';
            confirmPasswordLabel.style.display = 'block';
            confirmPassword.style.display = 'block';
        }
    }

    function openForgotPasswordModal(event) {
        event.preventDefault();
        forgotPasswordModal.style.display = 'block';
    }

    function closeForgotPasswordModal() {
        forgotPasswordModal.style.display = 'none';
    }
});

async function authenticateUser(phone, password, action) {
    // Update API endpoint URL
    const apiUrl = `/auth/${action}`;

    try {
        // Make request to server
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({phone, password}),
        });

        // Handle response
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        return response.json();
    } catch (error) {
        console.error('Error:', error);
        throw new Error('An error occurred during authentication');
    }    
}

async function submitForm(event) {
    event.preventDefault();

    const phone = document.getElementById('phone').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirmPassword').value;

    if (!isLoginForm && !isPhoneValid(phone)) {
        alert('Please enter a valid phone number.');
        return;
    }

    // Password strength validation
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
        // Call server-side authentication function
        const response = await authenticateUser(phone, password, action);

        // Handle server response
        if (response.success) {
            if (isLoginForm) {
                // Redirect to another page upon successful login
                window.location.href = 'Website/home.html';
            } else {
                // Alert "successful" upon successful registration
                alert('Registration successful.');
                window.location.href = 'Website/account-settings.html';
            }
        } else {
            handleAuthError(response, action);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    }
}

function handleAuthError(response, action) {
    if (action === 'login' && response.message === 'Invalid credentials') {
        alert('Incorrect password. Please check your password and try again.');
    } else if (!isLoginForm && response.message === 'Phone number already exists') {
        alert('This phone number is already registered. Please use a different phone number or log in with your existing account.');
    } else {
        alert(`${action.charAt(0).toUpperCase() + action.slice(1)} failed. Please check your credentials.`);
    }
}

function isStrongPassword(password) {
    // Regular expression to enforce strong password criteria
    const strongPasswordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;
    return strongPasswordRegex.test(password);
}

function isPhoneValid(phone) {
    const phoneRegex = /^[0-9]{10}$/;
    return phoneRegex.test(phone);
}
