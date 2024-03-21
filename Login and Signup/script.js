let isLoginForm = true;

function toggleForm() {
    isLoginForm = !isLoginForm;

    const formTitle = document.getElementById('formTitle');
    const submitButton = document.getElementById('submitButton');
    const toggleFormText = document.getElementById('toggleFormText');
    const confirmPasswordLabel = document.getElementById('confirmPasswordLabel');
    const confirmPassword = document.getElementById('confirmPassword');

    if (isLoginForm) {
        formTitle.innerText = 'Login';
        submitButton.innerText = 'Login';
        toggleFormText.innerHTML = 'Don\'t have an account? <a href="#" onclick="toggleForm()">Register here</a>';
        confirmPasswordLabel.style.display = 'none';
        confirmPassword.style.display = 'none';
       
    } else {
        formTitle.innerText = 'Register';
        submitButton.innerText = 'Register';
        toggleFormText.innerHTML = 'Already have an account? <a href="#" onclick="toggleForm()">Login here</a>';
        confirmPasswordLabel.style.display = 'block';
        confirmPassword.style.display = 'block';
    }
}

async function authenticateUser(phone, password, action) {
    // Update API endpoint URL
    const apiUrl = `/auth/${action}`;

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
            if (action === 'login' && response.message === 'Invalid credentials') {
                alert('Incorrect password. Please check your password and try again.');
            } else if (!isLoginForm && response.message === 'Phone number already exists') {
                alert('This phone number is already registered. Please use a different phone number or log in with your existing account.');
            } else {
                alert(`${action.charAt(0).toUpperCase() + action.slice(1)} failed. Please check your credentials.`);
            }
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
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
