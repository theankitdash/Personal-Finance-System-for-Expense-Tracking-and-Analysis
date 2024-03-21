document.addEventListener('DOMContentLoaded', function() {
    // Fetch current phone number and password
    fetch('/accountSettings') // Endpoint to get the current user's phone number
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error('Failed to fetch current credentials');
            }
        })
        .then(data => {
            document.getElementById('phoneNumber').innerText = data.phone;
            document.getElementById('newPassword').value = ''; // Clear the new password field
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while fetching current credentials');
        });

    // Add event listener to the change password button
    const changePasswordBtn = document.getElementById('changePasswordBtn');
    changePasswordBtn.addEventListener('click', function() {
        const newPassword = document.getElementById('newPassword').value;

        // Send request to change password
        fetch('/changePassword', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ newPassword })
        })
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error('Failed to change password');
            }
        })
        .then(data => {
            // Handle successful password change
            console.log('Password changed successfully:', data);
            alert('Password changed successfully');
        })
        .catch(error => {
            // Handle error
            console.error('Error changing password:', error);
            alert('An error occurred while changing password');
        });
    });
    
    // Add event listener to the logout button
    const logoutBtn = document.getElementById('logoutBtn');
    logoutBtn.addEventListener('click', function() {
        // Send request to logout
        fetch('/logout')
        .then(response => {
            if (response.redirected) {
                window.location.href = 'http://localhost:3000'; // Redirect to the login page
            } else {
                throw new Error('Failed to logout');
            }
        })
        .catch(error => {
            console.error('Error logging out:', error);
            alert('An error occurred while logging out');
        });
    });
});
