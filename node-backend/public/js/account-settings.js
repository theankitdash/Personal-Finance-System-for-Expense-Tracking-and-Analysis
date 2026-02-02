document.addEventListener('DOMContentLoaded', function () {

    fetchWithCredentials('/personalDetails') // Endpoint to get the current user's Details
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

            // Display personal details
            document.getElementById('name').value = data.personalDetails.name || '';
            document.getElementById('gender').value = data.personalDetails.gender || '';
            document.getElementById('dateOfBirth').value = formatDate(data.personalDetails.dateOfBirth) || '';
        })
        .catch(error => {
            console.error('Error:', error);
            alert("Personal details not found. Please add to continue further!");
        });

    // formatDate is now provided by utils.js

    // Add event listener to the change password button
    const changePasswordBtn = document.getElementById('changePasswordBtn');
    changePasswordBtn.addEventListener('click', function () {
        const newPassword = document.getElementById('newPassword').value;

        // Send request to change password
        fetchWithCredentials('/changePassword', {
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
                devLog('Password changed successfully:', data);
                alert('Password changed successfully');
            })
            .catch(error => {
                // Handle error
                devLog('Error changing password:', error);
                alert('An error occurred while changing password');
            });
    });

    // Add event listener to the form submission
    const saveProfileBtn = document.getElementById('saveProfileBtn');
    saveProfileBtn.addEventListener('click', function () {
        const name = document.getElementById('name').value;
        const gender = document.getElementById('gender').value;
        const dateOfBirth = document.getElementById('dateOfBirth').value;

        // Send request to save personal details
        fetchWithCredentials('/savePersonalDetails', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name, gender, dateOfBirth })
        })
            .then(response => {
                if (response.ok) {
                    return response.json();
                } else {
                    throw new Error('Failed to save personal details');
                }
            })
            .then(data => {
                // Handle successful saving of personal details
                devLog('Personal details saved successfully:', data);
                alert('Personal details saved successfully');
            })
            .catch(error => {
                // Handle error
                devLog('Error saving personal details:', error);
                alert('An error occurred while saving personal details');
            });
    });

    // Use getBaseUrl helper from utils.js
    const loginUrl = getBaseUrl();

    const logoutBtn = document.getElementById('logoutBtn');
    logoutBtn.addEventListener('click', function () {
        // Send request to logout
        fetchWithCredentials('/auth/logout')
            .then(response => {
                if (response.redirected) {
                    // Redirect to the login page
                    window.location.href = response.url;
                } else if (response.ok) {
                    // Redirect to the login page
                    window.location.href = loginUrl;
                } else {
                    throw new Error('Failed to logout');
                }
            })
            .catch(error => {
                devLog('Error logging out:', error);
                alert('An error occurred while logging out');
            });
    });
});
