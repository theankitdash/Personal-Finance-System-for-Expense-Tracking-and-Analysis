const express = require('express');
const path = require('path');
const mysql = require('mysql');
const bcrypt = require('bcrypt');
const session = require('express-session');
const { spawn } = require('child_process');

const app = express();
const port = 3000;

// Initialize express-session middleware
app.use(session({
    secret: 'your_secret_key',
    resave: false,
    saveUninitialized: true
}));

app.use(express.json());
app.use(express.static(path.join(__dirname, 'Login and Signup')));
app.use(express.static(path.join(__dirname, 'Website')));

// MySQL database setup
const db = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: 'Chiku@4009',
    database: 'finance-tracker',
});

db.connect((err) => {
    if (err) {
        console.error('Error connecting to MySQL:', err.message);
    } else {
        console.log('Connected to MySQL database');
        // Create a 'credentials' table if it doesn't exist
        db.query(`
            CREATE TABLE IF NOT EXISTS credentials (
                phone BIGINT UNIQUE PRIMARY KEY,
                password VARCHAR(255)
            )
        `, (err) => {
            if (err) {
                console.error('Error creating table:', err.message);
            }
        });

        // Create a 'personal_details' table if it doesn't exist
        db.query(`
            CREATE TABLE IF NOT EXISTS personal_details (
                phone BIGINT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                gender VARCHAR(10) NOT NULL,
                date_of_birth DATE NOT NULL,
                FOREIGN KEY (phone) REFERENCES credentials(phone) ON DELETE CASCADE ON UPDATE CASCADE
            )
        `, (err) => {
            if (err) {
                console.error('Error creating table:', err.message);
            }
        });

        // Create a 'expenses' table if it doesn't exist
        db.query(`
            CREATE TABLE IF NOT EXISTS expenses (
                id INT AUTO_INCREMENT PRIMARY KEY,
                phone BIGINT NOT NULL,
                date DATE NOT NULL,
                amount DECIMAL(10, 2) NOT NULL,
                description VARCHAR(200) NOT NULL,
                category VARCHAR(100) NOT NULL,
                FOREIGN KEY (phone) REFERENCES credentials(phone) ON DELETE CASCADE ON UPDATE CASCADE
            )
        `, (err) => {
            if (err) {
                console.error('Error creating budget table:', err.message);
            }
        });
    }
});

// Authentication endpoint for login
app.post('/auth/login', async (req, res) => {
    const {phone, password } = req.body;

    // Retrieve hashed password from the database based on the phone number
    const query = 'SELECT * FROM credentials WHERE phone = ?';
    db.query(query, [phone], async (err, results) => {
        if (err) {
            console.error('Error retrieving user:', err.message);
            return res.status(500).json({ success: false, message: 'Internal Server Error' });
        }

        if (results.length === 0) {
            return res.status(401).json({ success: false, message: 'Invalid credentials' });
        }

        const hashedPassword = results[0].password;

        // Compare hashed password with the submitted password
        const passwordMatch = await bcrypt.compare(password, hashedPassword);

        if (passwordMatch) {
            // Store the phone number in the session
            req.session.phone = phone;
            req.session.password = hashedPassword;
            res.json({ success: true });

        } else {
            res.status(401).json({ success: false, message: 'Invalid credentials' });
        }
    });
});

// Authentication endpoint for user registration
app.post('/auth/register', async (req, res) => {   
    const { phone, password } = req.body;

    // Hash the password before storing it in the database
    const hashedPassword = await bcrypt.hash(password, 10);

    // Insert user into the 'credentials' table
    const insertQuery = 'INSERT INTO credentials (phone, password) VALUES (?, ?)';
    db.query(insertQuery, [phone, hashedPassword], (err) => {
        if (err) {
            console.error('Error inserting user:', err.message);
            return res.status(400).json({ success: false, message: 'Phone number already exists' });
        }

        req.session.phone = phone;
        req.session.password = hashedPassword;
        res.json({ success: true });
    });
});

//Endpoint to get current phone number and password
app.get('/Details', (req, res) => {
    // Check if the user is logged in
    if (!req.session.phone || !req.session.password) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }

    // Fetch current credentials for the logged-in user
    const phone = req.session.phone;
    const password = req.session.password;

    // Combine both queries into a single query using JOIN
    const combinedQuery = `
        SELECT pd.name, pd.gender, pd.date_of_birth, e.date, e.amount, e.description, e.category
        FROM personal_details pd
        LEFT JOIN expenses e ON pd.phone = e.phone
        WHERE pd.phone = ?
    `;

    // Execute the combined query
    db.query(combinedQuery, [phone], (err, results) => {
        if (err) {
            console.error('Error retrieving details:', err.message);
            return res.status(500).json({ success: false, message: 'Internal Server Error' });
        }

        if (results.length === 0) {
            return res.json({ success: true, message: 'No details available' });
        }


        // Send the response with combined personal and budget information
        const { name, gender, date_of_birth: dateOfBirth, date, amount, description, category} = results[0];
        res.json({ success: true, phone, password, name, gender, dateOfBirth, date, amount, description, category});
    });
});

// Endpoint to save personal details
app.post('/savePersonalDetails', (req, res) => {
    // Check if the user is logged in
    if (!req.session.phone || !req.session.password) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }

    const { name, gender, dateOfBirth } = req.body;
    const phone = req.session.phone;

    // Insert or update personal details for the user
    const insertOrUpdateQuery = `
        INSERT INTO personal_details (phone, name, gender, date_of_birth)
        VALUES (?, ?, ?, ?)
        ON DUPLICATE KEY UPDATE name=?, gender=?, date_of_birth=?
    `;
    db.query(insertOrUpdateQuery, [phone, name, gender, dateOfBirth, name, gender, dateOfBirth], (err) => {
        if (err) {
            console.error('Error saving personal details:', err.message);
            return res.status(500).json({ success: false, message: 'Internal Server Error' });
        }
        res.json({ success: true, message: 'Personal details saved successfully' });
    });
});

// Endpoint to retrieve expenses for the current month
app.get('/currentMonthExpenses', (req, res) => {
    // Check if the user is logged in
    if (!req.session.phone || !req.session.password) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }

    // Fetch expenses for the logged-in user for the current month
    const phone = req.session.phone;
    const currentMonth = new Date().getMonth() + 1; // Get the current month (1-indexed)
    const currentYear = new Date().getFullYear(); // Get the current year

    // Construct the start and end date of the current month
    const startDate = new Date(currentYear, currentMonth - 1, 1); // 1st day of the current month
    const endDate = new Date(currentYear, currentMonth, 0); // Last day of the current month

    // Format the dates as YYYY-MM-DD strings
    const formattedStartDate = startDate.toISOString().split('T')[0];
    const formattedEndDate = endDate.toISOString().split('T')[0];

    // SQL query to fetch expenses for the current month
    const query = `
        SELECT id, date, amount, description, category
        FROM expenses
        WHERE phone = ? AND date >= ? AND date <= ?
    `;
    const queryParams = [phone, formattedStartDate, formattedEndDate];

    db.query(query, queryParams, (err, results) => {
        if (err) {
            console.error('Error retrieving current month expenses:', err.message);
            return res.status(500).json({ success: false, message: 'Internal Server Error' });
        }

        res.json(results);
    });
});


// Authentication endpoint for accessing budget information
app.post('/saveExpenses', (req, res) => {
    // Check if the user is logged in
    if (!req.session.phone || !req.session.password) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }

    // Extract expense details from request body
    const { date, amount, description, category} = req.body;
    const phone = req.session.phone;

    // Insert expense into the database
    const insertQuery = `
        INSERT INTO expenses (phone, date, amount, description, category)
        VALUES (?, ?, ?, ?, ?)
    `;
    db.query(insertQuery, [phone, date, amount, description, category, date, amount, description, category], (err, result) => {
        if (err) {
            console.error('Error adding expense:', err.message);
            return res.status(500).json({ success: false, message: 'Internal Server Error' });
        }
        // Send the inserted expense's id back to the client
        res.json({ success: true, message: 'Expense added successfully', id: result.insertId });
    });
});

// Endpoint to retrieve expenses history for the current user
app.get('/expensesHistory', (req, res) => {
    // Check if the user is logged in
    if (!req.session.phone || !req.session.password) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }

    // Fetch expenses history for the logged-in user based on selected category
    const phone = req.session.phone;
    const category = req.query.category; // Retrieve selected category from query parameters

    let query = 'SELECT id, date, amount, description FROM expenses WHERE phone = ? ORDER BY date DESC;';
    const queryParams = [phone];

    // If category is provided, add category filter to the query
    if (category && category !== 'all') {
        query += ' AND category = ?';
        queryParams.push(category);
    }

    db.query(query, queryParams, (err, results) => {
        if (err) {
            console.error('Error retrieving expenses history:', err.message);
            return res.status(500).json({ success: false, message: 'Internal Server Error' });
        }

        res.json(results);
    });
});

// Endpoint to remove an expense
app.delete('/expenses/:id', (req, res) => {
    // Check if the user is logged in
    if (!req.session.phone || !req.session.password) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }

    // Extract the id of the expense to be removed from the request parameters
    const id = parseInt(req.params.id, 10); // Parse the id as an integer with base 10

    // Delete the expense from the database
    const deleteQuery = `
        DELETE FROM expenses WHERE id = ?
    `;
    db.query(deleteQuery, [id], (err) => {
        if (err) {
            console.error('Error removing expense:', err.message);
            return res.status(500).json({ success: false, message: 'Internal Server Error' });
        }
        res.json({ success: true, message: 'Expense removed successfully' });
    });
});


// Endpoint to retrieve data for the analysis
app.get('/expensesData', (req, res) => {
    // Check if the user is logged in
    if (!req.session.phone || !req.session.password) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }

    // Fetch data for the analysis based on selected date range
    const { fromDate, toDate } = req.query;

    // Query to fetch data for the analysis within the specified date range
    let query = `
        SELECT category, SUM(amount) AS total_amount
        FROM expenses
        WHERE date >= ? AND date <= ?
        GROUP BY category
    `;

    const queryParams = [fromDate, toDate];

    db.query(query, queryParams, (err, results) => {
        if (err) {
            console.error('Error retrieving graph data:', err.message);
            return res.status(500).json({ success: false, message: 'Internal Server Error' });
        }

        res.json(results);
    });
});

// Route to perform financial analysis
app.post('/analyzeFinancialData', (req, res) => {
    // Extract financial data from request body
    const { data } = req.body;

    console.log(`Data to be analyzed: ${JSON.stringify(data)}`);

    // Spawn a Python process to execute the analysis script
    const pythonProcess = spawn('python', ['analysis.py', JSON.stringify(data)]);

    let analysisResult = '';

    // Capture stdout data from the Python script
    pythonProcess.stdout.on('data', (result) => {
        analysisResult += result.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });

    // When Python process ends
    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            console.error('Python process exited with error code:', code);
            res.status(500).json({ error: 'Internal Server Error' });
        } else {
            try {
                // Send the result back to the client as plain text
                res.set('Content-Type', 'text/plain');
                res.send(analysisResult);
            } catch (error) {
                console.error('Error parsing JSON:', error);
                res.status(500).json({ error: 'Internal Server Error' });
            }
        }
    });

    // Handle any errors from the Python process
    pythonProcess.on('error', (error) => {
        console.error('Python process error:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    });
});


//Endpoint to change the user's password
app.post('/changePassword', async (req, res) => {
    // Check if the user is logged in
    if (!req.session.phone || !req.session.password) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }

    const { newPassword } = req.body;

    try {
        // Hash the new password
        const newHashedPassword = await bcrypt.hash(newPassword, 10);

        // Update the hashed password in the session
        req.session.password = newHashedPassword;

        // Update the hashed password in the database
        const phone = req.session.phone;
        const updateQuery = 'UPDATE credentials SET password = ? WHERE phone = ?';
        db.query(updateQuery, [newHashedPassword, phone], (err) => {
            if (err) {
                console.error('Error updating password:', err.message);
                return res.status(500).json({ success: false, message: 'Internal Server Error' });
            }
            res.json({ success: true, message: 'Password updated successfully' });
        });
    } catch (error) {
        console.error('Error changing password:', error);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});

//Endpoint for user logout
app.get('/logout', (req, res) => {
    // Clear session data
    req.session.destroy(err => {
        if (err) {
            console.error('Error destroying session:', err);
            return res.status(500).json({ success: false, message: 'Internal Server Error' });
        }
        // Redirect user to the login page
        res.redirect('http://localhost:3000'); // Change the path to your actual login page
    });
});


app.listen(port, () => {
    console.log(`Server is listening on port ${port}`);
});