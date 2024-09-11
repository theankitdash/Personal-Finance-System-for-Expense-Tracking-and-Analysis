const express = require('express');
const path = require('path');
const mysql = require('mysql');
const bcrypt = require('bcryptjs');
const session = require('express-session');
const { spawn } = require('child_process');

const app = express();
const port = process.env.PORT || 3000;

// Initialize express-session middleware
app.use(session({
    secret: process.env.SESSION_SECRET || 'your_secret_key',
    resave: false,
    saveUninitialized: true
}));

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));


// MySQL database setup
const db = mysql.createConnection({
    host: process.env.AZURE_MYSQL_HOST || 'localhost',
    user: process.env.AZURE_MYSQL_USER || 'root',
    password: process.env.AZURE_MYSQL_PASSWORD || 'Chiku@4009',
    database: process.env.AZURE_MYSQL_DATABASE || 'finance-tracker',
    port: process.env.AZURE_MYSQL_PORT || 3306,
    ssl: process.env.AZURE_MYSQL_SSL ? JSON.parse(process.env.AZURE_MYSQL_SSL) : false
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

        // Create a 'budget' table if it doesn't exist
        db.query(`
            CREATE TABLE IF NOT EXISTS budget (
                phone BIGINT NOT NULL,
                category VARCHAR(100) NOT NULL,
                amount DECIMAL(10, 2),
                PRIMARY KEY (phone, category),
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
                FOREIGN KEY (phone) REFERENCES credentials(phone) ON DELETE CASCADE ON UPDATE CASCADE,
                FOREIGN KEY (category) REFERENCES budget(category) ON DELETE CASCADE ON UPDATE CASCADE
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
            return res.status(401).json({ success: false, message: 'Invalid Credentials!' });
        }

        const hashedPassword = results[0].password;

        // Compare hashed password with the submitted password
        const passwordMatch = await bcrypt.compare(password, hashedPassword);

        if (passwordMatch) {
            // Store the phone number in the session
            req.session.phone = phone;
            res.json({ success: true });

        } else {
            res.status(401).json({ success: false, message: 'Invalid Credentials!' });
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
            return res.status(400).json({ success: false, message: 'Phone Number Already Exists!' });
        }
        req.session.phone = phone;
        res.json({ success: true });
    });
});

// Endpoint to verify phone number and date of birth
app.post('/auth/verify', async (req, res) => {
    const { phone, dob } = req.body;

    try {
        // Verify Date of Birth
        const query = 'SELECT * FROM personal_details WHERE phone = ? AND date_of_birth = ?';
        db.query(query, [phone, dob], (err, results) => {
            if (err) {
                console.error('Error retrieving details:', err.message);
                return res.status(500).json({ success: false, message: 'Internal Server Error' });
            }

            if (results.length === 0) {
                return res.status(400).json({ success: false, message: 'Verification failed' });
            }
            res.json({ success: true });
        });
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ success: false, message: 'Server error' });
    }
});

// Endpoint to reset password
app.post('/auth/resetPassword', async (req, res) => {
    const { phone, newPassword } = req.body;

    try {
        const newHashedPassword = await bcrypt.hash(newPassword, 10);

        // Update the hashed password in the database
        const updateQuery = 'UPDATE credentials SET password = ? WHERE phone = ?';
        db.query(updateQuery, [newHashedPassword, phone], (err) => {
            if (err) {
                console.error('Error updating password:', err.message);
                return res.status(500).json({ success: false, message: 'Internal Server Error' });
            }
            res.json({ success: true, message: 'Password updated successfully' });
        });
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});


//Endpoint to get current phone number and password
app.get('/Details', (req, res) => {
    // Check if the user is logged in
    if (!req.session.phone) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }

    // Fetch current credentials for the logged-in user
    const phone = req.session.phone;

    // Combine both queries into a single query using JOIN
    const budgetQuery = `
        SELECT pd.name, pd.gender, pd.date_of_birth AS dateOfBirth,
               b.category AS budgetCategory, b.amount AS budgetAmount       
        FROM personal_details pd
        LEFT JOIN budget b ON pd.phone = b.phone 
        WHERE pd.phone = ?
    `;

    const expenseQuery = `
        SELECT pd.name, pd.gender, pd.date_of_birth AS dateOfBirth, 
               e.date AS expenseDate, e.amount AS expenseAmount, e.description AS expenseDescription, e.category AS expenseCategory       
        FROM personal_details pd
        LEFT JOIN expenses e ON pd.phone = e.phone
        WHERE pd.phone = ?
    `;

    // Execute both queries
    db.query(budgetQuery, [phone], (err, budgetResults) => {
        if (err) {
            console.error('Error retrieving budget details:', err.message);
            return res.status(500).json({ success: false, message: 'Internal Server Error' });
        }

        // If no results for the first query
        if (budgetResults.length === 0) {
            return res.json({ success: true, message: 'No budget details available' });
        }

        const personalDetails = {
            name: budgetResults[0].name,
            gender: budgetResults[0].gender,
            dateOfBirth: budgetResults[0].dateOfBirth
        };

        const budgets = budgetResults.map(row => ({
            category: row.budgetCategory,
            amount: row.budgetAmount
        }));

        // Now execute the expense query after getting the budget data
        db.query(expenseQuery, [phone], (err, expenseResults) => {
            if (err) {
                console.error('Error retrieving expense details:', err.message);
                return res.status(500).json({ success: false, message: 'Internal Server Error' });
            }

            const expenses = expenseResults.map(row => ({
                date: row.expenseDate,
                amount: row.expenseAmount,
                description: row.expenseDescription,
                category: row.expenseCategory
            }));

            // Send the combined response with personal, budget, and expense details
            res.json({success: true, phone, personalDetails, budgets, expenses});
        });
    });
});

// Endpoint to save personal details
app.post('/savePersonalDetails', (req, res) => {
    // Check if the user is logged in
    if (!req.session.phone ) {
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
    if (!req.session.phone ) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }

    // Fetch expenses for the logged-in user for the current month
    const phone = req.session.phone;

    // SQL query to fetch expenses for the current month
    const query = `
        SELECT id, date, amount, description, category
        FROM expenses
        WHERE phone = ? AND MONTH(date) = MONTH(CURRENT_DATE()) AND YEAR(date) = YEAR(CURRENT_DATE())
    `;

    db.query(query, phone, (err, results) => {
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
    if (!req.session.phone ) {
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

    const queryParams = [phone, date, amount, description, category];

    db.query(insertQuery, queryParams, (err, result) => {
        if (err) {
            console.error('Error adding expense:', err.message);
            return res.status(500).json({ success: false, message: 'Internal Server Error' });
        }
        // Send the inserted expense's id back to the client
        res.json({ success: true, message: 'Expense added successfully', id: result.insertId });
    });
});

// Endpoint for updating an existing expense
app.put('/updateExpenses/:id', (req, res) => {
    // Check if the user is logged in
    if (!req.session.phone) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }

    // Extract expense details from request body
    const { date, amount, description, category } = req.body;
    const { id } = req.params;
    const phone = req.session.phone;

    // Update expense in the database
    const updateQuery = `
        UPDATE expenses
        SET date = ?, amount = ?, description = ?, category = ?
        WHERE id = ? AND phone = ?
    `;

    const queryParams = [date, amount, description, category, id, phone];
    
    db.query(updateQuery, queryParams, (err, result) => {
        if (err) {
            console.error('Error updating expense:', err.message);
            return res.status(500).json({ success: false, message: 'Internal Server Error' });
        }

        // Check if the expense was found and updated
        if (result.affectedRows === 0) {
            return res.status(404).json({ success: false, message: 'Expense not found or not authorized' });
        }

        res.json({ success: true, message: 'Expense updated successfully' });
    });
});


// Endpoint to retrieve expenses history for the current user
app.get('/expensesHistory', (req, res) => {
    // Check if the user is logged in
    if (!req.session.phone ) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }

    // Fetch expenses history for the logged-in user based on selected category
    const phone = req.session.phone;
    const filter = req.query.filter || 'all'; // Default to 'all' if not specified
    const value = req.query.value || '';

    let query = `   SELECT id, 
                        date, 
                        amount, 
                        description, 
                        category 
                    FROM expenses 
                    WHERE phone = ?
                `;
    const queryParams = [phone];

    // Modify query based on the filter type
    if (filter !== 'all') {
        if (filter === 'category') {
            query += ' AND category = ?';
            queryParams.push(value);
        } else if (filter === 'description') {
            query += ' AND description = ?';
            queryParams.push(value);
        } else if (filter === 'date') {
            query += ' AND date = ?';
                queryParams.push(value);
        }
    }

    query += ' ORDER BY date DESC';

    db.query(query, queryParams, (err, results) => {
        if (err) {
            console.error('Error retrieving expenses history:', err.message);
            return res.status(500).json({ success: false, message: 'Internal Server Error' });
        }

        res.json(results);
    });
});

// Endpoint to retrieve unique options for the selected filter type
app.get('/uniqueOptions', (req, res) => {
    const phone = req.session.phone;
    const filterType = req.query.filter;

    let query = '';
    if (filterType === 'category') {
        query = 'SELECT DISTINCT category AS uniqueOption FROM expenses WHERE phone = ?';
    } else if (filterType === 'description') {
        query = 'SELECT DISTINCT description AS uniqueOption FROM expenses WHERE phone = ?';
    } else if (filterType === 'date') {
        query = 'SELECT DISTINCT date AS uniqueOption FROM expenses WHERE phone = ?';
    } else {
        return res.status(400).json({ success: false, message: 'Invalid filter type' });
    }

    db.query(query, [phone], (err, results) => {
        if (err) {
            console.error('Error retrieving unique options:', err.message);
            return res.status(500).json({ success: false, message: 'Internal Server Error' });
        }

        const uniqueOptions = results.map(result => result.uniqueOption);
        res.json(uniqueOptions);
    });
});

// Endpoint to remove an expense
app.delete('/expenses/:id', (req, res) => {
    // Check if the user is logged in
    if (!req.session.phone ) {
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

// Endpoint to save or update budget details
app.post('/saveBudgetDetails', (req, res) => {
    // Check if the user is logged in
    if (!req.session.phone) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }

    const { category, amount } = req.body;
    const phone = req.session.phone;

    // Handle blank or invalid amount value
    let amountValue;
    if (amount === '' || amount === null || isNaN(parseFloat(amount))) {
        amountValue = null;
    } else {
        amountValue = parseFloat(amount);
    }

    // Insert or update the budget details for the user
    const insertOrUpdateBudgetQuery = `
        INSERT INTO budget (phone, category, amount)
        VALUES (?, ?, ?)
        ON DUPLICATE KEY UPDATE 
            amount = VALUES(amount)
    `;

    db.query(insertOrUpdateBudgetQuery, [phone, category, amountValue], (err) => {
        if (err) {
            console.error('Error saving budget details:', err.message);
            return res.status(500).json({ success: false, message: 'Internal Server Error' });
        }
        res.json({ success: true, message: 'Budget details saved successfully' });
    });
});


// Endpoint to retrieve data for the analysis
app.get('/expensesData', (req, res) => {
    // Check if the user is logged in
    if (!req.session.phone ) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }

    // Fetch data for the analysis based on selected date range
    const { fromDate, toDate } = req.query;
    

    // Query to fetch data within the specified date range
    let Query = `
        SELECT category, amount, description, date 
        FROM expenses
        WHERE date >= ? AND date <= ?
    `;

    db.query(Query, [fromDate, toDate], (err, Result) => {
        if (err) {
            console.error('Error retrieving Data:', err.message);
            return res.status(500).json({ success: false, message: 'Internal Server Error' });
        }

        res.json({aggregatedData: Result});
    });
});

// Route to perform financial analysis
app.post('/analyzeFinancialData', (req, res) => {

    // Extract financial data from request body
    const { aggregatedData } = req.body;

    console.log(`Aggregated Data to be analyzed: ${JSON.stringify(aggregatedData)}`);

    // Prepare data for the Python script
    const inputData = JSON.stringify({aggregatedData});

    // Spawn a Python process and pass data via stdin
    const pythonProcess = spawn('python', ['analysis.py']);

    let analysisResult = '';
    let errorOutput = '';

    // Write data to the Python process stdin
    pythonProcess.stdin.write(inputData);
    pythonProcess.stdin.end();

    // Capture stdout data from the Python script
    pythonProcess.stdout.on('data', (data) => {
        analysisResult += data.toString();
    });

    // Capture stderr data from the Python script
    pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
    });

    // When Python process ends
    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            console.error('Python process exited with error code:', code);
            res.status(500).json({ error: 'Internal Server Error', details: errorOutput });
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
    if (!req.session.phone ) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }

    const { newPassword } = req.body;

    // Validate the new password (e.g., check length, complexity)
    if (!newPassword || newPassword.length < 8) {
        return res.status(400).json({ success: false, message: 'Password must be at least 8 characters long' });
    }

    try {
        // Hash the new password
        const newHashedPassword = await bcrypt.hash(newPassword, 10);

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
        res.status(200).json({ success: true, message: 'Logged out successfully' });
    });
});


app.listen(port, () => {
    console.log(`Server is listening on port ${port}`);
});
