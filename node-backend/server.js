const express = require('express');
const path = require('path');
require('dotenv').config();
const { Pool } = require('pg');
const bcrypt = require('bcryptjs');
const session = require('express-session');
const jwt = require("jsonwebtoken");

const app = express();
const port = process.env.PORT || 3000;

// Initialize express-session middleware
app.use(session({
    secret: process.env.SESSION_SECRET,
    resave: false,
    saveUninitialized: true
}));

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));


// PostgreSQL database setup
const pool = new Pool({
    host: process.env.PG_HOST,
    user: process.env.PG_USER,
    password: process.env.PG_PASSWORD,
    database: process.env.PG_DATABASE,
    port: process.env.PG_PORT
});

// Table creation
(async () => {
    try {
        await pool.query(`
            CREATE TABLE IF NOT EXISTS credentials (
                phone BIGINT PRIMARY KEY,
                password VARCHAR(255)
            );
        `);

        await pool.query(`
            CREATE TABLE IF NOT EXISTS personal_details (
                phone BIGINT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                gender VARCHAR(10) NOT NULL,
                date_of_birth DATE NOT NULL,
                FOREIGN KEY (phone) REFERENCES credentials(phone) ON DELETE CASCADE ON UPDATE CASCADE
            );
        `);

        await pool.query(`
            CREATE TABLE IF NOT EXISTS budget (
                phone BIGINT NOT NULL,
                category VARCHAR(100) NOT NULL,
                amount NUMERIC(10, 2),
                PRIMARY KEY (phone, category),
                FOREIGN KEY (phone) REFERENCES credentials(phone) ON DELETE CASCADE ON UPDATE CASCADE
            );
        `);

        await pool.query(`
            CREATE TABLE IF NOT EXISTS expenses (
                id SERIAL PRIMARY KEY,
                phone BIGINT NOT NULL,
                date DATE NOT NULL,
                amount NUMERIC(10, 2) NOT NULL,
                description VARCHAR(200) NOT NULL,
                category VARCHAR(100) NOT NULL,
                FOREIGN KEY (phone) REFERENCES credentials(phone) ON DELETE CASCADE ON UPDATE CASCADE
            );
        `);

        console.log('Connected to PostgreSQL and ensured tables exist');
    } catch (err) {
        console.error('Error setting up PostgreSQL:', err.message);
    }
})();

// Authentication endpoint for login
app.post('/auth/login', async (req, res) => {
    const { phone, password } = req.body;
    try {
        const result = await pool.query('SELECT * FROM credentials WHERE phone = $1', [phone]);
        if (result.rows.length === 0) {
            return res.status(401).json({ success: false, message: 'Invalid Credentials!' });
        }
        const hashedPassword = result.rows[0].password;
        const passwordMatch = await bcrypt.compare(password, hashedPassword);
        if (passwordMatch) {
            req.session.phone = phone;
            const token = jwt.sign(
                { phone },
                process.env.JWT_SECRET,
                { expiresIn: "7d" }
            );

            res.cookie("auth_token", token, {
                httpOnly: true,
                secure: false,         // true if HTTPS (use false on localhost)
                sameSite: "strict",
                maxAge: 7 * 24 * 60 * 60 * 1000 // 7 days
            });
            res.json({ success: true });
        } else {
            res.status(401).json({ success: false, message: 'Invalid Credentials!' });
        }
    } catch (err) {
        console.error('Error retrieving user:', err.message);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});

// Authentication endpoint for user registration
app.post('/auth/register', async (req, res) => {   
    const { phone, password } = req.body;
    const hashedPassword = await bcrypt.hash(password, 10);
    try {
        await pool.query('INSERT INTO credentials (phone, password) VALUES ($1, $2)', [phone, hashedPassword]);
        req.session.phone = phone;
        res.json({ success: true });
    } catch (err) {
        console.error('Error inserting user:', err.message);
        res.status(400).json({ success: false, message: 'Phone Number Already Exists!' });
    }
});

// Endpoint to verify phone number and date of birth
app.post('/auth/verify', async (req, res) => {
    const { phone, dob } = req.body;
    try {
        const result = await pool.query('SELECT * FROM personal_details WHERE phone = $1 AND date_of_birth = $2', [phone, dob]);
        if (result.rows.length === 0) {
            return res.status(400).json({ success: false, message: 'Verification failed' });
        }
        res.json({ success: true });
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
        await pool.query('UPDATE credentials SET password = $1 WHERE phone = $2', [newHashedPassword, phone]);
        res.json({ success: true, message: 'Password updated successfully' });
    } catch (error) {
        console.error('Error updating password:', error.message);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});


//Endpoint to get current phone number and password
app.get('/personalDetails', async (req, res) => {
    if (!req.session.phone) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }
    const phone = req.session.phone;
    try {
        const result = await pool.query(
            'SELECT name, gender, date_of_birth AS "dateOfBirth" FROM personal_details WHERE phone = $1',
            [phone]
        );
        if (result.rows.length === 0) {
            return res.status(404).json({ success: false, message: 'No personal details found' });
        }
        const personalDetails = result.rows[0];
        res.json({ success: true, phone, personalDetails });
    } catch (err) {
        console.error('Error retrieving personal details:', err.message);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});

// Endpoint to save personal details
app.post('/savePersonalDetails', async (req, res) => {
    if (!req.session.phone) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }
    const { name, gender, dateOfBirth } = req.body;
    const phone = req.session.phone;
    try {
        await pool.query(
            `INSERT INTO personal_details (phone, name, gender, date_of_birth)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (phone) DO UPDATE SET name = EXCLUDED.name, gender = EXCLUDED.gender, date_of_birth = EXCLUDED.date_of_birth`,
            [phone, name, gender, dateOfBirth]
        );
        res.json({ success: true, message: 'Personal details saved successfully' });
    } catch (err) {
        console.error('Error saving personal details:', err.message);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});

// Endpoint to retrieve expenses for the current month
app.get('/currentMonthExpenses', async (req, res) => {
    if (!req.session.phone) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }
    const phone = req.session.phone;
    try {
        const result = await pool.query(
            `SELECT id, date, amount, description, category
            FROM expenses
            WHERE phone = $1 AND EXTRACT(MONTH FROM date) = EXTRACT(MONTH FROM CURRENT_DATE) AND EXTRACT(YEAR FROM date) = EXTRACT(YEAR FROM CURRENT_DATE)`,
            [phone]
        );
        res.json(result.rows);
    } catch (err) {
        console.error('Error retrieving current month expenses:', err.message);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});


// Endpoint to retrieve expenses history for the current user
app.get('/expensesHistory', async (req, res) => {
    if (!req.session.phone) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }
    const phone = req.session.phone;
    const filter = req.query.filter || 'all';
    const value = req.query.value || '';
    let query = `SELECT id, date, amount, description, category FROM expenses WHERE phone = $1`;
    const queryParams = [phone];
    if (filter !== 'all') {
        if (filter === 'category') {
            query += ' AND category = $2';
            queryParams.push(value);
        } else if (filter === 'description') {
            query += ' AND description = $2';
            queryParams.push(value);
        } else if (filter === 'date') {
            query += ' AND date = $2';
            queryParams.push(value);
        }
    }
    query += ' ORDER BY date DESC';
    try {
        const result = await pool.query(query, queryParams);
        res.json(result.rows);
    } catch (err) {
        console.error('Error retrieving expenses history:', err.message);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});

// Endpoint to retrieve unique options for the selected filter type
app.get('/uniqueOptions', async (req, res) => {
    const phone = req.session.phone;
    const filterType = req.query.filter;
    let query = '';
    if (filterType === 'category') {
        query = 'SELECT DISTINCT category AS "uniqueOption" FROM expenses WHERE phone = $1';
    } else if (filterType === 'description') {
        query = 'SELECT DISTINCT description AS "uniqueOption" FROM expenses WHERE phone = $1';
    } else if (filterType === 'date') {
        query = 'SELECT DISTINCT date AS "uniqueOption" FROM expenses WHERE phone = $1';
    } else {
        return res.status(400).json({ success: false, message: 'Invalid filter type' });
    }
    try {
        const result = await pool.query(query, [phone]);
        const uniqueOptions = result.rows.map(row => row.uniqueOption);
        res.json(uniqueOptions);
    } catch (err) {
        console.error('Error retrieving unique options:', err.message);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});

// Endpoint to retrieve distinct expense categories
app.get('/categories', async (req, res) => {
    if (!req.session.phone) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }
    const phone = req.session.phone;
    try {
        const result = await pool.query('SELECT DISTINCT category FROM budget WHERE phone = $1', [phone]);
        const categories = result.rows.map(row => row.category);
        res.json({ success: true, categories });
    } catch (err) {
        console.error('Error retrieving categories:', err.message);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});

// Authentication endpoint for accessing budget information
app.post('/saveExpenses', async (req, res) => {
    if (!req.session.phone) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }
    const { date, amount, description, category } = req.body;
    const phone = req.session.phone;
    try {
        const result = await pool.query(
            'INSERT INTO expenses (phone, date, amount, description, category) VALUES ($1, $2, $3, $4, $5) RETURNING id',
            [phone, date, amount, description, category]
        );
        res.json({ success: true, message: 'Expense added successfully', id: result.rows[0].id });
    } catch (err) {
        console.error('Error adding expense:', err.message);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});

// Endpoint for updating an existing expense
app.put('/updateExpenses/:id', async (req, res) => {
    if (!req.session.phone) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }
    const { date, amount, description, category } = req.body;
    const { id } = req.params;
    const phone = req.session.phone;
    try {
        const result = await pool.query(
            `UPDATE expenses SET date = $1, amount = $2, description = $3, category = $4 WHERE id = $5 AND phone = $6`,
            [date, amount, description, category, id, phone]
        );
        if (result.rowCount === 0) {
            return res.status(404).json({ success: false, message: 'Expense not found or not authorized' });
        }
        res.json({ success: true, message: 'Expense updated successfully' });
    } catch (err) {
        console.error('Error updating expense:', err.message);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});


// Endpoint to remove an expense
app.delete('/expenses/:id', async (req, res) => {
    if (!req.session.phone) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }
    const id = parseInt(req.params.id, 10);
    try {
        await pool.query('DELETE FROM expenses WHERE id = $1', [id]);
        res.json({ success: true, message: 'Expense removed successfully' });
    } catch (err) {
        console.error('Error removing expense:', err.message);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});

// Endpoint to save or update budget details
app.post('/saveBudgetDetails', async (req, res) => {
    if (!req.session.phone) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }
    const { category, amount } = req.body;
    const phone = req.session.phone;
    let amountValue;
    if (amount === '' || amount === null || isNaN(parseFloat(amount))) {
        amountValue = null;
    } else {
        amountValue = parseFloat(amount);
    }
    try {
        await pool.query(
            `INSERT INTO budget (phone, category, amount)
            VALUES ($1, $2, $3)
            ON CONFLICT (phone, category) DO UPDATE SET amount = EXCLUDED.amount`,
            [phone, category, amountValue]
        );
        res.json({ success: true, message: 'Budget details saved successfully' });
    } catch (err) {
        console.error('Error saving budget details:', err.message);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});

// Endpoint to get budget details
app.get('/getBudgetDetails', async (req, res) => {
    if (!req.session.phone) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }
    const phone = req.session.phone;
    try {
        const result = await pool.query('SELECT category, amount FROM budget WHERE phone = $1', [phone]);
        res.json({ success: true, budgets: result.rows });
    } catch (err) {
        console.error('Error retrieving budget details:', err.message);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});

// Endpoint to analyze financial data by Python API
app.post('/analyzeFinancialData', async (req, res) => {
    const { fromDate, toDate } = req.body;
    try {
        const response = await fetch(`${process.env.PYTHON_API_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Cookie: req.headers.cookie  
            },
            body: JSON.stringify({ fromDate, toDate })
        });

        if (!response.ok) {
            return res.status(response.status).json({ error: 'Python API error' });
        }

        // Convert the Python stream to a Node buffer
        const arrayBuffer = await response.arrayBuffer();
        const buffer = Buffer.from(arrayBuffer);

        // Prepare correct file headers
        const filename = `analysis_${fromDate}_to_${toDate}.xlsx`;

        res.setHeader(
            'Content-Type',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        );
        res.setHeader(
            'Content-Disposition',
            `attachment; filename="${filename}"`
        );

        // Send the Excel file
        res.send(buffer);

    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Failed to call Python API' });
    }
});
   
//Endpoint to change the user's password
app.post('/changePassword', async (req, res) => {
    if (!req.session.phone) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }
    const { newPassword } = req.body;
    if (!newPassword || newPassword.length < 8) {
        return res.status(400).json({ success: false, message: 'Password must be at least 8 characters long' });
    }
    try {
        const newHashedPassword = await bcrypt.hash(newPassword, 10);
        const phone = req.session.phone;
        await pool.query('UPDATE credentials SET password = $1 WHERE phone = $2', [newHashedPassword, phone]);
        res.json({ success: true, message: 'Password updated successfully' });
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

// Run with command: npx nodemon node-backend/server.js