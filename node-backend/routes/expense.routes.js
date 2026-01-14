const express = require('express');
const router = express.Router();
const { body } = require('express-validator');
const pool = require('../config/database');
const { requireAuth } = require('../middleware/auth');
const { handleValidationErrors } = require('../middleware/validation');

// GET /currentMonthExpenses
router.get('/currentMonthExpenses', requireAuth, async (req, res) => {
    const phone = req.session.phone;
    try {
        const result = await pool.query(
            `SELECT id, date, amount, description, category
            FROM expenses
            WHERE phone = $1 
              AND date >= DATE_TRUNC('month', CURRENT_DATE)
              AND date < DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month'
            ORDER BY date DESC`,
            [phone]
        );
        res.json(result.rows);
    } catch (err) {
        console.error('Error retrieving current month expenses:', err.message);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});

// GET /expensesHistory
router.get('/expensesHistory', requireAuth, async (req, res) => {
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

// GET /uniqueOptions
router.get('/uniqueOptions', async (req, res) => {
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

// POST /saveExpenses
router.post('/saveExpenses',
    requireAuth,
    body('date').isISO8601(),
    body('amount').isFloat({ min: 0.01 }),
    body('description').trim().notEmpty().isLength({ max: 200 }),
    body('category').trim().notEmpty().isLength({ max: 100 }),
    handleValidationErrors,
    async (req, res) => {
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
    }
);

// PUT /updateExpenses/:id
router.put('/updateExpenses/:id',
    requireAuth,
    body('date').isISO8601(),
    body('amount').isFloat({ min: 0.01 }),
    body('description').trim().notEmpty().isLength({ max: 200 }),
    body('category').trim().notEmpty().isLength({ max: 100 }),
    handleValidationErrors,
    async (req, res) => {
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
    }
);

// DELETE /expenses/:id
router.delete('/expenses/:id', requireAuth, async (req, res) => {
    const id = parseInt(req.params.id, 10);
    const phone = req.session.phone;
    try {
        const result = await pool.query('DELETE FROM expenses WHERE id = $1 AND phone = $2', [id, phone]);
        if (result.rowCount === 0) {
            return res.status(404).json({ success: false, message: 'Expense not found or not authorized' });
        }
        res.json({ success: true, message: 'Expense removed successfully' });
    } catch (err) {
        console.error('Error removing expense:', err.message);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});

// POST /analyzeFinancialData
router.post('/analyzeFinancialData', requireAuth, async (req, res) => {
    const { fromDate, toDate } = req.body;
    const phone = req.session.phone;

    // Generate JWT token for Python API (Python API expects this)
    const jwt = require('jsonwebtoken');
    const token = jwt.sign(
        { phone },
        process.env.JWT_SECRET,
        { expiresIn: "7d" }
    );

    try {
        const response = await fetch(`${process.env.PYTHON_API_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Cookie': `auth_token=${token}`  // Send JWT token to Python API
            },
            body: JSON.stringify({ fromDate, toDate })
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Python API error:', errorText);
            return res.status(response.status).json({ error: 'Python API error', details: errorText });
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
        console.error('Error calling Python API:', err);
        res.status(500).json({ error: 'Failed to call Python API', details: err.message });
    }
});

module.exports = router;
