const express = require('express');
const router = express.Router();
const { body } = require('express-validator');
const pool = require('../config/database');
const { requireAuth } = require('../middleware/auth');
const { handleValidationErrors } = require('../middleware/validation');

// GET /categories
router.get('/categories', requireAuth, async (req, res) => {
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

// GET /getBudgetDetails
router.get('/getBudgetDetails', requireAuth, async (req, res) => {
    const phone = req.session.phone;
    try {
        const result = await pool.query('SELECT category, amount FROM budget WHERE phone = $1', [phone]);
        res.json({ success: true, budgets: result.rows });
    } catch (err) {
        console.error('Error retrieving budget details:', err.message);
        res.status(500).json({ success: false, message: 'Internal Server Error' });
    }
});

// POST /saveBudgetDetails
router.post('/saveBudgetDetails',
    requireAuth,
    body('category').trim().notEmpty().isLength({ max: 100 }),
    body('amount').optional({ nullable: true }).isFloat({ min: 0 }),
    handleValidationErrors,
    async (req, res) => {
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
    }
);

module.exports = router;
