const express = require('express');
const router = express.Router();
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { body } = require('express-validator');
const pool = require('../config/database');
const { handleValidationErrors } = require('../middleware/validation');
const { loginLimiter, registerLimiter, passwordResetLimiter } = require('../middleware/rateLimiter');

// POST /auth/login
router.post('/login',
    loginLimiter,
    body('phone').isNumeric().isLength({ min: 10, max: 15 }),
    body('password').notEmpty(),
    handleValidationErrors,
    async (req, res) => {
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
    }
);

// POST /auth/register
router.post('/register',
    registerLimiter,
    body('phone').isNumeric().isLength({ min: 10, max: 15 }),
    body('password').isLength({ min: 8 }).withMessage('Password must be at least 8 characters long'),
    handleValidationErrors,
    async (req, res) => {
        const { phone, password } = req.body;
        const hashedPassword = await bcrypt.hash(password, 10);
        try {
            await pool.query('INSERT INTO credentials (phone, password) VALUES ($1, $2)', [phone, hashedPassword]);
            req.session.phone = phone;

            // Create JWT token (same as login)
            const token = jwt.sign(
                { phone },
                process.env.JWT_SECRET,
                { expiresIn: "7d" }
            );

            // Set auth token cookie
            res.cookie("auth_token", token, {
                httpOnly: true,
                secure: false,         // true if HTTPS (use false on localhost)
                sameSite: "strict",
                maxAge: 7 * 24 * 60 * 60 * 1000 // 7 days
            });

            res.json({ success: true });
        } catch (err) {
            console.error('Error inserting user:', err.message);
            res.status(400).json({ success: false, message: 'Phone Number Already Exists!' });
        }
    }
);

// POST /auth/verify
router.post('/verify',
    passwordResetLimiter,
    body('phone').isNumeric().isLength({ min: 10, max: 15 }),
    body('dob').isISO8601(),
    handleValidationErrors,
    async (req, res) => {
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
    }
);

// POST /auth/resetPassword
router.post('/resetPassword',
    passwordResetLimiter,
    body('phone').isNumeric().isLength({ min: 10, max: 15 }),
    body('newPassword').isLength({ min: 8 }).withMessage('Password must be at least 8 characters long'),
    handleValidationErrors,
    async (req, res) => {
        const { phone, newPassword } = req.body;
        try {
            const newHashedPassword = await bcrypt.hash(newPassword, 10);
            await pool.query('UPDATE credentials SET password = $1 WHERE phone = $2', [newHashedPassword, phone]);
            res.json({ success: true, message: 'Password updated successfully' });
        } catch (error) {
            console.error('Error updating password:', error.message);
            res.status(500).json({ success: false, message: 'Internal Server Error' });
        }
    }
);

// GET /logout
router.get('/logout', (req, res) => {
    // Clear session data
    req.session.destroy(err => {
        if (err) {
            console.error('Error destroying session:', err);
            return res.status(500).json({ success: false, message: 'Internal Server Error' });
        }
        res.status(200).json({ success: true, message: 'Logged out successfully' });
    });
});

module.exports = router;
