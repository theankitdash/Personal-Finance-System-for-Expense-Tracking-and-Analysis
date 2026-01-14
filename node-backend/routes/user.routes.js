const express = require('express');
const router = express.Router();
const bcrypt = require('bcryptjs');
const { body } = require('express-validator');
const pool = require('../config/database');
const { requireAuth } = require('../middleware/auth');
const { handleValidationErrors } = require('../middleware/validation');

// GET /personalDetails
router.get('/personalDetails', requireAuth, async (req, res) => {
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

// POST /savePersonalDetails
router.post('/savePersonalDetails',
    requireAuth,
    body('name').trim().notEmpty().isLength({ max: 255 }),
    body('gender').isIn(['Male', 'Female', 'Other']),
    body('dateOfBirth').isISO8601(),
    handleValidationErrors,
    async (req, res) => {
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
    }
);

// POST /changePassword
router.post('/changePassword',
    requireAuth,
    body('newPassword').isLength({ min: 8 }).withMessage('Password must be at least 8 characters long'),
    handleValidationErrors,
    async (req, res) => {
        const { newPassword } = req.body;
        try {
            const newHashedPassword = await bcrypt.hash(newPassword, 10);
            const phone = req.session.phone;
            await pool.query('UPDATE credentials SET password = $1 WHERE phone = $2', [newHashedPassword, phone]);
            res.json({ success: true, message: 'Password updated successfully' });
        } catch (error) {
            console.error('Error changing password:', error);
            res.status(500).json({ success: false, message: 'Internal Server Error' });
        }
    }
);

module.exports = router;
