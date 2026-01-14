const rateLimit = require('express-rate-limit');

// Rate limiter for login endpoint
const loginLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5, // 5 attempts
    message: { success: false, message: 'Too many login attempts. Please try again later.' },
    standardHeaders: true,
    legacyHeaders: false
});

// Rate limiter for registration endpoint
const registerLimiter = rateLimit({
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 3, // 3 attempts
    message: { success: false, message: 'Too many registration attempts. Please try again later.' },
    standardHeaders: true,
    legacyHeaders: false
});

// Rate limiter for password reset endpoints
const passwordResetLimiter = rateLimit({
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 3, // 3 attempts
    message: { success: false, message: 'Too many password reset attempts. Please try again later.' },
    standardHeaders: true,
    legacyHeaders: false
});

module.exports = { loginLimiter, registerLimiter, passwordResetLimiter };
