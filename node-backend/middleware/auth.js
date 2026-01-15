const jwt = require('jsonwebtoken');

// Authentication middleware
function requireAuth(req, res, next) {
    // First check session
    if (req.session && req.session.phone) {
        return next();
    }

    // If no session, check for JWT token in cookies
    const token = req.cookies.auth_token;
    if (token) {
        try {
            const decoded = jwt.verify(token, process.env.JWT_SECRET);
            // Set session phone from token for downstream handlers
            req.session.phone = decoded.phone;
            return next();
        } catch (err) {
            // Token is invalid or expired
            return res.status(401).json({ success: false, message: 'Invalid or expired token' });
        }
    }

    // No valid authentication found
    return res.status(401).json({ success: false, message: 'Unauthorized' });
}

module.exports = { requireAuth };
