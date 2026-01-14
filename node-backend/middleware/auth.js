// Authentication middleware
function requireAuth(req, res, next) {
    if (!req.session.phone) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
    }
    next();
}

module.exports = { requireAuth };
