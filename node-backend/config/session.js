const session = require('express-session');

// Session middleware configuration
const sessionMiddleware = session({
    secret: process.env.SESSION_SECRET,
    resave: false,
    saveUninitialized: false,  // Only create sessions for authenticated users
    cookie: {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',  // HTTPS in production
        sameSite: 'strict',
        maxAge: 7 * 24 * 60 * 60 * 1000  // 7 days
    }
});

module.exports = sessionMiddleware;
