const express = require('express');
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '.env') });

// Utilities
const validateEnv = require('./utils/envValidator');
validateEnv();

// Configuration
const pool = require('./config/database');
const sessionMiddleware = require('./config/session');

// Routes
const authRoutes = require('./routes/auth.routes');
const userRoutes = require('./routes/user.routes');
const expenseRoutes = require('./routes/expense.routes');
const budgetRoutes = require('./routes/budget.routes');

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));
app.use(sessionMiddleware);

// Mount routes
app.use('/auth', authRoutes);
app.use('/', userRoutes);
app.use('/', expenseRoutes);
app.use('/', budgetRoutes);

// Start server
const server = app.listen(port, () => {
    console.log(`🚀 Server is listening on port ${port}`);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
    console.log('SIGTERM received, closing server gracefully...');
    server.close(async () => {
        await pool.end();
        console.log('Server closed');
        process.exit(0);
    });
});

process.on('SIGINT', async () => {
    console.log('\nSIGINT received, closing server gracefully...');
    server.close(async () => {
        await pool.end();
        console.log('Server closed');
        process.exit(0);
    });
});

module.exports = { app, server };

// Run with command: cd node-backend && npx nodemon server.js