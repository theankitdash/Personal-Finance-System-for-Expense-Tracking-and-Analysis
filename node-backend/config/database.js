const { Pool } = require('pg');

// PostgreSQL database setup
const pool = new Pool({
    host: process.env.PG_HOST,
    user: process.env.PG_USER,
    password: process.env.PG_PASSWORD,
    database: process.env.PG_DATABASE,
    port: process.env.PG_PORT,
    max: 20,                      // Maximum pool size
    idleTimeoutMillis: 30000,     // Close idle clients after 30 seconds
    connectionTimeoutMillis: 2000 // Return error after 2 seconds if no connection
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

        // Create indexes for frequently queried columns
        await pool.query(`
            CREATE INDEX IF NOT EXISTS idx_expenses_phone_date ON expenses(phone, date);
        `);
        await pool.query(`
            CREATE INDEX IF NOT EXISTS idx_budget_phone ON budget(phone);
        `);

        console.log('✅ Connected to PostgreSQL and ensured tables exist');
    } catch (err) {
        console.error('❌ Error setting up PostgreSQL:', err.message);
    }
})();

module.exports = pool;
