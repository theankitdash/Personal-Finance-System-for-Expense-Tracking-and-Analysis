// Environment variable validation utility
const requiredEnvVars = [
    'PG_HOST', 'PG_USER', 'PG_PASSWORD', 'PG_DATABASE', 'PG_PORT',
    'SESSION_SECRET', 'JWT_SECRET', 'PYTHON_API_URL'
];

function validateEnv() {
    requiredEnvVars.forEach(varName => {
        if (!process.env[varName]) {
            console.error(`❌ Missing required environment variable: ${varName}`);
            process.exit(1);
        }
    });
    console.log('✅ Environment variables validated');
}

module.exports = validateEnv;
