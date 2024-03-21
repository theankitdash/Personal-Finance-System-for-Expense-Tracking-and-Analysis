const mysql = require('mysql');

// Create a connection to the MySQL server
const connection = mysql.createConnection({
  host: 'localhost',
  user: 'root',
  password: 'Chiku@4009',
  database: 'tutoring-system'
});

// Connect to the MySQL server
connection.connect((err) => {
  if (err) {
    console.error('Error connecting to MySQL:', err);
    return;
  }
  console.log('Connected to MySQL server');
});

// Example query
const query = 'SELECT * FROM your_table';

connection.query(query, (error, results, fields) => {
  if (error) throw error;

  // Process the results here
  console.log('Query results:', results);
});

// Close the connection
connection.end();
