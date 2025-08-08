# Use an official Node.js runtime as the base image
FROM node:22

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy package.json and package-lock.json if available
COPY package*.json ./

# Install Node.js dependencies
RUN npm install

# Copy wait-for-it and make executable
COPY wait-for-it.sh /usr/src/app/wait-for-it.sh
RUN chmod +x /usr/src/app/wait-for-it.sh

# Copy the rest of the application code into the container
COPY . .

# Expose port 3000 to the outside world
EXPOSE 3000

# Start app only after MySQL is ready
CMD ["./node-backend/wait-for-it.sh", "mysql:3306", "--", "node", "node-backend/server.js"]
