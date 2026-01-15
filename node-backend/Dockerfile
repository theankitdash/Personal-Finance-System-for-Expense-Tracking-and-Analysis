# Use an official Node.js runtime as the base image
FROM node:22-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy package.json and package-lock.json if available
COPY package*.json ./

# Set NODE_ENV to production
ENV NODE_ENV=production

# Install Node.js dependencies (use install to handle package.json changes)
RUN npm install --only=production

# Copy the rest of the application code into the container
COPY . .

# Expose port 3000 to the outside world
EXPOSE 3000

# Start the application
CMD ["node", "server.js"]
