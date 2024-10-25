# Use an official Node.js runtime as the base image
FROM node:20

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy package.json and package-lock.json if available
COPY package*.json ./

# Install Node.js dependencies
RUN npm install

# Install Python 3, pip, and the venv package, then clean up to reduce image size
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Set up a virtual environment for Python packages
RUN python3 -m venv /usr/src/app/venv

# Copy the requirements.txt file and install Python dependencies in the virtual environment
COPY requirements.txt .
RUN /usr/src/app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Update PATH to prioritize the virtual environment
ENV PATH="/usr/src/app/venv/bin:$PATH"

# Copy the rest of the application code into the container
COPY . .

# Expose port 3000 to the outside world
EXPOSE 3000

# Command to run the app (start Node.js server)
CMD ["node", "server.js"]
