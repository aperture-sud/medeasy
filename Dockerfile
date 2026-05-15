FROM node:18-alpine

# Create app directory
WORKDIR /app

# Install app dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy the rest of the source code
COPY . .

# Set environment
ENV NODE_ENV=production

# Expose the port Railway will use
EXPOSE 3000

# Run the app
CMD ["npm", "start"]
