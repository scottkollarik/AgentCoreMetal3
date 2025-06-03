#!/bin/bash

echo "🔄 Starting deployment process..."

# Stop and remove existing containers
echo "📥 Bringing down existing containers..."
docker compose down

# Remove any dangling images
echo "🧹 Cleaning up unused images..."
docker image prune -f

# Build and start containers
echo "🚀 Building and starting containers..."
docker compose up --build -d

# Check if containers are running
echo "🔍 Checking container status..."
docker compose ps

echo "✅ Deployment complete!" 