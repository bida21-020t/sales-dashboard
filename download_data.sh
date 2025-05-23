#!/bin/bash

FILE_ID="1NdErQNL4JcqBAQaOlLk8GzZLnzgjS8qp"
OUTPUT="web_interaction_data_with_clusters.csv"
LOCKFILE="download.lock"
MAX_RETRIES=3
RETRY_DELAY=5

# Check if file already exists
if [ -f "$OUTPUT" ]; then
    echo "Dataset already exists at $OUTPUT"
    exit 0
fi

# Create lock file to prevent multiple downloads
if [ -f "$LOCKFILE" ]; then
    echo "Download already in progress"
    exit 0
fi

touch "$LOCKFILE"

# Function to attempt download
download_file() {
    for i in $(seq 1 $MAX_RETRIES); do
        echo "Attempt $i of $MAX_RETRIES..."
        wget --no-check-certificate \
            "https://drive.google.com/uc?export=download&id=${FILE_ID}" \
            -O "${OUTPUT}"
        
        if [ $? -eq 0 ]; then
            # Verify the downloaded file is not empty and is a CSV
            if [ -s "$OUTPUT" ] && file "$OUTPUT" | grep -q "text"; then
                echo "✅ Dataset downloaded successfully!"
                rm "$LOCKFILE"
                exit 0
            else
                echo "Downloaded file appears invalid, retrying..."
                rm -f "$OUTPUT"
            fi
        fi
        
        if [ $i -lt $MAX_RETRIES ]; then
            echo "Waiting $RETRY_DELAY seconds before retry..."
            sleep $RETRY_DELAY
        fi
    done
    
    echo "❌ Failed to download dataset after $MAX_RETRIES attempts"
    rm "$LOCKFILE"
    exit 1
}

# Execute download
download_file
