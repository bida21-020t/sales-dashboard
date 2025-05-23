#!/bin/bash
FILE_ID="1NdErQNL4JcqBAQaOlLk8GzZLnzgjS8qp"
OUTPUT="web_interaction_data_with_clusters.csv"
LOCKFILE="download.lock"

# Create lock file to prevent multiple downloads
if [ -f "$LOCKFILE" ]; then
    echo "Download already in progress"
    exit 0
fi

touch "$LOCKFILE"

wget --no-check-certificate \
  "https://drive.google.com/uc?export=download&id=${FILE_ID}" \
  -O "${OUTPUT}"

rm "$LOCKFILE"
echo "✅ Dataset downloaded!"
