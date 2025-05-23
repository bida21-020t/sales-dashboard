#!/bin/bash
# Google Drive file ID
FILE_ID="1NdErQNL4JcqBAQaOlLk8GzZLnzgjS8qp"
OUTPUT="web_interaction_data_with_clusters.csv"

wget --no-check-certificate \
  "https://drive.google.com/uc?export=download&id=${FILE_ID}" \
  -O "${OUTPUT}"
echo "✅ Dataset downloaded!"
