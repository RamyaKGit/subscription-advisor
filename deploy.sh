#!/bin/bash
# ── deploy.sh — Build, push and deploy to Cloud Run ──────────────────────────
# Run this from your subscription_advisor folder:
#   chmod +x deploy.sh
#   ./deploy.sh

set -e  # Stop on any error

# ── Config — update these before running ─────────────────────────────────────
PROJECT_ID="your-project-id"          # e.g. subscription-advisor-123456
REGION="us-central1"
IMAGE_NAME="subscription-advisor"
BUCKET_NAME="subscription-advisor-data"

# ── Derived values ────────────────────────────────────────────────────────────
IMAGE_URL="gcr.io/${PROJECT_ID}/${IMAGE_NAME}"

echo "▶ Step 1: Authenticating with GCP..."
gcloud auth configure-docker

echo "▶ Step 2: Building Docker image..."
docker build --platform linux/amd64 -t ${IMAGE_URL} .

echo "▶ Step 3: Pushing image to Google Container Registry..."
docker push ${IMAGE_URL}

echo "▶ Step 4: Creating GCS bucket for persistent data (if not exists)..."
gsutil mb -p ${PROJECT_ID} -l ${REGION} gs://${BUCKET_NAME} 2>/dev/null || echo "Bucket already exists, skipping."

echo "▶ Step 5: Deploying to Cloud Run with GCS volume mount..."
gcloud run deploy ${IMAGE_NAME} \
  --image ${IMAGE_URL} \
  --region ${REGION} \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --add-volume name=data-vol,type=cloud-storage,bucket=${BUCKET_NAME} \
  --add-volume-mount volume=data-vol,mount-path=/app/data \
  --project ${PROJECT_ID}

echo "✅ Deployment complete!"
echo "Your app is live at the URL shown above."
