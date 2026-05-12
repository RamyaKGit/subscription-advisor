#!/bin/bash
# Start Streamlit immediately so Cloud Run health check passes
streamlit run app.py \
    --server.port=8080 \
    --server.address=0.0.0.0 \
    --server.headless=true &

STREAMLIT_PID=$!

# Copy data files to GCS mounted /app/data/ if not already there
for FILE in subscription_plan_details.csv test_bank_data.csv test_usage_data.csv features.csv; do
    [ ! -f "/app/data/$FILE" ] && cp "/app/init_data/$FILE" "/app/data/$FILE" && echo "Copied $FILE to /app/data/"
done

# Copy model files to GCS mounted /app/model/ if not already there
for FILE in model.pkl cat_encoder.pkl billing_encoder.pkl label_encoder.pkl; do
    [ ! -f "/app/model/$FILE" ] && cp "/app/init_model/$FILE" "/app/model/$FILE" && echo "Copied $FILE to /app/model/"
done

echo "Startup complete."
wait $STREAMLIT_PID
