FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python generate_data.py && \
    python generate_labels.py && \
    python train_model.py && \
    python generate_test_data.py && \
    mkdir -p /app/init_data /app/init_model && \
    cp /app/data/subscription_plan_details.csv /app/init_data/ && \
    cp /app/data/test_bank_data.csv            /app/init_data/ && \
    cp /app/data/test_usage_data.csv           /app/init_data/ && \
    cp /app/data/features.csv                  /app/init_data/ && \
    cp /app/model/model.pkl                    /app/init_model/ && \
    cp /app/model/cat_encoder.pkl              /app/init_model/ && \
    cp /app/model/billing_encoder.pkl          /app/init_model/ && \
    cp /app/model/label_encoder.pkl            /app/init_model/

RUN chmod +x /app/startup.sh

EXPOSE 8080

CMD ["/app/startup.sh"]
