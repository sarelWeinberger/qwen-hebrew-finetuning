FROM python:3.9-slim

# התקנת dependencies מערכת
RUN apt-get update && apt-get install -y \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# יצירת דירקטורי עבודה
WORKDIR /app

# העתקת requirements והתקנה
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# העתקת קבצי הפרויקט
COPY worker.py .
COPY .env .

# הגדרת משתני סביבה
ENV AWS_DEFAULT_REGION=us-east-1

# פקודת ברירת מחדל
CMD ["python", "worker.py", "--prefix", "Geektime", "--part", "0", "--dataset", "geektime"]