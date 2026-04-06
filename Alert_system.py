from flask import Flask, jsonify
from datetime import datetime
import pytz
from pymongo import MongoClient
import smtplib
import os
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

print("MONGO_URI:", os.getenv("MONGO_URI"))
print("EMAIL_USER:", os.getenv("EMAIL_USER"))

app = Flask(__name__)

# MongoDB Connection
client = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=5000)
db = client["contactDB"]
collection = db["users"]

# Email Config
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
def send_email(to_email):
    subject = "🚨 Alert Notification"
    ist = pytz.timezone('Asia/Kolkata')
    body = f"Suspicious activity detected at {datetime.now(ist).strftime('%d-%m-%Y %H:%M:%S')}."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = to_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email to {to_email}: {e}")
        return False


@app.route('/test')
def test():
    return "OK"


@app.route('/alert-detection', methods=['GET'])
def send_alert():
    print("Fetching users from MongoDB", flush=True)
    try:
        users = list(collection.find({}, {"email": 1, "_id": 0}))
        print(f"Found {len(users)} users", flush=True)
    except Exception as e:
        print(f"Error fetching users: {e}", flush=True)
        return jsonify({"error": "Database connection failed"}), 500

    success = []
    failed = []

    for user in users:
        email = user.get("email")
        if email:
            print(f"Sending email to {email}", flush=True)
            if send_email(email):
                success.append(email)
            else:
                failed.append(email)

    print("Processing complete", flush=True)
    return jsonify({
        "message": "Emails processed",
        "success": success,
        "failed": failed
    })


if __name__ == '__main__':
    app.run(debug=False)