from flask import Flask, jsonify
from datetime import datetime
import pytz
from pymongo import MongoClient
import smtplib
import os
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# MongoDB Connection (REPLACE with your actual URI)
MONGO_URI = "mongodb+srv://Soumaditya700:Souma1234@soumaditya-1.ghhhasn.mongodb.net/?appName=Soumaditya-1"

client = MongoClient(MONGO_URI)
db = client["contactDB"]
collection = db["users"]

# Email Config (REPLACE with your details)
EMAIL_USER = "soumadityaghosh700@gmail.com"
EMAIL_PASS = "xhbzbmzwkyoffprz"
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


@app.route('/alert-detection', methods=['GET'])
def send_alert():
    users = list(collection.find({}, {"email": 1, "_id": 0}))

    success = []
    failed = []

    for user in users:
        email = user.get("email")
        if email:
            if send_email(email):
                success.append(email)
            else:
                failed.append(email)

    return jsonify({
        "message": "Emails processed",
        "success": success,
        "failed": failed
    })


if __name__ == '__main__':
    app.run(debug=False)