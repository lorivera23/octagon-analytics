import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


def build_report(new_event_count, new_version, new_auc, prod_auc, decision):
    improvement = new_auc - prod_auc
    return f"""
UFC Weekly Pipeline Report — {datetime.now().strftime('%B %d, %Y')}
================================================
New events scraped:  {new_event_count}
New model version:   v{new_version}
New AUC:             {new_auc:.4f}
Production AUC:      {prod_auc:.4f}
Improvement:         {improvement:+.4f}
Decision:            {decision}
"""


def send_email(subject, body, sender, receiver, password):
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())