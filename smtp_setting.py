import smtplib
from email.mime.text import MIMEText
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart


def send_email():
    text = "인스타 크롤링 끝났음"
    # msg = MIMEText(text)
    msg = MIMEMultipart()
    msg['Subject'] = "크롤링 결과 파일"
    msg['From'] = ''
    msg['To'] = ''
    msg.attach(MIMEText(text, _charset='utf-8'))
    print(msg.as_string())

    with open('./data/post_data.csv', 'rb') as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
    with open('./data/user_data.csv', 'rb') as f:
        part1 = MIMEBase('application', 'octet-stream')
        part1.set_payload(f.read())
    encoders.encode_base64(part)
    encoders.encode_base64(part1)

    part1.add_header('Content-Disposition', 'attachment; filename="user_data.csv"')
    msg.attach(part1)
    part.add_header('Content-Disposition', 'attachment; filename="post_data.csv"')
    msg.attach(part)

    s = smtplib.SMTP('smtp.naver.com', 587)
    s.starttls()  # TLS 보안 처리
    s.login('', '')  # 네이버로그인
    s.sendmail('', '', msg.as_string())
    s.close()
