import vonage
def send_msg():
    client = vonage.Client(key="0f236922", secret="IYe8RRcortpNhS9e")
    sms = vonage.Sms(client)
    responseData = sms.send_message(
        {
            "from": "Vonage APIs",
            "to": "527712711009",  
            "text": "Se a detectado una persona",
        }
    )

    if responseData["messages"][0]["status"] == "0":
        print("Message sent successfully.")
    else:
        print(f"Message failed with error: {responseData['messages'][0]['error-text']}")
if __name__=='__main__':
    send_msg()