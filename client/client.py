# -*- coding: utf-8 -*-
import socket
import json
import sys
import time
from datetime import datetime
import pytz

########## Change only this

# Group secret
secret = "g10nxdu"
# Group port, change the last two digits to your group number
port = 39010
# Path to your python file containing the prediction
#sys.path.append("/ai/predict")
#sys.path.append("/home/felixtsoi/group10_final_0209")
# Module(s) to perform prediction
from FinalPrediction_new import result_list
# Functions that are used to get your predictions
#def predict_hour():
#    return prediction.hour()
predict_hour=str(result_list[0])
#def predict_day():   
#    return prediction.day()
predict_day=str(result_list[1])
#def predict_week():
#    return prediction.week()
predict_week=str(result_list[2])

##########

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

host = socket.gethostbyname(socket.gethostname() + '.local') 

s.bind((host,port))
s.listen(5)


while True:
    conn, addr = s.accept()	# accept the connection
    conn.close()
    ts = str(datetime.now(pytz.timezone("Europe/Berlin")))[:-13]
    if addr[0] == "129.187.240.34":
        message = {"secret": secret, "time": ts, "hour": round(predict_hour(),4), "day": round(predict_day(),4), "week": round(predict_week(),4)}
        data = json.dumps(message)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print("Connecting to server.")
            sock.connect(("129.187.240.34", 39000))
            print("Sending data.")
            sock.sendall(bytes(data,encoding="utf-8"))
            print("Sent: {}".format(data))
            print("Data successfully sent.")
        except:
            print("Failed, retrying.")
            time.sleep(1)
        finally:
            print("Closing connection.")
            sock.close()
