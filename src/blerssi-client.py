import os
import random
import numpy
import pandas as pd
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import regression_pb2
from grpc.beta import implementations

dict1 = {'O02': 0,'P01': 1,'P02': 2,'R01': 3,'R02': 4,'S01': 5,'S02': 6,'T01': 7,'U02': 8,'U01': 9,'J03': 10,'K03': 11,'L03': 12,'M03': 13,'N03': 14,'O03': 15,'P03': 16,'Q03': 17,'R03': 18,'S03': 19,'T03': 20,'U03': 21,'U04': 22,'T04': 23,'S04': 24,'R04': 25,'Q04': 26,'P04': 27,'O04': 28,'N04': 29,'M04': 30,'L04': 31,'K04': 32,'J04': 33,'I04': 34,'I05': 35,'J05': 36,'K05': 37,'L05': 38,'M05': 39,'N05': 40,'O05': 41,'P05': 42,'Q05': 43,'R05': 44,'S05': 45,'T05': 46,'U05': 47,'S06': 48,'R06': 49,'Q06': 50,'P06': 51,'O06': 52,'N06': 53,'M06': 54,'L06': 55,'K06': 56,'J06': 57,'I06': 58,'F08': 59,'J02': 60,'J07': 61,'I07': 62,'I10': 63,'J10': 64,'D15': 65,'E15': 66,'G15': 67,'J15': 68,'L15': 69,'R15': 70,'T15': 71,'W15': 72,'I08': 73,'I03': 74,'J08': 75,'I01': 76,'I02': 77,'J01': 78,'K01': 79,'K02': 80,'L01': 81,'L02': 82,'M01': 83,'M02': 84,'N01': 85,'N02': 86,'O01': 87,'I09': 88,'D14': 89,'D13': 90,'K07': 91,'K08': 92,'N15': 93,'P15': 94,'I15': 95,'S15': 96,'U15': 97,'V15': 98,'S07': 99,'S08': 100,'L09': 101,'L08': 102,'Q02': 103,'Q01': 104}

TF_MODEL_SERVER_HOST = os.getenv("TF_MODEL_SERVER_HOST", "127.0.0.1")
TF_MODEL_SERVER_PORT = int(os.getenv("TF_MODEL_SERVER_PORT", 9000))

BLE_RSSI_UL = pd.read_csv('./data/iBeacon_RSSI_Unlabeled.csv')#Unlabeled dataset
COLUMNS = list(BLE_RSSI_UL.columns)
FEATURES = COLUMNS[2:]
LABEL = [COLUMNS[0]]

 
df_predict = pd.read_csv('./data/iBeacon_RSSI_Unlabeled.csv')#Unlabeled dataset
df_predict = df_predict.drop(['date','location'],axis = 1)
df_predict_output = df_predict.copy()
df_predict[FEATURES] = (df_predict[FEATURES] - df_predict[FEATURES].mean())/df_predict[FEATURES].std()
df_predict1 = df_predict.drop(df_predict.index[3190:5191])
df_predict_output1  = df_predict_output.drop(df_predict_output.index[3190:5191])
examples = []
for index, row in df_predict1.iterrows():
    example = tf.train.Example()
    for col, value in row.iteritems():
        example.features.feature[col].float_list.value.append(value)
    examples.append(example)


channel = implementations.insecure_channel(TF_MODEL_SERVER_HOST, TF_MODEL_SERVER_PORT)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = classification_pb2.ClassificationRequest()
request.model_spec.name = "blerssi"
request.model_spec.signature_name = "serving_default"
request.input.example_list.examples.extend(examples)

response = stub.Classify(request, 10.0)



def get_key(val):
    for key, value in dict1.items():
         if val == value:
             return key


outputs = df_predict_output1.copy()
for index, row in outputs.iterrows():
    max_class = max(response.result.classifications[index].classes, key=lambda c: c.score)
    outputs.loc[index, 'Location'] = get_key(int(max_class.label))
    outputs.loc[index, 'Probability'] = max_class.score
	
print(outputs)
