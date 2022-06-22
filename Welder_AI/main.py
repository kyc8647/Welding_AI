from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import logging
from datetime import datetime
import torch
from torch import nn
from sklearn import preprocessing
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, classification_report, f1_score, precision_recall_fscore_support)

from opcua import Client
from opcua import ua

# Excel 데이터 세트 불러오기
welding_data = pd.read_excel("./Weld_field_device/Welding Data_Set_01.xlsx") # 훈련 데이터
welding_test_data = pd.read_excel("./Weld_field_device/Welding Data_Set_Test.xlsx") # 테스트 데이터
Welding_customized = pd.read_excel("./Weld_field_device/Welding Data_Set_custom.xlsx") # 실제 데이터
preprocessed_weld = pd.read_csv("./Weld_field_device/scaled_data.csv")
process_preprocessed = preprocessed_weld.drop("idx", axis=1)
today = datetime.now()
today_serial = today.strftime("%Y%m%d")
# Datetime to int - 데이터베이스에 datetime object는 String으로 저장을 못하기 때문.
workingtime_arr = []
for index, row in Welding_customized.iterrows():
    working_date = datetime.date(row['working date'])
    working_time = row['working clock']
    datetime_obj = datetime.combine(working_date, working_time)
    datetime_int = int(datetime_obj.strftime("%m%d%H%M%S")) #예) 2022-01-11 09:54:34 => 20220111095434 Nopt compatible with Int32 => 111095434
    workingtime_arr.append(datetime_int)

#OPCUA Server connect - OPCUA 서버 연결
# logging.basicConfig(level=logging.DEBUG)
client = Client("opc.tcp://192.168.219.15:4840/")

client.connect()
root = client.get_root_node()

print("Root is", root)
print("childs of root are: ", root.get_children())
print("name of root is", root.get_browse_name())
objects = client.get_objects_node()
print("childs og objects are: ", objects.get_children())


class SubHandler(object):
    """
    Client to subscription. It will receive events from server
    """

    def datachange_notification(self, node, val, data):
        print("Python: New data change event", node, val)

    def event_notification(self, event):
        print("Python: New event", event)


handler = SubHandler()
sub = client.create_subscription(1000, handler)

# Training force, current, Voltage and time
new_welding_data = welding_data.iloc[:, 6:]
new_test_welding_data = welding_test_data.iloc[:, 6:]
new_real_life_weld = Welding_customized.iloc[:, 6:]
measurement_welding_data = pd.DataFrame(new_welding_data)
test_welding_frame = pd.DataFrame(new_test_welding_data)
real_life_custom = pd.DataFrame(new_real_life_weld)

## sklearn의 preprocessing모듈에 들어있는 MinMaxScaler함수를 이용해 정규화 적용
scaler = preprocessing.MinMaxScaler()
scaler.fit(new_welding_data)
scaled_data = scaler.transform(new_welding_data)

scalar_test = preprocessing.MinMaxScaler()
scalar_test.fit(new_test_welding_data)
scaled_test_data = scalar_test.transform(new_test_welding_data)

scaler_realex = preprocessing.MinMaxScaler()
scaler_realex.fit(new_real_life_weld)
scaled_realex_data = scaler_realex.transform(new_real_life_weld)


## AutoEncoder Class
class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AutoEncoder, self).__init__()
        ## initialize
        self.input_size = input_size

        self.output_size = output_size
        ##오토인코더 구현
        self.AutoEncoder = nn.Sequential(
            ## 인코더 부분
            nn.Linear(input_size, hidden_size[0]),
            nn.RReLU(),
            nn.Linear(hidden_size[0], output_size),
            nn.RReLU(),
            ## 디코더 부분
            nn.Linear(output_size, hidden_size[0]),
            nn.RReLU(),
            nn.Linear(hidden_size[0], input_size)
        )

    def forward(self, inputs):
        output = self.AutoEncoder(inputs)

        return output


##기존이 데이터를 텐서 형태로 변환, 그리고 훈련세트와 테스트세트로 나눔
# train_data = torch.Tensor(scaled_data)  ## [:8470] 처음부터 8469번까지의 데이터를 훈련세트로 지정
# test_data = torch.Tensor(scaled_data[8470:])  ## [8470:] 8470번째 데이터부터 끝까지를 테스트세트로 지정
train_data = torch.Tensor(scaled_data)  ## KAMP에서 받은 데이터 전체를 훈련세트로 지정
test_data = torch.Tensor(scaled_test_data)  ## 데이터를 기바능로 커스텀화한 것 중 적은 데이터를 테스트 세트로 지정
real_data = torch.Tensor(scaled_realex_data) ## 실제 있었던 데이터를 AAS기반 데이터 수집으로 지정

## 훈련 하이퍼파라미터
epoch = 50
batch_size = 64
lr = 0.01
## 모델 하이퍼파라미터
input_size = len(train_data[0])
hidden_size = [3]
output_size = 2
## 손실 함수로 제곱근 오차 사용
criterion = nn.MSELoss()
## 매개변수 조정 방식으로 Adam사용
optimizer = torch.optim.Adam
##오토인코더 정의
AutoEncoder = AutoEncoder(input_size, hidden_size, output_size)

epochs_data = []
loss_data = []
## 학습 함수에 대한 정의
def train_net(AutoEncoder, data, criterion, epochs, lr_rate=0.01):
    ## Optimizer에 대한 정의
    optim = optimizer(AutoEncoder.parameters(), lr=lr_rate)
    ## 배치 학습을 시키기 위한 데이터 변환
    data_iter = DataLoader(data, batch_size=batch_size, shuffle=True)
    ## 에포크 학습
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for x in data_iter:
            ## 매개변수 0으로 초기화
            optim.zero_grad()
            output = AutoEncoder(x)
            ## 입력값과 출력값간의 차이인 손실값
            loss = criterion(x, output)
            ## 손실값을 기준으로 매개변수 조정
            loss.backward()
            optim.step()
            running_loss += loss.item()

        ## 각 에포크마다 손실 값 표기
        epochs_data.append(epoch)
        loss_data.append(running_loss)
        print("epoch: {}, loss: {:.2f}".format(epoch, running_loss))
    return AutoEncoder

## 학습 함수를 이용한 오토인코더 학습
AutoEncoder = train_net(AutoEncoder, train_data, criterion, epoch, lr)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_xlim([0, 50])
plt.plot(loss_data)
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train'], loc='upper right')
plt.savefig("./graphdata/KAMP_epochtrain{}.png".format(today_serial))

torch.save(AutoEncoder.state_dict(), './Weld_field_device/welder_autoencoder.pth')

## 훈련세트의 손실값 이용한 임계값 정의
train_loss_chart = []
for data in train_data:
    output = AutoEncoder(data)
    loss = criterion(output, data)
    train_loss_chart.append(loss.item())

threshold = np.mean(train_loss_chart) + np.std(train_loss_chart) * 4
print("Threshold :", threshold)  # 결과는 아래에서 확인 가능하다.

test_loss_chart = []
for data in test_data:
    output = AutoEncoder(data)
    loss = criterion(output, data)
    test_loss_chart.append(loss.item())

outlier = list(test_loss_chart >= threshold)
print(outlier.count(True))

recondataset = []
threshold_data = []
reconerror_data = []
for data in real_data:
    output = AutoEncoder(data)
    recondataset.append(output)
    loss = criterion(output, data)
    threshold_data.append(threshold)
    reconerror_data.append(loss.item())

preprocess_df = pd.DataFrame(recondataset).astype("float")

weldforce_ae = preprocess_df[0]
weldcurrent_ae = preprocess_df[1]
weldvoltage_ae = preprocess_df[2]
weldtime_ae = preprocess_df[3]

weld_ae_df = pd.DataFrame(
    {'datetimeindex':workingtime_arr, 'weld_force_recon': weldforce_ae, 'weld_current_recon': weldcurrent_ae, 'weld_voltatge_recon': weldvoltage_ae,
     'weld_time_recon': weldtime_ae, 'threshold': threshold_data, 'reconstruction_error': reconerror_data})

result = Welding_customized.join(weld_ae_df)
result.to_excel("./Weld_field_device/Preprocessed_welding.xlsx", engine='xlsxwriter')

# welding_data_recon = pd.read_excel("./Weld_field_device/Preprocessed_welding.xlsx")

for index, row in result.iterrows():
    thickness1 = row['Thickness 1(mm)']
    thickness2 = row['Thickness 2(mm)']
    weldforce = row['weld force(bar)']
    weldcurrent = row['weld current(kA)']
    weldvoltage = row['weld Voltage(v)']
    weldtime = row['weld time(ms)']
    weldforcerecon = row['weld_force_recon']
    weldcurrentrecon = row['weld_current_recon']
    weldvoltagerecon = row['weld_voltatge_recon']
    weldtimerecon = row['weld_time_recon']
    weldthreshold = row['threshold']
    weldreconerror = row['reconstruction_error']
    wtindex = row['datetimeindex']

    uploadstatus = row['Machine_Status']
    # print(weldtime)

    wtindexvar = client.get_node('ns=3;s="Weld_datetime_index"')
    wtindexval = wtindex
    wtindexvar.set_value(ua.DataValue(ua.Variant(wtindexval, ua.VariantType.Int32)))

    thickness1var = client.get_node('ns=3;s="Weld_thickness1"')
    thickness1val = thickness1
    thickness1var.set_value(ua.DataValue(ua.Variant(thickness1val, ua.VariantType.Float)))

    thickness2var = client.get_node('ns=3;s="Weld_thickness2"')
    thickness2val = thickness2
    thickness2var.set_value(ua.DataValue(ua.Variant(thickness2val, ua.VariantType.Float)))

    weldforcevar = client.get_node('ns=3;s="Weld_force"')
    weldforceval = weldforce
    weldforcevar.set_value(ua.DataValue(ua.Variant(weldforceval, ua.VariantType.Float)))

    weldcurrentvar = client.get_node('ns=3;s="Weld_current"')
    weldcurrentval = weldcurrent
    weldcurrentvar.set_value(ua.DataValue(ua.Variant(weldcurrentval, ua.VariantType.Float)))

    weldvoltagevar = client.get_node('ns=3;s="Weld_voltage"')
    weldvoltageval = weldvoltage
    weldvoltagevar.set_value(ua.DataValue(ua.Variant(weldvoltageval, ua.VariantType.Float)))

    weldtimevar = client.get_node('ns=3;s="Weld_time"')
    weldtimeval = weldtime
    weldtimevar.set_value(ua.DataValue(ua.Variant(weldtimeval, ua.VariantType.Float)))

    rweldforcevar = client.get_node('ns=3;s="Weld_force_recon"')
    rweldforceval = weldforcerecon
    rweldforcevar.set_value(ua.DataValue(ua.Variant(rweldforceval, ua.VariantType.Float)))

    rweldcurrentvar = client.get_node('ns=3;s="Weld_current_recon"')
    rweldcurrentval = weldcurrentrecon
    rweldcurrentvar.set_value(ua.DataValue(ua.Variant(rweldcurrentval, ua.VariantType.Float)))

    rweldvoltagevar = client.get_node('ns=3;s="Weld_voltage_recon"')
    rweldvoltageval = weldvoltagerecon
    rweldvoltagevar.set_value(ua.DataValue(ua.Variant(rweldvoltageval, ua.VariantType.Float)))

    rweldtimevar = client.get_node('ns=3;s="Weld_time_recon"')
    rweldtimeval = weldtimerecon
    rweldtimevar.set_value(ua.DataValue(ua.Variant(rweldtimeval, ua.VariantType.Float)))

    rweldthresholdvar = client.get_node('ns=3;s="Weld_threshold"')
    rweldthresholdval = weldthreshold
    rweldthresholdvar.set_value(ua.DataValue(ua.Variant(rweldthresholdval, ua.VariantType.Float)))

    rweldreconerrorvar = client.get_node('ns=3;s="Weld_recon_error"')
    rweldreconerrorval = weldreconerror
    rweldreconerrorvar.set_value(ua.DataValue(ua.Variant(rweldreconerrorval, ua.VariantType.Float)))

    suploadvar = client.get_node('ns=3;s="weldin_M_Status"')
    suploadval = uploadstatus
    suploadvar.set_value(ua.DataValue(ua.Variant(suploadval, ua.VariantType.Int32)))

# uploadstatus = 3
# suploadvar = client.get_node("ns=8;i=1020")
# suploadval = uploadstatus
# suploadvar.set_value(ua.Variant(suploadval, ua.VariantType.Int32))

time.sleep(3)

sub.delete()
# client.close_session()
client.disconnect()
