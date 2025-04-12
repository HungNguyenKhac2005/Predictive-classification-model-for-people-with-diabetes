# IMPORT THƯ VIỆN

# import thư viện xử lý dữ liệu và thống kế 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# import thư viện trực quan hóa dữ liệu
import matplotlib.pyplot as plt
import seaborn as sns 

# import thư viện lấy báo cáo thống kê dữ liệu cơ bản 
from ydata_profiling import ProfileReport

#import các model cần thiết cho bài toán 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# import thư viện đánh giá mô hình, báo cáo
from sklearn.metrics import classification_report , recall_score , precision_score

# import thư viện giúp chọn mô hình tối ưu và chọn siêu tham số
from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyClassifier
from sklearn.pipeline import Pipeline

# tạo list màu cho bước trực quan
color_list = [
    "red", "blue", "green", "yellow", "purple",
    "orange", "pink", "brown", "gray", "cyan",
    "magenta", "lime", "indigo", "violet", "gold",
    "silver", "navy", "teal", "coral", "maroon"
]

# create def
def my_recall(y_test,y_predict):
    return recall_score(y_test,y_predict,average='binary')

def my_precision(y_test,y_predict):
    return precision_score(y_test,y_predict,average='binary')

# DATA COLLECTION
data = pd.read_csv("C:\DATA\data Việt nguyễn\diabetes.csv")

# STATISTICS
# chú ý : có hai cách chạy để xem thông tin dữ liệu 
# c1 : sử dụng chạy trên terminal bình thường ( các câu lệnh print())
# c2 : sử dụng chạy bằng cách bôi đen câu lệnh cần chạy , chuột phải chọn run in interactive window, 
# sau đó chọn run selection/line in interactive window ( với cách 2 phải chọn tất cả mã và chạy bằng cách chuột phải
# trước một lần sau đó mới chạy từng câu lệnh)
# Khuyến nghị nên kết hợp linh hoạt cả hai cách 

# thông tin cơ bản về data
print(data) 
data
print(data.head(5))
data.head()
print(data.tail(5))
data.tail()
print(data.info())
data.info()
print(data.shape)
data.shape

# kiểm tra các cột có giá trị thiếu và số lượng thiếu của mỗi cột là bao nhiêu
data.columns[data.isnull().any()]
print(data.columns[data.isnull().any()])
total_null = data.isnull().sum()
print(total_null)
total_null

ten_cot = total_null.index.to_list()
null = total_null.values.tolist()
print(ten_cot)

# tính phần trăm số lượng null trong từng cột
percent_null = []
for i in range(len(ten_cot)):
    percent = (null[i]/len(data))*100
    percent_null.append(percent)
for i,j in zip(ten_cot,percent_null):
    print("Tên cột: " + i + " , " + "Phần trăm null: " + str(j))

# lấy ra tổng số giá trị bị trùng lặp
data.duplicated().sum()

# thống kê cơ bản về dữ liệu
data.describe()

# số class cần dự đoán và tổng số lượng của chúng trong dataset
data['Outcome'].value_counts()

# Lấy ra báo cáo thống kê tổng quát về dữ liệu và chuyển sang file.HTML
# profile = ProfileReport(data,title="Báo cáo",explorative=True)
# profile.to_file("bao_cao.html")

# DATA PREPROCESSING

# tính toán tứ vị phân
Q1 = data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]].quantile(0.25)
Q1
print("Q1" + str(Q1))
Q3 = data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]].quantile(0.75)
Q3
print("Q3" + str(Q3))
IQR = Q3-Q1
IQR
print("IQR" + str(IQR))

# tính toán giới hạn trên và giới hạn dưới
lower_bound = Q1 - 1.5*IQR
lower_bound = lower_bound.reset_index()
lower_bound.rename(columns={0 : "IQR"},inplace=True)
lower_bound
print("lower_bound" + str(lower_bound))
upper_bound = Q3 + 1.5*IQR
upper_bound = upper_bound.reset_index()
upper_bound.rename(columns={0 : "IQR"},inplace=True)
upper_bound
print("upper_bound" + str(upper_bound))


for i in range(len(lower_bound)):
    outlier = data[(data[lower_bound['index'][i]] <= lower_bound['IQR'][0]) & (data[upper_bound['index'][i]] <= upper_bound["IQR"][0])]
    print(upper_bound['index'][i] + " " + str(len(outlier)))

# Chia dữ liệu ra thành imdependent và dependent
x = data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
x
y = data[['Outcome']]
y     

# Chia dữ liệu ra thành bộ train và bộ test
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train
x_test
y_train
y_test

# scaler dữ liệu về dạng phân phối chuẩn
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_train
x_test = scaler.transform(x_test)
x_test

# DATA VISUALIZATION

# vẽ biểu đồ phân phối của các biến numeric
fig,ax = plt.subplots(ncols=4,nrows=2,figsize=(10,5))
index=0
for i in range(2):
    for j in range(4):
        sns.histplot(data[upper_bound['index'][index]],bins=20,ax=ax[i,j],color=color_list[index],kde=True)
        # ax[i][j].hist(data[upper_bound['index'][index]],bins=20,color=color_list[index])
        index+=1
fig.suptitle("Biểu đồ Phân phối của các biến numerical")
plt.show()

# vẽ biểu đồ tương quan heapmap của các biến numeric
corr = data.corr()
sns.heatmap(corr,annot=True)
plt.title('Biểu đồ tương quan giữa các biến numerical')
plt.show()

# vẽ biểu đồ scatter của các biến numeric với target
fig,ax = plt.subplots(ncols=4,nrows=2,figsize=(10,8))
index=0
for i in range(2):
    for j in range(4):
        sns.scatterplot(x = upper_bound['index'][index],y = 'Outcome',data=data,ax=ax[i,j],color=color_list[index])
        # ax[i][j].hist(data[upper_bound['index'][index]],bins=20,color=color_list[index])
        index+=1
fig.suptitle("Biểu đồ Tương quan của các biến numerical với target")
plt.show()

sns.pairplot(data,hue='Outcome')
plt.show()


# MODEL BUIDING AND TRAINING

# sử dụng thư viện lazy_predict để xem mô hình nào dự đoán tốt nhất
model_dict = [
    "SVM" ,
    "Logistics" ,
    "RandomForrest" ,
    "K-NN", 
]
clf = LazyClassifier(verbose=0,ignore_warnings=True,custom_metric=my_recall,predictions=True)
models , predict = clf.fit(x_train,x_test,y_train,y_test)
df = models.reset_index()
df
df[df["Model"].isin(["RandomForestClassifier","DecisionTreeClassifier","LogisticRegression","SVC","KNeighborsClassifier"])]

# thử cho mô hình dự đoán trên cả bộ train và bộ test để xem có bị overfitting không
# model = SVC(random_state=42)
# model.fit(x_train,y_train)
# test_predict = model.predict(x_test)
# recall_test = recall_score(y_test,test_predict)
# recall_test
# train_predict = model.predict(x_train)
# recall_train = recall_score(y_train,train_predict)
# recall_train

# sử dụng pipeline và gridsearch để tạo đường dẫn và tìm ra siêu tham số tốt nhất cho mô hình
pipe = Pipeline([('model',SVC())])

param_grid = [
    {
        'model' : [SVC()],
        'model__C' : [0.1, 1, 10 ],
        'model__kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
    },
    {
        'model' : [LogisticRegression()],
        'model__C' : [0.1,1,10],
        'model__penalty' : ['l1','l2'],
        'model__solver' : ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]

    }
]

gridsearch = GridSearchCV(pipe,param_grid=param_grid,cv=5,scoring='precision',verbose=1,return_train_score=True)
gridsearch.fit(x_train,y_train)

# pipe.get_params().keys()

# mô hình tốt nhất
gridsearch.best_estimator_

# phần trăm dự đoán cao nhất
gridsearch.best_score_

# siêu tham số tốt nhất
gridsearch.best_params_

# xây dựng mô hình với các tham số đã tìm được
model = SVC(C=10,kernel='rbf',probability=True)
model.fit(x_train,y_train)

y_pre = model.predict(x_test)
print(classification_report(y_test,y_pre))

# MODEL EVALUATION

# mô hình này cần recall nên phải giảm ngưỡng thresold
y_proba = model.predict_proba(x_test)[:, 1]
print(y_proba)

threshold = 0.1
y_pred_custom = (y_proba >= threshold).astype(int)
print(y_pred_custom)
print(f"\n==> Threshold {threshold}:")
print(classification_report(y_test, y_pred_custom))

# MODEL DEPLOYMENT


























