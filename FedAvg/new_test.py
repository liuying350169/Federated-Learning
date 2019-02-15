import numpy as np


# arr = [1,2,3,4,5,6]
# #求均值
# arr_mean = np.mean(arr)
# #求方差
# arr_var = np.var(arr)
# #求标准差
# arr_std = np.std(arr,ddof=1)
# print("平均值为：%f" % arr_mean)
# print("方差为：%f" % arr_var)
# print("标准差为:%f" % arr_std)

num_client = 100
num_params = 450
dict_users = {i: np.array([]) for i in range(num_params)}

#print(len(dict_users))

for i in range(num_params):
     dict_users[i] = np.random.randint(0,1,size=[3,6,5,5])
print(dict_users[i])


a = {i: np.array([]) for i in range(num_client)}
for i in range(num_client):
    a[i]=np.random.randint(0,100,size=[3,6,5,5])



print(a[0][0][0][0][0])
x = []
res_var = []
res_std = []
for j in range(3):
    for k in range(6):
        for m in range(5):
            for n in range(5):
                if(len(x)!=0):
                    res_var.append(np.var(x))
                    res_std.append(np.std(x,ddof=1))
                x = []
                for i in range(num_client):
                    x.append(a[i][j][k][m][n])


print(res_var,len(res_var))
print(res_std,len(res_std))

mean_var = np.mean(res_var)
print(mean_var)
mean_std = np.mean(res_std)
print(mean_std)
#print(dict_users[0])

