# import numpy as np
#
# import matplotlib
# #matplotlib.use('Agg')
# import matplotlib.pyplot as plt
#
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def f(t):
#     s1 = np.sin(2*np.pi*t)
#     e1 = np.exp(-t)
#     return np.multiply(s1, e1)
#
# t1 = np.arange(0.0, 5.0, 0.1)
# t2 = np.arange(0.0, 5.0, 0.02)
#
# fig, ax = plt.subplots()
# plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
# plt.text(3.0, 0.6, 'f(t) = exp(-t) sin(2 pi t)')
# ttext = plt.title('Fun with text!')
# ytext = plt.ylabel('Damped oscillation')
# xtext = plt.xlabel('time (s)')
#
# plt.setp(ttext, size='large', color='r', style='italic')
# plt.setp(xtext, size='medium', name=['Courier', 'DejaVu Sans Mono'],
#      weight='bold', color='g')
# plt.setp(ytext, size='medium', name=['Helvetica', 'DejaVu Sans'],
#      weight='light', color='b')
# plt.show()
#
# import matplotlib.pyplot as plt
# import matplotlib.cbook as cbook
#
# fname = cbook.get_sample_data('msft.csv', asfileobj=False)
# fname2 = cbook.get_sample_data('data_x_x2_x3.csv', asfileobj=False)
#
# # test 1; use ints
# plt.plotfile(fname, (0, 5, 6))
#
# # test 2; use names
# plt.plotfile(fname, ('date', 'volume', 'adj_close'))
#
# # test 3; use semilogy for volume
# plt.plotfile(fname, ('date', 'volume', 'adj_close'),
#              plotfuncs={'volume': 'semilogy'})
#
# # test 4; use semilogy for volume
# plt.plotfile(fname, (0, 5, 6), plotfuncs={5: 'semilogy'})
#
# # test 5; single subplot
# plt.plotfile(fname, ('date', 'open', 'high', 'low', 'close'), subplots=False)
#
# # test 6; labeling, if no names in csv-file
# plt.plotfile(fname2, cols=(0, 1, 2), delimiter=' ',
#              names=['$x$', '$f(x)=x^2$', '$f(x)=x^3$'])
#
# # test 7; more than one file per figure--illustrated here with a single file
# plt.plotfile(fname2, cols=(0, 1), delimiter=' ')
# plt.plotfile(fname2, cols=(0, 2), newfig=False,
#              delimiter=' ')  # use current figure
# plt.xlabel(r'$x$')
# plt.ylabel(r'$f(x) = x^2, x^3$')
#
# # test 8; use bar for volume
# plt.plotfile(fname, (0, 5, 6), plotfuncs={5: 'bar'})
#
# plt.show()

import numpy as np
arr = [1,2,3,4,5,6]
#求均值
arr_mean = np.mean(arr)
#求方差
arr_var = np.var(arr)
#求标准差
arr_std = np.std(arr,ddof=1)
print("平均值为：%f" % arr_mean)
print("方差为：%f" % arr_var)
print("标准差为:%f" % arr_std)

num_client = 100
num_params = 450
dict_users = {i: np.array([]) for i in range(num_params)}

#print(len(dict_users))

for i in range(num_params):
     dict_users[i] = np.random.randint(0,1,size=[3,6,5,5])
print(dict_users[i])


a = {i: np.array([]) for i in range(num_client)}
for i in range(num_client):
    a[i] = np.random.randint(0,100,size=[3,6,5,5])


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
print(a[0].flatten(),len(a[0].flatten()))
print(res_var,len(res_var))
print(res_std,len(res_std))

mean_var = np.mean(res_var)
print(mean_var)
mean_std = np.mean(res_std)
print(mean_std)

import tensorflow as tf
img1 = tf.constant(value=[[[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]]]],dtype=tf.float32)
img2 = tf.constant(value=[[[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]]]],dtype=tf.float32)
img = tf.concat(values=[img1,img2],axis=3)
sess=tf.Session()
#sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer())
print(img)
print("out1=",type(img))


#转化为numpy数组
img_numpy=img.eval(session=sess)
print(img_numpy)
print(img_numpy.flatten(),len(img_numpy.flatten()))
print("out2=",type(img_numpy))


#转化为tensor
img_tensor= tf.convert_to_tensor(img_numpy)
print(img_tensor)
print("out2=",type(img_tensor))

#print(dict_users[0])

