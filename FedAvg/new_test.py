# # import numpy as np
# #
# # import matplotlib
# # #matplotlib.use('Agg')
# # import matplotlib.pyplot as plt
# #
# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# #
# # def f(t):
# #     s1 = np.sin(2*np.pi*t)
# #     e1 = np.exp(-t)
# #     return np.multiply(s1, e1)
# #
# # t1 = np.arange(0.0, 5.0, 0.1)
# # t2 = np.arange(0.0, 5.0, 0.02)
# #
# # fig, ax = plt.subplots()
# # plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
# # plt.text(3.0, 0.6, 'f(t) = exp(-t) sin(2 pi t)')
# # ttext = plt.title('Fun with text!')
# # ytext = plt.ylabel('Damped oscillation')
# # xtext = plt.xlabel('time (s)')
# #
# # plt.setp(ttext, size='large', color='r', style='italic')
# # plt.setp(xtext, size='medium', name=['Courier', 'DejaVu Sans Mono'],
# #      weight='bold', color='g')
# # plt.setp(ytext, size='medium', name=['Helvetica', 'DejaVu Sans'],
# #      weight='light', color='b')
# # plt.show()
# #
# # import matplotlib.pyplot as plt
# # import matplotlib.cbook as cbook
# #
# # fname = cbook.get_sample_data('msft.csv', asfileobj=False)
# # fname2 = cbook.get_sample_data('data_x_x2_x3.csv', asfileobj=False)
# #
# # # test 1; use ints
# # plt.plotfile(fname, (0, 5, 6))
# #
# # # test 2; use names
# # plt.plotfile(fname, ('date', 'volume', 'adj_close'))
# #
# # # test 3; use semilogy for volume
# # plt.plotfile(fname, ('date', 'volume', 'adj_close'),
# #              plotfuncs={'volume': 'semilogy'})
# #
# # # test 4; use semilogy for volume
# # plt.plotfile(fname, (0, 5, 6), plotfuncs={5: 'semilogy'})
# #
# # # test 5; single subplot
# # plt.plotfile(fname, ('date', 'open', 'high', 'low', 'close'), subplots=False)
# #
# # # test 6; labeling, if no names in csv-file
# # plt.plotfile(fname2, cols=(0, 1, 2), delimiter=' ',
# #              names=['$x$', '$f(x)=x^2$', '$f(x)=x^3$'])
# #
# # # test 7; more than one file per figure--illustrated here with a single file
# # plt.plotfile(fname2, cols=(0, 1), delimiter=' ')
# # plt.plotfile(fname2, cols=(0, 2), newfig=False,
# #              delimiter=' ')  # use current figure
# # plt.xlabel(r'$x$')
# # plt.ylabel(r'$f(x) = x^2, x^3$')
# #
# # # test 8; use bar for volume
# # plt.plotfile(fname, (0, 5, 6), plotfuncs={5: 'bar'})
# #
# # plt.show()
#
# import numpy as np
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
#
# num_client = 100
# num_params = 450
# dict_users = {i: np.array([]) for i in range(num_params)}
#
# #print(len(dict_users))
#
# for i in range(num_params):
#      dict_users[i] = np.random.randint(0,1,size=[3,6,5,5])
# print(dict_users[i])
#
#
# a = {i: np.array([]) for i in range(num_client)}
# for i in range(num_client):
#     a[i] = np.random.randint(0,100,size=[3,6,5,5])
#
#
# print(a[0][0][0][0][0])
# x = []
# res_var = []
# res_std = []
# for j in range(3):
#     for k in range(6):
#         for m in range(5):
#             for n in range(5):
#                 if(len(x)!=0):
#                     res_var.append(np.var(x))
#                     res_std.append(np.std(x,ddof=1))
#                 x = []
#                 for i in range(num_client):
#                     x.append(a[i][j][k][m][n])
# print(a[0].flatten(),len(a[0].flatten()))
# print(res_var,len(res_var))
# print(res_std,len(res_std))
#
# mean_var = np.mean(res_var)
# print(mean_var)
# mean_std = np.mean(res_std)
# print(mean_std)
#
# import tensorflow as tf
# img1 = tf.constant(value=[[[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]]]],dtype=tf.float32)
# img2 = tf.constant(value=[[[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]]]],dtype=tf.float32)
# img = tf.concat(values=[img1,img2],axis=3)
# sess=tf.Session()
# #sess.run(tf.initialize_all_variables())
# sess.run(tf.global_variables_initializer())
# print(img)
# print("out1=",type(img))
#
#
# #转化为numpy数组
# img_numpy=img.eval(session=sess)
# print(img_numpy)
# print(img_numpy.flatten(),len(img_numpy.flatten()))
# print("out2=",type(img_numpy))
#
#
# #转化为tensor
# img_tensor= tf.convert_to_tensor(img_numpy)
# print(img_tensor)
# print("out2=",type(img_tensor))
#
# #print(dict_users[0])
#
"""
Copyright (c) 2017, Gavin Weiguang Ding
All rights reserved.
Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

NumDots = 4
NumConvMax = 8
NumFcMax = 20
White = 1.
Light = 0.7
Medium = 0.5
Dark = 0.3
Darker = 0.15
Black = 0.


def add_layer(patches, colors, size=(24, 24), num=5,
              top_left=[0, 0],
              loc_diff=[3, -3],
              ):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    for ind in range(num):
        patches.append(Rectangle(loc_start + ind * loc_diff, size[1], size[0]))
        if ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)


def add_layer_with_omission(patches, colors, size=(24, 24),
                            num=5, num_max=8,
                            num_dots=4,
                            top_left=[0, 0],
                            loc_diff=[3, -3],
                            ):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    this_num = min(num, num_max)
    start_omit = (this_num - num_dots) // 2
    end_omit = this_num - start_omit
    start_omit -= 1
    for ind in range(this_num):
        if (num > num_max) and (start_omit < ind < end_omit):
            omit = True
        else:
            omit = False

        if omit:
            patches.append(
                Circle(loc_start + ind * loc_diff + np.array(size) / 2, 0.5))
        else:
            patches.append(Rectangle(loc_start + ind * loc_diff,
                                     size[1], size[0]))

        if omit:
            colors.append(Black)
        elif ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)


def add_mapping(patches, colors, start_ratio, end_ratio, patch_size, ind_bgn,
                top_left_list, loc_diff_list, num_show_list, size_list):

    start_loc = top_left_list[ind_bgn] \
        + (num_show_list[ind_bgn] - 1) * np.array(loc_diff_list[ind_bgn]) \
        + np.array([start_ratio[0] * (size_list[ind_bgn][1] - patch_size[1]),
                    - start_ratio[1] * (size_list[ind_bgn][0] - patch_size[0])]
                   )




    end_loc = top_left_list[ind_bgn + 1] \
        + (num_show_list[ind_bgn + 1] - 1) * np.array(
            loc_diff_list[ind_bgn + 1]) \
        + np.array([end_ratio[0] * size_list[ind_bgn + 1][1],
                    - end_ratio[1] * size_list[ind_bgn + 1][0]])


    patches.append(Rectangle(start_loc, patch_size[1], -patch_size[0]))
    colors.append(Dark)
    patches.append(Line2D([start_loc[0], end_loc[0]],
                          [start_loc[1], end_loc[1]]))
    colors.append(Darker)
    patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]],
                          [start_loc[1], end_loc[1]]))
    colors.append(Darker)
    patches.append(Line2D([start_loc[0], end_loc[0]],
                          [start_loc[1] - patch_size[0], end_loc[1]]))
    colors.append(Darker)
    patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]],
                          [start_loc[1] - patch_size[0], end_loc[1]]))
    colors.append(Darker)



def label(xy, text, xy_off=[0, 4]):
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
             family='sans-serif', size=8)


if __name__ == '__main__':

    fc_unit_size = 2
    layer_width = 40
    flag_omit = True

    patches = []
    colors = []

    fig, ax = plt.subplots()


    ############################
    # conv layers
    size_list = [(32, 32), (28, 28), (14, 14), (10, 10), (5, 5)]
    num_list = [3, 6, 6, 16, 16]
    x_diff_list = [0, layer_width, layer_width, layer_width, layer_width]
    text_list = ['Inputs'] + ['Feature\nmaps'] * (len(size_list) - 1)
    loc_diff_list = [[3, -3]] * len(size_list)

    num_show_list = list(map(min, num_list, [NumConvMax] * len(num_list)))
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]

    for ind in range(len(size_list)-1,-1,-1):
        if flag_omit:
            add_layer_with_omission(patches, colors, size=size_list[ind],
                                    num=num_list[ind],
                                    num_max=NumConvMax,
                                    num_dots=NumDots,
                                    top_left=top_left_list[ind],
                                    loc_diff=loc_diff_list[ind])
        else:
            add_layer(patches, colors, size=size_list[ind],
                      num=num_show_list[ind],
                      top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}@{}x{}'.format(
            num_list[ind], size_list[ind][0], size_list[ind][1]))

    ############################
    # in between layers
    start_ratio_list = [[0.4, 0.5], [0.4, 0.8], [0.4, 0.5], [0.4, 0.8]]
    end_ratio_list = [[0.4, 0.5], [0.4, 0.8], [0.4, 0.5], [0.4, 0.8]]
    patch_size_list = [(5, 5), (2, 2), (5, 5), (2, 2)]
    ind_bgn_list = range(len(patch_size_list))
    text_list = ['Convolution', 'Max-pooling', 'Convolution', 'Max-pooling']

    for ind in range(len(patch_size_list)):
        add_mapping(
            patches, colors, start_ratio_list[ind], end_ratio_list[ind],
            patch_size_list[ind], ind,
            top_left_list, loc_diff_list, num_show_list, size_list)
        label(top_left_list[ind], text_list[ind] + '\n{}x{} kernel'.format(
            patch_size_list[ind][0], patch_size_list[ind][1]), xy_off=[26, -65]
        )


    ############################
    # fully connected layers
    size_list = [(fc_unit_size, fc_unit_size)] * 3
    num_list = [120, 84, 10]
    num_show_list = list(map(min, num_list, [NumFcMax] * len(num_list)))
    x_diff_list = [sum(x_diff_list) + layer_width, layer_width, layer_width]
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]
    loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)
    text_list = ['Hidden\nunits'] * (len(size_list) - 1) + ['Outputs']

    for ind in range(len(size_list)):
        if flag_omit:
            add_layer_with_omission(patches, colors, size=size_list[ind],
                                    num=num_list[ind],
                                    num_max=NumFcMax,
                                    num_dots=NumDots,
                                    top_left=top_left_list[ind],
                                    loc_diff=loc_diff_list[ind])
        else:
            add_layer(patches, colors, size=size_list[ind],
                      num=num_show_list[ind],
                      top_left=top_left_list[ind],
                      loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}'.format(
            num_list[ind]))

    text_list = ['Flatten\n', 'Fully\nconnected', 'Fully\nconnected']

    for ind in range(len(size_list)):
        label(top_left_list[ind], text_list[ind], xy_off=[-10, -65])

    ############################
    for patch, color in zip(patches, colors):
        patch.set_color(color * np.ones(3))
        if isinstance(patch, Line2D):
            ax.add_line(patch)
        else:
            patch.set_edgecolor(Black * np.ones(3))
            ax.add_patch(patch)

    plt.tight_layout()
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    fig.set_size_inches(8, 2.5)

    fig_dir = './'
    fig_ext = '.png'
    fig.savefig(os.path.join(fig_dir, 'convnet_fig' + fig_ext),
                bbox_inches='tight', pad_inches=0)