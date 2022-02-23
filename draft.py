# from common.plot import plot_ioncn
# x=list(range(10))
# y=[a**2 for a in range(10)]
# plot_ioncn(x,save_name="test1")
# plot_ioncn(y,save_name="test2",save_path='./ppo/res_png//')

# import time
# time_num=time.strftime("%Y%m%d",time.localtime(time.time()))
# print(time_num)

# from common.plot import plot_multi_cn
# x=list(range(10))
# y=[a**2 for a in range(10)]
# plot_multi_cn([x,y],save_path='./ppo/res_png/',save_name="testfig")


from common.plot import plot_cn_off
x=list(range(10))
plot_cn_off(x,save_path='./ppo/res_png/',save_name="testfig2")

