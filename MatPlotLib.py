import matplotlib.pyplot as plt
import numpy as np


####################################MATPLOTLIB PART 1#########################################
x = np.linspace(0, 5 ,11)
y = x ** 2
#
# print(x)
# print(y)

# plt.plot(x, y)
# plt.xlabel('X Label')
# plt.ylabel('Y Label')
# plt.title('Title')
# # plt.show()
#
# plt.plot(x, y , 'o')    # shoe me the points on the graph without thw line
# # plt.show()
#
# #-------------------subplot like Matlab------------------
# plt.subplot(1 ,2 ,1) # subplot(rows no., columns no., plot no.). first graph.
# plt.xlabel('X Label')
# plt.ylabel('Y Label')
# plt.title('Title: First Graph Blue line')
# plt.plot(x, y , 'b') # 'b' = blue
#
# plt.subplot(1 ,2 ,2) # subplot(rows no., columns no., plot no.). second graph.
# plt.plot(y, x, 'r') # 'r' = red
# plt.xlabel('X Label')
# plt.ylabel('Y Label')
# plt.title('Title: second Graph red line')
#
# # plt.show()                  # use me!
#
# #-----------------------------use figure in plt---------------------
# fig = plt.figure() #create new figure from zero(empty canbas)
# axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # ([left axis, bottom axis, width, high])  0 < argument < 1
# axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])  # ([left axis, bottom axis, width, high])  0 < argument < 1
# axes1.plot(x, y)
# axes2.plot(y, x)
# axes1.set_xlabel('X Label fig')
# axes1.set_ylabel('Y Label fig')
# axes1.set_title('Title: fig')
# axes2.set_xlabel('X Label fig')
# axes2.set_ylabel('Y Label fig')
# axes2.set_title('Title: fig')
#
# # plt.show()

####################################MATPLOTLIB PART 2#########################################
fig,axes = plt.subplots()
# axes.plot(x, y)
plt.show()

fig,axes = plt.subplots(nrows=1, ncols=2) #the axes is a list of axes. in this case we have 2 axes
axes[0].plot(x, y)
axes[1].plot(y, x)
plt.show()

fig,axes = plt.subplots(nrows=3, ncols=3) #the axes is a list of axes. in this case we have 9 axes
# axes.plot(x, y)
plt.show()
