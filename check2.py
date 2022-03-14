import matplotlib.pyplot as plt
import numpy as np

out_No_Signal_before = open("C:\\Users\\or_cohen\\Desktop\\out_No_Signal_before.txt", "r")
out_No_Signal_after = open("C:\\Users\\or_cohen\\Desktop\\out_No_Signal_after.txt", "r")

def AO_low_Frequency_before_afrer_replace_L54(Data, start_point=0, stop_point=1):
    count = 0
    max_hold = []
    frequency = []
    for index, line in enumerate(Data):
        if index > start_point + 2:
            pass
            if index == stop_point + 2:
                break
            line = line.split()
            frequency.append(line[0])
            max_hold.append(line[2])
    Data.close()
    frequency = np.asarray(np.array(frequency), dtype=np.float64, order='C')
    frequency = np.around(frequency, decimals=1)
    max_hold = np.asarray(np.array(max_hold), dtype=np.float64, order='C')
    max_hold = np.around(max_hold, decimals=1)
    return frequency, max_hold

start_point = 0
stop_point = 1000
frequency, max_hold_before = AO_low_Frequency_before_afrer_replace_L54(out_No_Signal_before, start_point, stop_point)
same_frequency, max_hold_after = AO_low_Frequency_before_afrer_replace_L54(out_No_Signal_after, start_point, stop_point)
print(frequency)
print(max_hold_before)
print(max_hold_after)

after_minus_before = np.subtract(max_hold_before, max_hold_after)
print(after_minus_before)


jumps = 4
x_label = frequency
y_label = after_minus_before
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.title('Difference in dB after replace inductor L54')
plt.xlabel('Frequency[MHz]')
plt.ylabel('Difference after replace L54[dB] ')
plt.plot(x_label[::jumps], y_label[::jumps])
plt.show()
