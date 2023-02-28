import logging
import time
import linecache
import paramiko
from datetime import datetime
import os
import ast
import colorama
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import shutil
from colorama import Fore, Back, Style
colorama.init(autoreset=True)

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(Fore.LIGHTBLUE_EX + f'new folder are made: {directory}')

def create_readme_file(SN, ip):
    list_graphs = os.listdir(ip + '/' + SN)
    with open(ip + '/' + SN + '/' + 'read_me.txt', 'w') as f:
        f.write('No. of graphs:  ' + str(len(list_graphs) - 1))
        f.writelines(['\n' + x for x in list_graphs])

def check_last_update_vs_time_create(file_path, IP):
    time_create = datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
    with open(str(IP) + '/' + 'last_update_station: ' + str(IP) + '.txt', 'r') as f:
        last_update = f.read()
    # print(f'time_create > now    -   {time_create} > {last_update}:     {time_create > last_update}')
    # print(f'time_create < now    -   {time_create} < {last_update}:     {time_create < last_update}')
    return time_create < last_update

def last_update_date(IP):
    with open(str(IP) + '/' + 'last_update_station: ' + str(IP) + '.txt', 'w') as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def count_file_with_same_name(IP, SN):
    files_list = os.listdir(IP)
    count = [str(SN) in file for file in files_list].count(True)
    return count

def search_str(file_path, find_next_data):

    def order_data(get_data, location_data_inside_lines, count, dict_2_list=False, US=False):
        if US == True:
            get_data = get_data[location_data_inside_lines[count]:-1]
        else:
            get_data = get_data[location_data_inside_lines[count]:]
        get_data = ast.literal_eval(get_data)
        if dict_2_list == True:
            get_data = [get_data]
        len_data = len(get_data)
        for j in range(len_data):
            if US == True:
                try:
                    get_data[j]['graph_data'][0][0]['curves'][0]['color'] = 'red'
                    data.append(get_data[j])
                except (TypeError, KeyError) as ErrorA:
                    print(Fore.LIGHTGREEN_EX + str(ErrorA))
                    logging.warning(f"Exception Name: {type(ErrorA).__name__}")
                    logging.warning(f"Exception Desc: {ErrorA}")
            else:
                data.append(get_data[j])
        return data

    founded_data = {'SN': False, 'end_test': False, 'full_band': False, 'flatness_and_tilt': False,
                    'US': False, 'Error': False}
    location_data_lines, location_data_inside_lines, data = [], [], []
    time.sleep(0.00000001)
    with open(file_path, 'r', encoding='UTF8') as file:
        lines = file.readlines()
        for line in lines:
            if line.find(find_next_data) != -1:
                location_data_lines.append(lines.index(line))
                location_data_inside_lines.append(line.find(find_next_data))
    file.close()

    for count, line_a in enumerate(location_data_lines):
        line_a += 1
        get_data = linecache.getline(file_path, line_a).strip()
        if 'serial number is: ' in find_next_data:
            founded_data['SN'] = True
            get_data = get_data[location_data_inside_lines[count]:]
            if 'serial number is:' in get_data:
                data.append(get_data)
        elif 'Full bandwidth signal' in find_next_data:
            data = order_data(get_data, location_data_inside_lines, count)
            founded_data['full_band'] = True
        elif 'Flatness and Tilt' in find_next_data:
            founded_data['flatness_and_tilt'] = True
            data = order_data(get_data, location_data_inside_lines, count, dict_2_list=True)
        elif 'end of Test UpStream Measurements' in find_next_data:
            founded_data['end_test'] = True
            data.append(get_data)
        elif 'US' in find_next_data:
            founded_data['US'] = True
            data = order_data(get_data, location_data_inside_lines, count, US=True)
        else:
            founded_data['Error'] = True
            print(Fore.LIGHTGREEN_EX + f'Error with search_str() function')
    len_lines = len(lines)
    return  data, len_lines, founded_data

def create_graph(data, host, SN):
    # generate each graph present in data
    graph_data_titles = []
    for graph in data:
        graph_data_titles.append(graph["graph_title"])
        print(str(graph["graph_title"] + '  for SN: ' + str(SN)))
        graph_data = graph["graph_data"]
        nbr_vertical_subplot = len(graph_data)
        nbr_horizontal_subplot = max(len(subplot) for subplot in graph_data if subplot is not None)
        fig, axs = plt.subplots(nbr_vertical_subplot, nbr_horizontal_subplot, squeeze=False)
        fig.suptitle(graph["graph_title"])
        for i in range(nbr_vertical_subplot):
            if graph_data[i] is not None:
                for j in range(nbr_horizontal_subplot):
                    subplot = graph_data[i][j]
                    if subplot is not None:
                        axs[i, j].set_title(subplot["plot_title"])
                        axs[i, j].set_xlabel(subplot["X_axis_label"], color=subplot["X_label_color"])
                        axs[i, j].set_ylabel(subplot["Y_axis_label"], color=subplot["Y_label_color"])
                        for curve in subplot["curves"]:
                            axs[i, j].plot(curve["X_data"], curve["Y_data"], color=curve["color"],
                                           label=curve["curve_name"], linewidth=curve["linewidth"],
                                           marker=curve["marker"], markersize=curve["marker_size"])

                        if any(curve["curve_name"] != "" for curve in subplot["curves"]):
                            axs[i, j].legend()
                        axs[i, j].grid()
        create_directory_if_not_exists(host + '/' + str(SN))
        time.sleep(0.0000001)
        # print(str(SN[:12]))
        plt.savefig(host + '/' + str(SN) + '/' + graph["graph_title"] + ' SN - ' + SN[:12] + ".png")
        # plt.pause(0.1)
        plt.close()
    return graph_data_titles

def connect_with_sftp(host, username, password, target_folder, local_folder):
    #------------------------------Paramiko - Open and connect transport-----------------------------------------------#
    transport = paramiko.Transport((host, 22))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    #-------------------------------------transfer file from remote to host--------------------------------------------#
    folders = sftp.listdir(target_folder)
    for count, folder in enumerate(folders):
        files = sftp.listdir(target_folder + folder)
        for file in files:
            filepath = target_folder + folder + '/' + file
            localpath = local_folder + str(count) + file
            # print('filepath:        ', filepath)
            # print('localpath:       ', localpath)
            sftp.get(filepath, localpath)
    #----------------------------------------serch "end test"----------------------------------------------------------#
            data, Log_len, founded_data = search_str(localpath, 'end of Test UpStream Measurements')
            data2, Log_len2, founded_data2 = search_str(localpath, 'serial number is: ')
            # print('founded_data:        ', founded_data)
            # print('founded_data2:        ', founded_data2)
            # print('Log_len:        ', Log_len)
            # print('Log_len2:        ', Log_len2)
            if founded_data['end_test']==True and founded_data2['SN']==True and Log_len>50000:
                # print(Fore.GREEN + 'PASS PASS PASS PASS')
                # print(Fore.LIGHTGREEN_EX + f'folder:       {folder}')
                # print(Fore.LIGHTGREEN_EX + f'Full test LOG')
                # print(Fore.LIGHTGREEN_EX + f'found serial number,')
                # print(Fore.LIGHTGREEN_EX + f'Log_len: = {Log_len}')
                if data2 == []:
                    os.remove(host + '/' + str(count) + file)
                else:
                    old_file = os.path.join(host, str(count) + file)
                    # print(Fore.GREEN + f'data:     {data2}')
                    SN = data2[0][-12:]
                    SN_exists = host + '/' + SN + '.txt'
                    if not os.path.isfile(SN_exists):
                        new_file = os.path.join(host, SN + '.txt')
                        os.rename(old_file, new_file)
                        # print(Fore.GREEN + str(data))
                    else:
                        count2 = count_file_with_same_name(host, SN)
                        # print(Fore.LIGHTCYAN_EX + 'count2', count2)
                        new_file = os.path.join(host, SN + ' - ' + (str(count2 + 1)) + '.txt')
                        os.rename(old_file, new_file)
                        # print(Fore.LIGHTRED_EX + 'data:     ' + str(data2))
            else:
                # print(Fore.RED + 'FAIL FAIL FAIL FAIL FAIL')
                os.remove(host + '/' + str(count) + file)

if __name__ == '__main__':
    # IPs = ['10.41.42.4', '10.41.42.10', '10.41.42.13', '10.41.42.28']
    IPs = ['10.41.42.28']
    for ip in IPs:
        host, username, password = ip, "harmonic", "harmonic"
        target_folder = "/home/harmonic/debug_logs/"
        create_directory_if_not_exists(host)
        time.sleep(0.00000001)
        # create_directory_if_not_exists(host + '/' + 'half_LOGS')
        time.sleep(0.00000001)
        connect_with_sftp(host, username, password, target_folder, host + '/')
        list_new_LOGs = [os.path.splitext(filename)[0] for filename in os.listdir(ip)]
        # if 'last_update_station: ' + str(ip) + '.txt' in list_new_LOGs: list_new_LOGs.remove('last_update_station: ' + str(ip) + '.txt')
        list_new_LOGs.sort()
        # print('list_new_LOGs:       ',list_new_LOGs)
        for LOG in list_new_LOGs:
            # print('LOG:     ', LOG)
            find_next_data = {'DS1_DS2_Band': "[{'graph_title': 'DS1 Full bandwidth signal",
                              'DS1_Tilt_0': "{'graph_title': 'DS1 Flatness and Tilt 0",
                              'DS1_Tilt_7': "{'graph_title': 'DS1 Flatness and Tilt 7",
                              'DS1_Tilt_13': "{'graph_title': 'DS1 Flatness and Tilt 13",
                              'DS2_Tilt_0': "{'graph_title': 'DS2 Flatness and Tilt 0",
                              'DS2_Tilt_7': "{'graph_title': 'DS2 Flatness and Tilt 7",
                              'DS2_Tilt_13': "{'graph_title': 'DS2 Flatness and Tilt 13",
                              'US_All': "{'graph_title': 'US"}
            for keys, value in find_next_data.items():
                # print('keys:        ', keys)
                # print('value:        ', value)
                datas, len_lines, founded_data = search_str(host + '/' + LOG + '.txt', value)
                for data in datas:
                    create_graph([data], host, LOG)
            shutil.move(os.path.join(host, str(LOG) + '.txt'), os.path.join(host + '/' + str(LOG), str(LOG) + '.txt'))
            create_readme_file(LOG, host)
        time.sleep(0.0000001)
        # check_last_update_vs_time_create('file_path_we_want_to_check',ip)
        # last_update_date(ip)
