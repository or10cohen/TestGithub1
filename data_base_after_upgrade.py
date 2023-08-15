import logging
import numpy as np
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
from zipfile import ZipFile
from pathlib import Path
from colorama import Fore, Back, Style
colorama.init(autoreset=True)

class make_graps_from_log():
    def __init__(self):
        pass

    def create_directory_if_not_exists(self, directory):

        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(Fore.LIGHTBLUE_EX + f'new folder are made: {directory}')
            except:
                print(Fore.RED + f'Error to create folder: {directory}')

    def create_readme_file(self, SN, ip):
        list_graphs = os.listdir(ip + '/' + SN)
        with open(ip + '/' + SN + '/' + 'read_me.txt', 'w') as f:
            f.write('No. of graphs:  ' + str(len(list_graphs) - 1))
            f.writelines(['\n' + x for x in list_graphs])

    def check_last_update(self):
        with open('last_update.txt', 'r') as f:
            last_update = f.read()
        print(Fore.LIGHTGREEN_EX + f'last update: {last_update}')

    def last_update_date(self, IP):
        with open('last_update.txt', 'w') as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def filter_logs_are_already_exist(self):
        print('len all logs before filters logs_are_already_exist', self.all_path_list)
        print('all logs before filters logs_are_already_exist', len(self.all_path_list))
        path_count = []
        [path_count.append(path.count('/')) for path in self.all_path_list]
        print(path_count)
        where_4_in_path_count = np.where(np.array(path_count)==4)
        print(where_4_in_path_count)
        where_4_in_path_count = where_4_in_path_count[0].tolist()
        print(where_4_in_path_count)

        # print('all logs after filters logs_are_already_exist', self.all_path_list)
        # print('len all logs after filters logs_are_already_exist', len(self.all_path_list))

    def count_file_with_same_name(self, IP, SN):
        files_list = os.listdir(IP)
        count = [str(SN) in file for file in files_list].count(True)
        return count

    def search_str(self, file_path, find_next_data):

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
                        'US': False, 'Error': False, 'len_lines': None, 'data_US': None, 'data_SN': None}
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
        founded_data['len_lines'] = len_lines
        return data, founded_data

    def create_graph(self, data, host, SN):
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
            time.sleep(0.0000001)
            # print(str(SN[:12]))
            plt.savefig(host + '/' + str(SN) + '/' + graph["graph_title"] + ' SN - ' + SN[:12] + ".png")
            # plt.pause(0.1)
            plt.close()
        return graph_data_titles

    def zip_logs(self, host, username, password):
        #------------------------------Paramiko - Open and connect transport-----------------------------------------------#
        client = paramiko.client.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, username=username, password=password)
        stdin, stdout, stderr = client.exec_command("zip -r debug_logs debug_logs")
        stdout.read().decode()
        print(stdout.read().decode())
        client.close()
        print(Fore.LIGHTGREEN_EX + f'ziping log files in host IP: {host}')
        return stdin, stdout, stderr

    def transfer_zip_file(self, host, username, password, target_folder, local_folder):
        transport = paramiko.Transport((host, 22))
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        # -------------------------------------transfer zip file from remote to host--------------------------------------------
        filepath = target_folder + 'debug_logs.zip'
        localpath = local_folder + '/' + host + '_logs_file.zip'
        sftp.get(filepath, localpath)
        transport.close()
        print(Fore.LIGHTGREEN_EX + f'transferring zip file to local host: {host}')

    def extract_zip_file(self, local_folder):
        print(Fore.LIGHTGREEN_EX + f'starting to extract files from zip in folder: {host}')
        self.path = local_folder + '/' + local_folder + '_logs_file.zip'
        shutil.unpack_archive(self.path, local_folder)

    def delete_zip_file(self, local_folder):
        try:
            os.remove(local_folder + '/' + local_folder + '_logs_file.zip')
            print(Fore.LIGHTGREEN_EX + f'deleted zip file in folder: {host}')
        except:
            print(f"{local_folder}/{local_folder}_logs_file.zip already REMOVED")

    def list_files(self, local_folder):
        self.all_path_list = []
        folders = os.listdir(local_folder + '/' + 'debug_logs')
        for log_folder in folders:
            path_list = os.listdir(local_folder + '/' + 'debug_logs' + '/' + log_folder)
            for path in path_list:
                if '.log' in path:
                    self.all_path_list.append(local_folder + '/' + 'debug_logs' + '/' + log_folder + '/' + os.path.relpath(path))
                else:
                    path_in_path_list = os.listdir(local_folder + '/' + 'debug_logs' + '/' + log_folder + '/' + path)
                    for path_in_path in path_in_path_list:
                        self.all_path_list.append(local_folder + '/' + 'debug_logs' + '/' + log_folder + '/' + path + '/' + os.path.relpath(path_in_path))
        print(self.all_path_list)

        return self.all_path_list

    def check_log(self):
        self.all_path_dict = dict.fromkeys(self.all_path_list, "search US")
        for path in self.all_path_list:
            self.data_US, self.founded_data = make_graps_from_log.search_str(self, path, 'end of Test UpStream Measurements')
            self.data_SN, self.founded_data2 = make_graps_from_log.search_str(self, path, 'serial number is: ')
            self.all_path_dict[path] = self.founded_data
            self.all_path_dict[path]['data_SN'] = self.data_SN
            self.all_path_dict[path]['data_US'] = self.data_US
            if self.founded_data2['SN'] == True:
                self.all_path_dict[path]['SN'] = True

    def graph_from_good_log(self, host):
        False_data, removed_empty_data, logs_are_good, total_logs = 0, 0, 0, len(self.all_path_dict)
        for key, value in self.all_path_dict.items():
            print(key)
            print(value)
            if value['SN'] == True and value['end_test'] == True and int(value['len_lines']) > 20000:
                if value["data_SN"] == []:
                    os.remove(key)
                    # print(Fore.GREEN + f'removed empty data:   {value["data_SN"]}  )
                    removed_empty_data += 1
                else:
                    logs_are_good += 1
                    # print(Fore.GREEN + f'full data:       {self.data2}')
                    SN = value["data_SN"][0][-12:]
                    SN_exists = host + '/' + SN + '.txt'
                    if not os.path.isfile(SN_exists):
                        new_file = os.path.join(host, SN + '.txt')
                        os.rename(key, new_file)
                        # print(Fore.GREEN + str(self.data))
                    else:
                        count2 = make_graphs.count_file_with_same_name(host, SN)
                        # print(Fore.LIGHTCYAN_EX + 'count2', count2)
                        new_file = os.path.join(host, SN + ' - ' + (str(count2 + 1)) + '.txt')
                        os.rename(key, new_file)
                        # print(Fore.LIGHTRED_EX + 'data:     ' + str(data2))
            else:
                False_data += 1
                print(Fore.RED + 'SN/end_test/len_lines are False')
                os.remove(key)
        print(Fore.LIGHTRED_EX + f'total logs:  {total_logs}')
        print(Fore.LIGHTRED_EX + f'False SN/end_test/len_lines:  {False_data}')
        print(Fore.LIGHTRED_EX + f'removed_empty_data:  {removed_empty_data}')
        print(Fore.LIGHTRED_EX + f'logs_are_good:  {logs_are_good}')

    def delete_folder(self, local_folder):
        try:
            shutil.rmtree(local_folder + '/' + 'debug_logs')
            print(Fore.LIGHTGREEN_EX + f'deleted folder debug_logs at : {host}')
        except:
            print(Fore.LIGHTGREEN_EX + f"{local_folder}/'debug_logs' already REMOVED")

if __name__ == '__main__':
    #IPs = ['10.41.42.4', '10.41.42.10', '10.41.42.13', '10.41.42.34']
    IPs = ['10.41.42.10']
    make_graphs = make_graps_from_log()
    make_graphs.check_last_update()
    for IP in IPs:
        host, username, password = IP, "harmonic", "harmonic"
        target_folder = "/home/harmonic/"
        make_graphs.create_directory_if_not_exists(host)
        local_folder = host
        stdin, stdout, stderr = make_graphs.zip_logs(host, username, password)
        make_graphs.transfer_zip_file(host, username, password, target_folder, host)
        make_graphs.extract_zip_file(local_folder)
        make_graphs.delete_zip_file(local_folder)
        make_graphs.list_files(local_folder)
        make_graphs.filter_logs_are_already_exist()
        make_graphs.check_log()
        make_graphs.graph_from_good_log(host)
        make_graphs.delete_folder(host)
        list_new_LOGs = [os.path.splitext(filename)[0] for filename in os.listdir(host)]
        for LOG in list_new_LOGs:
            make_graphs.create_directory_if_not_exists(host + '/' + str(LOG))
            find_next_data = {'DS1_DS2_Band': "[{'graph_title': 'DS1 Full bandwidth signal",
                              'DS1_Tilt_0': "{'graph_title': 'DS1 Flatness and Tilt 0",
                              'DS1_Tilt_7': "{'graph_title': 'DS1 Flatness and Tilt 7",
                              'DS1_Tilt_13': "{'graph_title': 'DS1 Flatness and Tilt 13",
                              'DS2_Tilt_0': "{'graph_title': 'DS2 Flatness and Tilt 0",
                              'DS2_Tilt_7': "{'graph_title': 'DS2 Flatness and Tilt 7",
                              'DS2_Tilt_13': "{'graph_title': 'DS2 Flatness and Tilt 13",
                              'US_All': "{'graph_title': 'US"}
            for keys, value in find_next_data.items():
                datas, founded_data = make_graphs.search_str(host + '/' + LOG + '.txt', value)
                for data in datas:
                    make_graphs.create_graph([data], host, LOG)
            shutil.move(os.path.join(host, str(LOG) + '.txt'), os.path.join(host + '/' + str(LOG), str(LOG) + '.txt'))
            make_graphs.create_readme_file(LOG, host)
        time.sleep(0.0000001)
