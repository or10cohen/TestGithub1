import os
import streamlit as st
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)
st.set_page_config(layout="wide")
from import_txt import search_str
from graph_data_analyzer import create_graph
import json
import ast
import io
import numpy as np

def get_directories_with_numeric_names(folder):
    return [item for item in os.listdir(folder) if os.path.isdir(os.path.join(folder, item)) and any(char.isdigit() for char in item)]

# Example usage
directories = get_directories_with_numeric_names('C:/Users/or_cohen/PycharmProjects/Gui_from_log_local')
def main():
##----------------------------------------------------------------------------------------------------------------------
##----------------------------------------------main page---------------------------------------------------------------
##-------------------------------------------Grpah GUI for Gen2 log-----------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
    global Run_Function
    st.title('Graphs from log file')
    if st.button('Run Function'):
        st.write('Running Function')
        Run_Function = 'Run Function'
    else:
        st.write('Press run function to start')
        Run_Function = 'Dont Run Function'

##---------------------------------------------------sidebar------------------------------------------------------------
##-------------------------------------------choose Datatype&Dataset----------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
    with st.sidebar:
        st.title('choose log.txt')
        IPS = get_directories_with_numeric_names('C:/Users/or_cohen/PycharmProjects/Gui_from_log_local')
        IP = st.selectbox(
            'choose ATE IP',
            IPS)

        Logs = os.listdir('C:/Users/or_cohen/PycharmProjects/Gui_from_log_local/' +  IP)
        log_file_text = st.selectbox(
            'choose SN',
            Logs)

        SN = str(log_file_text[:-4])
##---------------------------------------------------sidebar------------------------------------------------------------
##-----------------------------------------download files you want to share---------------------------------------------
##----------------------------------------------------------------------------------------------------------------------



##----------------------------------------------------------------------------------------------------------------------
##----------------------------------------------main page---------------------------------------------------------------
##----------------------------------Neural Network - Run Function-------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
    if Run_Function == 'Run Function':
        # log_file_path = IP + '//' + log_file_text
        # print(log_file_path)
        # log_file_path = "105222106473.txt"

##------------------------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
        list_graphs_path = [file for file in os.listdir(
            'C:/Users/or_cohen/PycharmProjects/Gui_from_log_local/' + IP + '/' + log_file_text) if
                            file.endswith('.png')]
        list_graphs = [os.path.splitext(filename)[0] for filename in list_graphs_path]

        list_graphs_path_DS = []
        list_graphs_path_US = []
        list_graphs_path_rest = []
        for name in list_graphs_path:
            if 'DS' in name:
                list_graphs_path_DS.append(name)
            elif 'US' in name:
                list_graphs_path_US.append(name)
            else:
                list_graphs_path_rest.append(name)

        list_graphs_path_DS.sort()
        list_graphs_DS = [os.path.splitext(filename)[0] for filename in list_graphs_path_DS]
        list_graphs_path_US.sort()
        list_graphs_US = [os.path.splitext(filename)[0] for filename in list_graphs_path_US]
        # list_graphs_path_rest.sort()
        # list_graphs_rest = [os.path.splitext(filename)[0] for filename in list_graphs_path_rest]
##------------------------------------------main page------------------------------------------------------------------
##----------------------------------------tabs and Graphs---------------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
        tabs_DS = st.tabs(list_graphs_DS)
        for i in range(len(list_graphs_path_DS)):
            with tabs_DS[i]:
                st.header(list_graphs_DS[i])
                st.image('C:\\Users\\or_cohen\\PycharmProjects\\Gui_from_log_local\\' + str(IP) + '\\' + str(
                    log_file_text) + '\\' + str(list_graphs_path_DS[i]))

        tabs_US = st.tabs(list_graphs_US)
        for i in range(len(list_graphs_path_US)):
            with tabs_US[i]:
                st.header(list_graphs_US[i])
                st.image('C:\\Users\\or_cohen\\PycharmProjects\\Gui_from_log_local\\' + str(IP) + '\\' + str(
                    log_file_text) + '\\' + str(list_graphs_path_US[i]))

        # tabs_rest = st.tabs(list_graphs_rest)
        # for i in range(len(list_graphs_path_rest)):
        #     with tabs_rest[i]:
        #         st.header(list_graphs_rest[i])
        #         st.image('C:\\Users\\or_cohen\\PycharmProjects\\Gui_from_log_local\\' + str(IP) + '\\' + str(
        #             log_file_text) + '\\' + str(list_graphs_path_rest[i]))


##-----------------------------------------------------------------------------------------------------------
main()



