# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Version history."""

# Version|Date       |Author     |Note                                  |
# 0.0.1  |20210217   |jupiter    |Draft version                         |
# 0.0.2  |20210226   |jupiter    |Move configuration files to 'cfg'     |
# 0.0.3  |20210228   |jupiter    |Check code via DeepSource and fix it  |
# 0.0.4  |20210312   |jupiter    |Add function to dump data to file     |

from os.path import dirname
import tkinter as tk
from tkinter import filedialog
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from os.path import join
# import random
import datetime
from sklearn.preprocessing import LabelEncoder
from pylab import get_current_fig_manager
from win32api import GetSystemMetrics
from win32api import EnumDisplayMonitors
import ctypes
from platform import platform
from math import ceil

from os.path import isfile
from os.path import splitext
from os.path import basename
from os.path import abspath
from os.path import exists
from os import makedirs
from os import remove
from os import listdir
# from os import chdir

import csv
import shutil


class CodeVersionInfo:
    def __init__(self):
        """Version information."""
        self.code_version = "0.0.4a"
        self.version_date = "20210313"


class FigEnlargeRatio:
    def __init__(self):
        """Save the ratio of figure."""
        self.screen_ratio_check = False
        self.ratio_x = 1.0
        self.ratio_y = 1.0


le = LabelEncoder()

COLOR_SEQ = ["black", "blue", "red", "green", "orange", "seagreen"]
for tmp_color_seq in ["goldenrod", "slateblue", "magenta", "cyan", "orchid"]:
    COLOR_SEQ.append(tmp_color_seq)
for tmp_color_seq in ["turquoise", "khaki", "grey", "violet", "slategray"]:
    COLOR_SEQ.append(tmp_color_seq)

WITH_DARK_COLOR = ["grey", "green", "red", "orange", "goldenrod", "khaki"]
for tmp_color_dark in ["seagreen", "slategray", "cyan", "turquoise", "blue"]:
    WITH_DARK_COLOR.append(tmp_color_dark)
for tmp_color_dark in ["slateblue", "orchid", "violet", "magenta"]:
    WITH_DARK_COLOR.append(tmp_color_dark)

# database = {}

RESERVED_WORDS = ["product_name", "value_keys", "non_value_keys"]


class CfgFiles:
    def __init__(self):
        """Parameters of configuration files."""
        self.plot_file = "load_default.glgp_plot"
        self.ref_file = None
        self.preset_file = "load_default.glgp_preset"

        self.filename_with_path, self.ext_name = splitext(__file__)
        self.global_setting_file = basename(self.filename_with_path) + ".cfg"

        self.cfg_file_path = join(dirname(abspath(__file__)), "cfg")


class PresetCfg:
    def __init__(self):
        """Init the parameters of preset data."""
        self.data = {}
        self.data['_old_words'] = []
        self.data['_new_words'] = []
        self.data['_remove_words'] = []
        self.data['_data_seg'] = []
        self.data['_key_value_sep'] = []

        self.data['_post_new_item'] = []
        self.data['_post_item_01'] = []
        self.data['_post_op_code'] = []
        self.data['_post_item_02'] = []

        self.data['_trans_item'] = []
        self.data['_trans_op_code'] = []

        self.data['_format_new_item'] = []
        self.data['_format_format'] = []
        self.data['_format_item_01'] = []
        self.data['_format_item_02'] = []

        self.data['_alias_new_item'] = []
        self.data['_alias_ori_item'] = []

        self.data['_time_step_sec'] = []


class LogDatabase:
    def __init__(self):
        """Just initial a void database."""
        self.database = {}


def check_data_is_all_value(data):
    """Decide all members of data are values or not."""
    is_value = True
    data_len = len(data)

    if (data_len != 0):
        for tmp_idx in range(data_len):
            data_tmp = data[tmp_idx]
            if is_value:
                if isinstance(data_tmp, (int, float)):
                    continue
                if not data_tmp.lstrip('-+').isnumeric():
                    is_value = False
    else:
        is_value = False
    return is_value


def update_cfg_files(_cfg_files):
    _cfg_files.cfg_file_with_path = join(
        _cfg_files.cfg_file_path, _cfg_files.global_setting_file)
    if not isfile(_cfg_files.cfg_file_with_path):
        _cfg_files.cfg_file_path = dirname(abspath(__file__))

    _cfg_files.cfg_file_with_path = join(_cfg_files.cfg_file_path,
                                         _cfg_files.global_setting_file)

    possible_key_words = ["preset_file", "plot_items_file",
                          "dump_possible_items_file"]

    with open(_cfg_files.cfg_file_with_path, "r", encoding='utf-8') as f:
        fr = f.read()
        fl = fr.splitlines()

        cur_file_type = None
        for line_data in fl:
            file_name_type = find_word_between_two_words(
                line_data, '\\<', "\\>")

            if file_name_type != "":  # change mode
                if file_name_type in possible_key_words:
                    cur_file_type = file_name_type
                else:  # unknown mode
                    cur_file_type = None
                continue

            tmp_line = line_data.strip()
            tmp_line = join(_cfg_files.cfg_file_path, tmp_line)

            if isfile(tmp_line):
                if cur_file_type == "preset_file":
                    _cfg_files.preset_file = tmp_line
                elif cur_file_type == "plot_items_file":
                    _cfg_files.plot_file = tmp_line
            elif cur_file_type == "dump_possible_items_file" and tmp_line[0] != '#':
                _cfg_files.ref_file = tmp_line
        if _cfg_files.ref_file is None:
            filename_with_path = splitext(_cfg_files.plot_file)[0]
            _cfg_files.ref_file = join(_cfg_files.cfg_file_path, basename(filename_with_path) +
                                       "_possible_items.glgp_plot")


def post_process_to_preset_cfg(preset_cfg, line_data):
    tmp_line = line_data.split(";")
    for i, item in enumerate(tmp_line):
        tmp_line[i] = item.strip()
    if(len(tmp_line) == 4):
        preset_cfg.data['_post_new_item'].append(tmp_line[0])
        preset_cfg.data['_post_item_01'].append(tmp_line[1])
        preset_cfg.data['_post_op_code'].append(tmp_line[2])
        preset_cfg.data['_post_item_02'].append(tmp_line[3])

    elif(len(tmp_line) == 2):
        preset_cfg.data['_trans_item'].append(tmp_line[0])
        preset_cfg.data['_trans_op_code'].append(tmp_line[1])
    elif(len(tmp_line) == 5 and tmp_line[1] == 'fmt'):
        preset_cfg.data['_format_new_item'].append(tmp_line[0])
        preset_cfg.data['_format_item_01'].append(tmp_line[2])
        preset_cfg.data['_format_item_02'].append(tmp_line[3])
        preset_cfg.data['_format_format'].append(tmp_line[4])


def add_data_to_preset_cfg(preset_cfg, preset_mode, line_data, tmp_line):
    if preset_mode == "replace_words":
        tmp_line = line_data.split("=")
        if(len(tmp_line) == 2):
            for i, item in enumerate(tmp_line):
                tmp_line[i] = item.strip()
            preset_cfg.data['_old_words'].append(tmp_line[0])
            preset_cfg.data['_new_words'].append(tmp_line[1])
    elif preset_mode == "remove_words":
        preset_cfg.data['_remove_words'].append(tmp_line)
    elif preset_mode == "data_segment":
        preset_cfg.data['_data_seg'].append(tmp_line)
    elif preset_mode == "key_value_separate":
        preset_cfg.data['_key_value_sep'].append(tmp_line)
    elif preset_mode == "post_process":
        post_process_to_preset_cfg(preset_cfg, line_data)
    elif preset_mode == "alias":
        tmp_line = line_data.split(";")
        for i, item in enumerate(tmp_line):
            tmp_line[i] = item.strip()
        if(len(tmp_line) == 2):
            preset_cfg.data['_alias_new_item'].append(tmp_line[0])
            preset_cfg.data['_alias_ori_item'].append(tmp_line[1])
    elif preset_mode == 'time_step_sec':
        tmp_line = line_data.strip()
        preset_cfg.data['_time_step_sec'].append(tmp_line)


def get_preset_cfg_from_file(preset_file, preset_cfg):
    with open(preset_file, "r", encoding='utf-8') as f:
        fr = f.read()
        fl = fr.splitlines()
        possible_preset_modes = ["replace_words", "remove_words",
                                 "data_segment", "key_value_separate",
                                 "post_process", "alias", "time_step_sec"]

        preset_mode = None
        for line_data in fl:
            mode = find_word_between_two_words(line_data, '\\<', "\\>")
            if mode != "":  # change mode
                if mode in possible_preset_modes:
                    preset_mode = mode
                else:  # unknown mode
                    preset_mode = None
                continue

            if preset_mode is not None:
                tmp_line = line_data.strip()
            if(len(tmp_line) == 0):
                continue
            if(tmp_line[0] == '#'):
                continue
            add_data_to_preset_cfg(
                preset_cfg, preset_mode, line_data, tmp_line)


def get_win_pos_cfg(index):
    backend = matplotlib.get_backend()
    base_x = 0
    base_y = 0
    x_step = int(ceil(GetSystemMetrics(0) / 2.0))
    y_step = int(ceil(GetSystemMetrics(1) / 2.0 * 0.88))
    tmp_x = int(float(x_step) * 0.5)
    tmp_y = int(float(y_step) * 0.5)

    if index == 0:
        x = base_x
        y = base_y
    elif index == 1:
        x = base_x + x_step
        y = base_y
    elif index == 2:
        x = base_x
        y = base_y + y_step
    elif index == 3:
        x = base_x + x_step
        y = base_y + y_step
    elif index == 4:
        x = base_x + tmp_x
        y = base_y + tmp_y
    else:
        tmp_x = int(float((1+np.random.rand())*tmp_x*0.5))
        tmp_y = int(float((1+np.random.rand())*tmp_y*0.5))
        x = base_x + tmp_x
        y = base_y + tmp_y
    ans = "+%s+%s" % (x, y)
    if backend != 'TkAgg':
        ans = (x, y)

    return ans


def debug_print(data):
    if len(data) > 5000:
        print(data)


def label_incremental(array):
    labels = []
    for tmp_label in array:
        if tmp_label not in labels:
            labels.append(tmp_label)
    return list(map(labels.index, array)), labels


def find_word_between_two_words(text, pre_word, post_word):
    try:
        found = re.search('%s(.+?)%s' % (pre_word, post_word), text).group(1)
    except AttributeError:
        found = ''  # apply your error handling
    return found


def find_word_before_suffix(text, post_word):
    try:
        found = re.search('(.+?)%s' % (post_word), text).group(1)
    except AttributeError:
        found = ''  # apply your error handling
    return found


def find_word_after_prefix(text, pre_word):
    try:
        _ = re.search('%s(.+?)' % (pre_word), text).group(1)
        found = text.split(pre_word)[1]
    except AttributeError:
        found = ''  # apply your error handling
    return found


def find_product_name(line_data):
    product_name = ''
    found = find_word_between_two_words(line_data, 'info ', ']')
    if found != '':
        product_name = found
        debug_print("product_name = %s" % (product_name))
    return product_name


def check_data_line(line_data):
    if '= ' in line_data:
        return True
    return False


def remove_time_stamp_in_prefix(line_data):
    found = re.findall(r"^\d{2}:\d{2}:\d{2}:\d{3}\s*", line_data)

    if len(found) == 0:
        return line_data
    return line_data[len(found[0]):]


def database_insert_data(database, new_data, data_key, value_flag):
    database[data_key] = new_data
    if value_flag:
        database["value_keys"].append(data_key)
        database["value_keys"] = list(set(database["value_keys"]))
    else:
        database["non_value_keys"].append(data_key)
        database["non_value_keys"] = list(set(database["non_value_keys"]))
    return database


def open_file_and_check_most_key_length(database, value_keys, non_value_keys):
    file_path = filedialog.askopenfilename()
    with open(file_path, "r", encoding='utf-8') as f:
        fr = f.read()
        fl = fr.splitlines()

        # For find the suitable total length
        database["product_name"] = ""
        for line_data in fl:
            line_data = remove_time_stamp_in_prefix(line_data)

            if(check_data_line(line_data)):
                debug_print(line_data)

                tmp_len = len(line_data.split('='))
                field = line_data.split('=')[0]
                value = line_data.split('=')[1]
                if tmp_len > 2:
                    for tmp_idx in range(2, tmp_len):
                        value += line_data.split('=')[tmp_idx]

                field, value = separate_filed_and_value(field, value)

                for tmp_idx in range(min(len(field), len(value))):
                    key = field[tmp_idx].strip()
                    key_val = value[tmp_idx].strip()
                    found = find_word_before_suffix(key_val, "dBm")
                    if found != '':
                        key_val = found.strip()
                    read_key_and_keyval_to_database(
                        database, value_keys, non_value_keys, key, key_val, False, 0)
        f.close()
    database["value_keys"] = value_keys
    database["non_value_keys"] = non_value_keys
    return file_path


def pre_process_data(line_data, preset_cfg):
    line_data = remove_time_stamp_in_prefix(line_data)

    # replace words start
    for tmp_idx, _old_words in enumerate(preset_cfg.data['_old_words']):
        _new_words = preset_cfg.data['_new_words'][tmp_idx]
        line_data = line_data.replace(_old_words, _new_words)
    # replace words end

    # remove pattern start
    for _, _remove_words in enumerate(preset_cfg.data['_remove_words']):
        line_data = line_data.replace(_remove_words, '')
    # remove pattern end

    # use one format to segment data start
    for _, _data_seg in enumerate(preset_cfg.data['_data_seg']):
        line_data = line_data.replace(_data_seg, '/')
    # use one format to segment data end

    # update _key_value_sep start
    if(len(preset_cfg.data['_key_value_sep']) > 0):
        _key_value_sep = preset_cfg.data['_key_value_sep'][-1]
    else:
        _key_value_sep = '='
    # update _key_value_sep end

    line_data = line_data.replace(_key_value_sep, '=')
    _key_value_sep = '='

    return line_data, _key_value_sep


def check_data_loss(database, value_keys, key, cur_len):
    loss_data_len = cur_len - 1 - len(database[key])
    if loss_data_len > 0:
        for _ in range(loss_data_len):
            if key in value_keys:
                key_val_redundancy = -0.01
            else:
                key_val_redundancy = "unknown"
            database[key].append(key_val_redundancy)


def separate_filed_and_value(field, value):
    field = field.split('/')
    if len(field) > 1:
        value = value.split('/')
    else:
        value = [str(value)]

    for tmp_idx in range(min(len(field), len(value))):
        field[tmp_idx] = field[tmp_idx].strip()
        value[tmp_idx] = value[tmp_idx].strip()
    return field, value


def read_key_and_keyval_to_database(database, value_keys, non_value_keys, key, key_val, loss_data_check, cur_len):
    if key in database:
        if loss_data_check:
            check_data_loss(database, value_keys, key, cur_len)
        if key in value_keys:
            try:
                key_val = float(key_val)
            except ValueError:
                key_val = -0.01
        database[key].append(key_val)
    else:
        if key_val.lstrip('-+').isnumeric():
            value_keys.append(key)
            key_val = float(key_val)
        else:
            non_value_keys.append(key)
        database[key] = [key_val]

        if loss_data_check:
            check_data_loss(database, value_keys, key, cur_len)


def get_data_from_file(database, preset_cfg):
    product_name_found_flag = False
    non_value_keys = []
    value_keys = []

    root = tk.Tk()
    screen_dpi = root.winfo_pixels('1i')
    root.tk.call('tk', 'scaling', 10.0)
    root.withdraw()

    file_path = open_file_and_check_most_key_length(
        database, value_keys, non_value_keys)

    item_nums = []

    for key in database.keys():
        item_nums.append(len(database[key]))

    counts = np.bincount(item_nums)
    data_len_most = np.argmax(counts)

    for key in database.keys():
        if len(database[key]) == data_len_most:
            data_len_most_pattern = key
            break

    # Real data
    database = {}
    cur_len = 0
    with open(file_path, "r", encoding='utf-8') as f:
        fr = f.read()
        fl = fr.splitlines()
        database["product_name"] = ""

        for line_data in fl:
            line_data, _key_value_sep = pre_process_data(line_data, preset_cfg)

            if not product_name_found_flag:
                found = find_product_name(line_data)
                if found != "":
                    product_name_found_flag = True
                    database["product_name"] = found

            if(check_data_line(line_data)):
                debug_print(line_data)

                tmp_len = len(line_data.split(_key_value_sep))
                field = line_data.split(_key_value_sep)[0]
                value = line_data.split(_key_value_sep)[1]

                if tmp_len > 2:
                    for tmp_idx in range(2, tmp_len):
                        value += line_data.split(_key_value_sep)[tmp_idx]

                field, value = separate_filed_and_value(field, value)

                for tmp_idx in range(min(len(field), len(value))):
                    key = field[tmp_idx].strip()
                    if key == data_len_most_pattern:
                        cur_len += 1
                    key_val = value[tmp_idx].strip()
                    found = find_word_before_suffix(key_val, "dBm")

                    if found != '':
                        key_val = found.strip()
                    read_key_and_keyval_to_database(
                        database, value_keys, non_value_keys, key, key_val, True, cur_len)

    database["value_keys"] = value_keys
    database["non_value_keys"] = non_value_keys

    # post precess start
    # format
    for tmp_idx, _format_new_item in enumerate(preset_cfg.data['_format_new_item']):
        _format_format = preset_cfg.data['_format_format'][tmp_idx]
        _format_item_01 = preset_cfg.data['_format_item_01'][tmp_idx]
        _format_item_02 = preset_cfg.data['_format_item_02'][tmp_idx]
        if _format_item_01 in database.keys() and \
                _format_item_02 in database.keys() and _format_new_item not in database.keys():
            is_value = True
            tmp_data_array = []
            for data_index in range(min(len(database[_format_item_01]),
                                        len(database[_format_item_02]))):
                data_01 = database[_format_item_01][data_index]
                data_02 = database[_format_item_02][data_index]

                tmp_data = _format_format % (data_01, data_02)
                tmp_data_array.append(tmp_data)

            database = database_insert_data(database, tmp_data_array,
                                            _format_new_item, False)

    # alias start
    for tmp_idx, _alias_ori_item in enumerate(preset_cfg.data['_alias_ori_item']):
        _alias_new_item = preset_cfg.data['_alias_new_item'][tmp_idx]

        if _alias_ori_item in database.keys() and _alias_new_item not in database.keys():
            is_value = check_data_is_all_value(database[_alias_ori_item])

            database = database_insert_data(
                database, database[_alias_ori_item],
                _alias_new_item, is_value)
    # alias end

    # calculation start
    for tmp_idx, _post_new_item in enumerate(preset_cfg.data['_post_new_item']):
        _post_item_01 = preset_cfg.data['_post_item_01'][tmp_idx]
        _post_op_code = preset_cfg.data['_post_op_code'][tmp_idx]
        _post_item_02 = preset_cfg.data['_post_item_02'][tmp_idx]
        data_002_is_const = False
        if (_post_item_01 in database.keys() and _post_item_02 in
                database.keys()) or _post_item_01 in database.keys():
            post_operation_valid = True
            if _post_item_02 in database.keys():
                post_operation_valid = bool(check_data_is_all_value(database[_post_item_01]) and
                                            check_data_is_all_value(database[_post_item_02]))
            else:
                if check_data_is_all_value(database[_post_item_01]) and \
                        check_data_is_all_value(_post_item_02):
                    post_operation_valid = True
                    data_002_is_const = True
                else:
                    post_operation_valid = False
            if post_operation_valid:
                tmp_data_array = []
                if not data_002_is_const:
                    data_len_tmp = min(len(database[_post_item_01]),
                                       len(database[_post_item_02]))
                else:
                    data_len_tmp = len(database[_post_item_01])

                for data_index in range(data_len_tmp):
                    data_01 = database[_post_item_01][data_index]
                    if not data_002_is_const:
                        data_02 = database[_post_item_02][data_index]
                    else:
                        data_02 = float(_post_item_02)
                    if _post_op_code == '+':
                        tmp_data = float(data_01) + float(data_02)
                    elif _post_op_code == '-':
                        tmp_data = float(data_01) - float(data_02)
                    elif _post_op_code == '*':
                        tmp_data = float(data_01) * float(data_02)
                    elif _post_op_code == '/':
                        if(float(data_02) != 0):
                            tmp_data = float(data_01) / float(data_02)
                        else:
                            tmp_data = -0.01
                    elif _post_op_code == '%':
                        if(float(data_02) != 0):
                            tmp_data = data_01 % data_02
                        else:
                            tmp_data = -0.01
                    else:
                        tmp_data = -0.01
                    tmp_data_array.append(tmp_data)
                database = database_insert_data(database, tmp_data_array,
                                                _post_new_item, True)

    # calculation end

    database["value_keys"] = list(set(database["value_keys"]))
    database["non_value_keys"] = list(set(database["non_value_keys"]))

    # trans_data start
    for tmp_idx, _trans_item in enumerate(preset_cfg.data['_trans_item']):
        _trans_op_code = preset_cfg.data['_trans_op_code'][tmp_idx]
        if _trans_item in database.keys():
            if _trans_op_code == 'hex2dec':
                data_tmp_array = []
                for _, source_data in enumerate(database[_trans_item]):
                    try:
                        data_tmp = int(source_data, 16)
                    except ValueError:
                        data_tmp = -0.01
                    data_tmp_array.append(data_tmp)
                database["non_value_keys"].remove(_trans_item)
                database = database_insert_data(
                    database, data_tmp_array, _trans_item, True)
                # database["value_keys"].append(_trans_item)
            elif _trans_op_code == 'char2int':
                data_tmp_array = []
                for _, source_data in enumerate(database[_trans_item]):
                    try:
                        data_tmp = 0
                        len_tmp = len(source_data)
                        for tmp_idx_c2i in range(len_tmp):
                            data_tmp += ord(source_data[tmp_idx_c2i])
                    except ValueError:
                        data_tmp = -0.01
                    data_tmp_array.append(data_tmp)
                database["non_value_keys"].remove(_trans_item)
                database = database_insert_data(
                    database, data_tmp_array, _trans_item, True)
    # trans_data end

    return database, screen_dpi, file_path


def get_plot_data_with_le(data):
    ndata = np.array(data)
    index, labels = label_incremental(ndata)
    # index_incremental = True
    # if index_incremental:
    #     index, labels = label_incremental(ndata)
    # else:
    #     le.fit(ndata)
    #     index = list(le.transform(ndata))
    #     labels = le.classes_
    return index, labels


def get_plot_data_without_le(data):
    ndata = np.array(data)
    return ndata


def get_plot_data(data, with_le):  # with_le shold be True if data is non-value
    if with_le:
        return get_plot_data_with_le(data)
    return get_plot_data_without_le(data), []


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def set_size(w, h, ax=None):
    """Set figure width/height in inches."""
    if not ax:
        ax = plt.gca()
    left = ax.figure.subplotpars.left
    right = ax.figure.subplotpars.right
    top = ax.figure.subplotpars.top
    bottom = ax.figure.subplotpars.bottom
    figw = float(w) * (right - left)
    figh = float(h) * (top - bottom)
    ax.figure.set_size_inches(figw, figh)


class LogFigure:
    def __init__(self):
        """Init figure all data."""
        self.cur_data_count = 0
        self.time_step_sec = 1
        self.y_range_max = 0
        self.y_cur_min = 0
        self.y_cur_max = 0
        self.ylim_ext = 0
        self.parameters = []
        self.pictures = []
        self.text_flag = []  # record each data text or not
        self.cur_annotate_count = 0  # for text only
        self.cur_numeric_count = 0  # for value only
        self.numeric_idx_fig_num_array = []
        self.product_name = []
        self.title = "unknown"
        self.share_y_axis_times = 0
        self.independent_y_axis_num = 0  # all data
        self.share_start = False
        self.last_host = False
        self.last_para = None
        self.win_position = 0
        self.with_dark_fig_count = 0
        self.show_max = False
        self.show_min = False
        self.fig = None
        self.host = None
        self.ratio_x = 1.0
        self.ratio_y = 1.0
        self.screen_ratio_check = False
        self.screen_dpi = 96
        self.total_fig_num = 0
        self.code_version = '0.0.0'
        self.version_date = '20999999'
        self.save_data = False
        self.host_flag = False

    def set_show_max(self, flag):
        self.show_max = flag

    def set_show_min(self, flag):
        self.show_min = flag

    def set_save_data(self, flag):
        self.save_data = flag

    def set_win_position(self, win_position):
        self.win_position = win_position

    def set_share_y_axis_times(self, share_y_axis_times):
        self.share_y_axis_times = share_y_axis_times

    def set_time_step_sec(self, time_step):
        self.time_step_sec = time_step

    def set_title(self, title):
        self.title = title

    def set_screen_dpi(self, screen_dpi):
        self.screen_dpi = screen_dpi

    def set_total_fig_num(self, total_fig_num):
        self.total_fig_num = total_fig_num

    def set_version_info(self, codeversioninfo):
        self.code_version = codeversioninfo.code_version
        self.version_date = codeversioninfo.version_date

    def set_fig_ratio(self, fig_ratio):
        if fig_ratio.screen_ratio_check:
            self.ratio_x = fig_ratio.ratio_x
            self.ratio_y = fig_ratio.ratio_y
        else:
            win8_higher_os_flag = False

            disp_x_size_pixel = GetSystemMetrics(0)
            disp_y_size_pixel = GetSystemMetrics(1)

            dpiX = ctypes.c_uint()
            dpiY = ctypes.c_uint()

            monitors = EnumDisplayMonitors()

            windows_version = platform().split('-')[1]
            if windows_version in ('10', '8'):
                win8_higher_os_flag = True
            elif int(windows_version) >= 8:
                win8_higher_os_flag = True

            if (len(monitors) == 1):
                monitor = monitors[0]
                if win8_higher_os_flag:
                    ctypes.windll.shcore.GetDpiForMonitor(monitor[0].handle, 0,
                                                          ctypes.byref(dpiX),
                                                          ctypes.byref(dpiY))

                    debug_print(f"Monitor (hmonitor: {monitor[0]}) = dpiX:\
                                {dpiX.value}, dpiY: {dpiY.value}")
            awareness = ctypes.c_int()

            if win8_higher_os_flag:
                shcore = ctypes.windll.shcore
                errorCode = shcore.GetProcessDpiAwareness(0,
                                                          ctypes.byref(awareness))
            else:
                errorCode = 1

            if errorCode != 0:
                debug_print("GetProcessDpiAwareness errorCode = %s" %
                            (errorCode))
            else:
                debug_print("GetProcessDpiAwareness = %s" % (awareness.value))
            ori_set_value = awareness.value

            if ori_set_value == 0:
                if win8_higher_os_flag:
                    # Set DPI Awareness  (Windows 10 and 8)
                    # errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)
                    # errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(1)
                    errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(1)
                    # the argument is the awareness level, which can be 0, 1 or 2:
                else:  # bypass other windows
                    success = ctypes.windll.user32.SetProcessDPIAware()
                    if success:
                        errorCode = 0
                    else:
                        errorCode = 1

                if errorCode != 0:
                    debug_print("SetProcessDPIAware errorCode = %d" %
                                (errorCode))

            # monitors = EnumDisplayMonitors()

            # if (len(monitors) == 1):
            #     monitor = monitors[0]
            #     if win8_higher_os_flag:
            #         ctypes.windll.shcore.GetDpiForMonitor(monitor[0].handle, 0,
            #                                               ctypes.byref(dpiX),
            #                                               ctypes.byref(dpiY))

            #         debug_print(f"Monitor (hmonitor: {monitor[0]}) = dpiX:\
            #                     {dpiX.value}, dpiY: {dpiY.value}")

            if not self.screen_ratio_check:
                fig_ratio.ratio_x = float(
                    GetSystemMetrics(0)/disp_x_size_pixel)
                fig_ratio.ratio_y = float(
                    GetSystemMetrics(1)/disp_y_size_pixel)
                self.ratio_x = fig_ratio.ratio_x
                self.ratio_y = fig_ratio.ratio_y
                fig_ratio.screen_ratio_check = True

    def gen_new_figure(self):
        self.cur_data_count = 0

        tmp_dpi = self.screen_dpi

        x_size_inch = self.ratio_x * GetSystemMetrics(0) / 2 / tmp_dpi
        y_size_inch = self.ratio_y * GetSystemMetrics(1) / 2 / tmp_dpi * 0.88

        self.fig, self.host = plt.subplots(figsize=(x_size_inch, y_size_inch))

        set_size(x_size_inch, y_size_inch, self.host)

    def set_share_y_axis_of_gen_figure(self):
        host_flag = (self.cur_data_count == 0)
        self.host_flag = host_flag

        if host_flag:
            para = self.host
            if self.share_y_axis_times > 1:  # before last data
                self.share_start = True
        else:
            if self.share_start:
                para = self.last_para
                if self.share_y_axis_times == 1:  # last data
                    self.share_start = False
            else:
                para = self.host.twinx()
                if self.share_y_axis_times > 1:  # before last data
                    self.share_start = True

        self.last_para = para

    def set_axis_settings_of_gen_figure(self, data_for_plot, x_data_for_plot, para, text_flag, cur_color, cur_zorder, labels, cur_annotate_count):

        if self.show_max:
            max_val_index = list(data_for_plot).index(max(data_for_plot))
            max_x_val = x_data_for_plot[max_val_index]
            max_y_val = data_for_plot[max_val_index]
            para.plot([max_x_val], [max_y_val], "^"	, color=cur_color,
                      markersize=8, zorder=cur_zorder)

        if self.show_min:
            min_val_index = list(data_for_plot).index(min(data_for_plot))
            min_x_val = x_data_for_plot[min_val_index]
            min_y_val = data_for_plot[min_val_index]
            para.plot([min_x_val], [min_y_val], "v", color=cur_color,
                      markersize=8, zorder=cur_zorder)

        if text_flag:
            y_max = len(labels) - 1
            y_min = 0
        else:
            y_max = float(max(data_for_plot))
            y_min = float(min(data_for_plot))
        y_range = y_max - y_min

        ylim_ext = y_range * 0.03

        if self.y_range_max == 0:
            self.y_range_max = y_range

        self.y_cur_max = y_max
        self.y_cur_min = y_min
        self.ylim_ext = ylim_ext

        if text_flag:  # hide axis and add annotate
            para.get_yaxis().set_visible(False)
            for i, item in enumerate(labels):
                if i < (len(labels)-1):
                    tmp_text_y_pos = i+(ylim_ext*cur_annotate_count)
                    plt.annotate(item, (-1, tmp_text_y_pos),
                                 color=cur_color, zorder=cur_zorder+1)
                else:
                    tmp_text_y_pos = i-(ylim_ext*cur_annotate_count)
                    plt.annotate(item, (-1, tmp_text_y_pos),
                                 color=cur_color, zorder=cur_zorder+1)
            self.cur_annotate_count += 1
        else:
            if self.host_flag and self.share_y_axis_times == 1:  # last data
                para.get_yaxis().set_visible(False)

            if self.share_y_axis_times > 0:
                self.share_y_axis_times -= 1

            if self.independent_y_axis_num == 0:
                para.yaxis.tick_left()
            elif self.independent_y_axis_num == 1:
                para.yaxis.tick_right()
            if self.share_y_axis_times == 0:
                self.independent_y_axis_num += 1
            self.cur_numeric_count += 1
        self.numeric_idx_fig_num_array.append(self.share_y_axis_times)
        self.cur_data_count += 1

    def gen_figure(self, database, new_data_key):
        if new_data_key not in database:
            return

        if database["product_name"] != "":
            self.product_name = database["product_name"]

        data = database[new_data_key]

        text_flag = new_data_key in database["non_value_keys"]
        if not text_flag:
            data = list(map(float, data))

        data_for_plot, labels = get_plot_data(data, text_flag)

        if text_flag and self.share_y_axis_times > 0:
            self.share_y_axis_times = 0
            debug_print("share cnt reset!!! key %s" % (new_data_key))

        self.set_share_y_axis_of_gen_figure()
        para = self.last_para

        self.parameters.append(para)
        self.text_flag.append(text_flag)

        time_step_sec = self.time_step_sec
        cur_annotate_count = self.cur_annotate_count
        # cur_numeric_count = self.cur_numeric_count
        x_data_for_plot = np.linspace(0, len(data_for_plot) * time_step_sec -
                                      time_step_sec, len(data_for_plot))

        tmp_idx_for_color = self.cur_data_count - self.with_dark_fig_count

        if tmp_idx_for_color < len(COLOR_SEQ):
            cur_color = COLOR_SEQ[tmp_idx_for_color]
        else:
            cur_color = "grey"

        cur_alpha = 0.7
        cur_marker = ""
        cur_zorder = self.cur_data_count + 2.5
        if text_flag:
            cur_zorder += 100
            cur_alpha = 0.8

        remove_suffix_key_name = find_word_before_suffix(new_data_key, "_avg")
        record_color_array = "__" + new_data_key + "color"
        if remove_suffix_key_name != "":
            cur_marker = "."
            tmp_record_color_array = "__" + remove_suffix_key_name + "color"

            if tmp_record_color_array in database:
                ori_color = database[tmp_record_color_array]
                tmp_color = find_word_after_prefix(ori_color, "dark")
                if tmp_color == "" and database[tmp_record_color_array] in WITH_DARK_COLOR:
                    tmp_color = "dark" + database[tmp_record_color_array]
                    cur_color = tmp_color
                    self.with_dark_fig_count += 1
                    cur_alpha = 0.3

        database[record_color_array] = cur_color

        tmp_pic, = para.plot(x_data_for_plot, data_for_plot,
                             label=new_data_key, color=cur_color,
                             zorder=cur_zorder, linewidth=2,
                             alpha=cur_alpha, marker=cur_marker)

        self.pictures.append(tmp_pic)
        self.set_axis_settings_of_gen_figure(
            data_for_plot, x_data_for_plot, para, text_flag, cur_color, cur_zorder, labels, cur_annotate_count)

    def plot_figure(self):
        lines = self.pictures
        cur_numeric_count = 0
        self.set_win_position(self.total_fig_num)

        tkw = dict(size=len(self.parameters)+1, width=1.5)  # Same color
        for tmp_idx, para in enumerate(self.parameters):
            pic_tmp = self.pictures[tmp_idx]
            para.tick_params(axis='y', colors=pic_tmp.get_color(), **tkw)

            # if tmp_idx in self.numeric_idx_fig_num_array:
            if not self.text_flag[tmp_idx]:  # value
                if cur_numeric_count == 0:
                    para.yaxis.tick_left()
                elif cur_numeric_count == 1:
                    para.yaxis.tick_right()
                else:
                    para.yaxis.tick_right()
                    make_patch_spines_invisible(para)
                    para.spines["right"].set_visible(True)
                if self.numeric_idx_fig_num_array[tmp_idx] == 0:
                    cur_numeric_count += 1
            para.set_xlabel("times or seconds")
        self.host.legend(lines, [l.get_label() for l in lines],
                         loc='upper right')

        plt.title("%s analysis" % (self.title))
        plt.xlabel("times or seconds")

        left, right = plt.xlim()
        bottom, top = plt.ylim()
        x_dym = abs(right - left)
        y_dym = abs(top - bottom)

        x_tmp = right - x_dym*0.1
        y_tmp = bottom - y_dym*0.1
        text_tmp = "Version %s/%s" % (self.code_version, self.version_date)
        plt.text(x_tmp, y_tmp, text_tmp, color='orange')
        x_tmp = right - x_dym*0.15
        y_tmp = bottom - y_dym*0.13
        text_tmp = datetime.datetime.now()
        plt.text(x_tmp, y_tmp, text_tmp, color='grey')

        thismanager = get_current_fig_manager()
        backend = matplotlib.get_backend()
        if backend == 'TkAgg':
            thismanager.window.wm_geometry(get_win_pos_cfg(self.win_position))
        else:
            thismanager.window.move(get_win_pos_cfg(self.win_position)[0],
                                    get_win_pos_cfg(self.win_position)[1])

        self.total_fig_num += 1
        return self.total_fig_num


def init_cur_fig(cur_fig, version_info, screen_dpi, fig_ratio):
    cur_fig.set_version_info(version_info)
    cur_fig.set_screen_dpi(screen_dpi)
    cur_fig.set_fig_ratio(fig_ratio)
    cur_fig.gen_new_figure()


def fig_operation(fig, operation_item, avg_flag):
    # def fig_operation(fig, operation_item):
    # avg_flag_final = False
    if operation_item.isnumeric():
        fig.set_share_y_axis_times(float(operation_item))
    elif operation_item == "avg":
        avg_flag = True
    elif operation_item == "show max":
        fig.set_show_max(True)
    elif operation_item == "show min":
        fig.set_show_min(True)
    elif operation_item == "hide max":
        fig.set_show_max(False)
    elif operation_item == "hide min":
        fig.set_show_min(False)
    elif operation_item == "save data":
        fig.set_save_data(True)
    elif operation_item == "unsave data":
        fig.set_save_data(False)
    return avg_flag


def first_plot_of_fig(last_fig, new_fig_title, first_fig_flag, save_file_new):
    first_fig_flag = False
    last_fig.set_title(new_fig_title)
    save_file_new = True
    return first_fig_flag, save_file_new


def second_and_others_plot_of_fig(fig, new_fig_title, cur_fig_num,  version_info, screen_dpi, fig_ratio):
    fig[-1].set_total_fig_num(cur_fig_num)
    cur_fig_num = fig[-1].plot_figure()
    fig.append(LogFigure())  # first figure
    init_cur_fig(fig[-1], version_info, screen_dpi, fig_ratio)
    fig[-1].set_title(new_fig_title)
    return cur_fig_num


def new_fig_for_plot_figs(fig, new_fig_title, preset_cfg, first_fig_flag, save_file_new, cur_fig_num, version_info, screen_dpi, fig_ratio):
    if first_fig_flag:
        first_fig_flag, save_file_new = first_plot_of_fig(
            fig[-1], new_fig_title, cur_fig_num, save_file_new)
    else:
        cur_fig_num = second_and_others_plot_of_fig(
            fig, new_fig_title, cur_fig_num,  version_info, screen_dpi, fig_ratio)

    if(len(preset_cfg.data['_time_step_sec']) > 0):
        step_sec = preset_cfg.data['_time_step_sec'][-1]
        if(check_data_is_all_value(step_sec)):
            fig[-1].set_time_step_sec(float(step_sec))
    return first_fig_flag, save_file_new, cur_fig_num


def plot_figs(cfg_file_with_path, preset_cfg, database, version_info, screen_dpi, fig, fig_ratio):
    first_fig_flag = True
    avg_flag = False
    cur_fig_num = 0
    save_data_with_new_file = False
    with open(cfg_file_with_path, "r") as f:

        fr = f.read()
        fl = fr.splitlines()

        for line_data in fl:
            new_fig_title = find_word_between_two_words(
                line_data, '\\[', "\\]")
            operation_item = find_word_between_two_words(line_data,
                                                         '\\<', "\\>")

            if new_fig_title != "":  # new figure
                first_fig_flag, save_data_with_new_file, cur_fig_num = new_fig_for_plot_figs(fig, new_fig_title, preset_cfg, first_fig_flag,
                                                                                             save_data_with_new_file, cur_fig_num, version_info, screen_dpi, fig_ratio)

            elif operation_item != "":  # share y axis
                avg_flag = fig_operation(fig[-1], operation_item, avg_flag)
            else:  # data only
                if not avg_flag:
                    fig[-1].gen_figure(database, line_data)
                    if first_fig_flag:
                        first_fig_flag = False

                    #  record data to file start
                    if fig[-1].save_data and line_data in database:
                        save_data_path = join(
                            dirname(abspath(__file__)), "output")
                        if not exists(save_data_path):
                            makedirs(save_data_path)
                        output_file = "%s.log" % (fig[-1].title)
                        output_file_with_path = join(
                            save_data_path, output_file)
                        if save_data_with_new_file:
                            if exists(output_file_with_path):
                                remove(output_file_with_path)
                            save_data_with_new_file = False
                        with open(output_file_with_path, mode='a', newline='', encoding='utf-8') as save_data_file:
                            save_data_writer = csv.writer(
                                save_data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            write_data = database[line_data][:]
                            write_data.insert(0, line_data)
                            save_data_writer.writerow(write_data)
                    #  record data to file end

                else:  # average the data and name original_name_avg
                    if line_data in database:
                        tmp_data_len = len(database[line_data])
                        try:
                            tmp_avg = sum(database[line_data]) / tmp_data_len
                        except TypeError:
                            tmp_avg = -0.01

                        new_name = line_data + "_avg"
                        database[new_name] = [tmp_avg] * tmp_data_len
                    avg_flag = False
    return cur_fig_num


def main():
    _cfg_files = CfgFiles()
    update_cfg_files(_cfg_files)
    preset_cfg = PresetCfg()

    preset_file = _cfg_files.preset_file
    get_preset_cfg_from_file(preset_file, preset_cfg)
    log_data = LogDatabase()
    log_data.database, screen_dpi, log_file_with_path = get_data_from_file(
        log_data.database, preset_cfg)
    database = log_data.database
    fig = []
    # cur_path = getcwd()
    cur_path = dirname(__file__)
    cfg_file_with_path = join(_cfg_files.cfg_file_path, _cfg_files.plot_file)
    cfg_ref_file_with_path = join(cur_path, _cfg_files.ref_file)
    version_info = CodeVersionInfo()

    fig_ratio = FigEnlargeRatio()

    with open(cfg_ref_file_with_path, 'w') as fw:
        for key in database.keys():
            if key not in RESERVED_WORDS:
                f_data = "%s\n" % (key)
                fw.write(f_data)
        fw.close()

    fig.append(LogFigure())  # first figure
    init_cur_fig(fig[-1], version_info, screen_dpi, fig_ratio)

    cur_fig_num = plot_figs(cfg_file_with_path, preset_cfg,
                            database, version_info, screen_dpi, fig, fig_ratio)

    fig[-1].set_total_fig_num(cur_fig_num)
    fig[-1].plot_figure()  # last figure

    #  collect record files start
    check_record_data = True
    output_path = join(dirname(abspath(__file__)), "output")
    if not exists(output_path):
        check_record_data = False
        # makedirs(output_path)

    if check_record_data:
        confirm_per_output_path = True
        file_names = listdir(output_path)

        for file_name in file_names:
            if splitext(file_name)[1] == '.log':
                if confirm_per_output_path:
                    per_output_path = join(output_path, splitext(
                        basename(log_file_with_path))[0])
                    if not exists(per_output_path):
                        makedirs(per_output_path)
                    confirm_per_output_path = False
                shutil.move(join(output_path, file_name),
                            join(per_output_path, file_name))
    #  collect record files end

    plt.ion()
    plt.show()
    plt.draw()
    plt.pause(0.001)
    input("Press [enter] to exit.")


if __name__ == "__main__":
    main()
