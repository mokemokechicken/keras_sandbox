# coding: utf8

import os


def create_basedir(file_path):
    if not os.path.isdir(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
