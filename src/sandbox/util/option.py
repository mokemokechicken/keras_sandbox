# coding: utf8

import os
import sys
from argparse import ArgumentParser


def standard_option(name='Standard Options'):
    parser = ArgumentParser(name)
    parser.add_argument('--name', default='no_name')
    parser.add_argument('--model', type=str)
    parser.add_argument('--epoch', type=int)
    return parser
