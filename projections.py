#!/usr/bin/env python

import sys
import configparser
from collections import namedtuple
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from ipdb import set_trace



def main():
	'''
	projections.py

	Calculate and plot revenue projections as a function of number of customers.
	'''

	# get arguments/filenames
	config_file = sys.argv[1]

	# read config file
	settings = Settings(config_file)

	# read in data

	# process data

	# generate plots

	print("Finished.")
	sys.exit()



class Settings:
	def __init__(self, config_file):
		config = configparser.ConfigParser()
		config.read(config_file)
		self.plot = self.get_section_settings(config, 'plot')
		self.data = self.get_section_settings(config, 'data')


	def get_section_settings(self, config, section):
		settings_list = config.items(section)
		settings_dict = dict(settings_list)
		keys = settings_dict.keys()
		key_string = ' '.join(keys)
		SettingsStruct = namedtuple('SettingsStruct', key_string)
		settings = SettingsStruct(**settings_dict)
		return settings



if __name__ == '__main__':
	main()

