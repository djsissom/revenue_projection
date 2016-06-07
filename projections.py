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
	revenue = Revenue(settings.data_settings)

	# process data

	# generate plots

	print("Finished.")
	sys.exit()



def read_files(files, header_line=None, comment_char='#', rec_array=False):
	header = None
	data = None
	if type(files) == str:
		files = [files]

	if header_line != None:
		with open(files[0], 'r') as fd:
			for line in range(header_line):
				fd.readline()
			header = fd.readline()
		if header[0] != comment_char:
			print "Header must start with a '%s'." % comment_char
			sys.exit(4)
		header = header[1:]
		header = header.split()

	for file in files:
		print "Reading file%s..." % (file)
		if data == None:
			if rec_array:
				data = np.genfromtxt(file, dtype=None, comments=comment_char, names=header, deletechars='[]/|')
				data = data.view(np.recarray)
			else:
				data = np.genfromtxt(file, dtype=None, comments=comment_char)
		else:
			if rec_array:
				data = np.append(data, np.genfromtxt(file, dtype=None, comments=comment_char, names=header, deletechars='[]/|'), axis=0)
				data = data.view(np.recarray)
			else:
				data = np.append(data, np.genfromtxt(file, dtype=None, comments=comment_char), axis=0)

	if header_line == None:
		return data
	else:
		return header, data



class Revenue:
	def __init__(self, settings):
		self.fixed_costs = read_files(settings.fixed_costs_file)
		self.variable_costs = read_files(settings.variable_costs_file)
		self.income = read_files(settings.income_file)



class Settings:
	def __init__(self, config_file):
		config = configparser.ConfigParser()
		config.read(config_file)
		self.plot_settings = self.get_section_settings(config, 'plot')
		self.data_settings = self.get_section_settings(config, 'data')


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

