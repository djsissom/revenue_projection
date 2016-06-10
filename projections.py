#!/usr/bin/env python

import sys
import configparser
from collections import namedtuple
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from ipdb import set_trace



def main():
	'''
	projections.py

	Calculate and plot revenue projections as a function of time and number of customers.
	'''

	# get arguments/filenames
	config_file = 'settings.conf'
	if len(sys.argv > 1):
		config_file = sys.argv[1]

	# read config file
	settings = Settings(config_file)

	# read in data
	revenue = Revenue(settings.data_settings)

	# process data
	set_trace()

	# generate plots
	make_plots(revenue)

	print("Finished.")
	sys.exit()



class Revenue:
	def __init__(self, settings):
		fields = (['cost', 'month_rate', 'conservative', 'estimate', 'aggressive'],
		          ['cost', 'month_rate', 'customer_rate', 'conservative', 'estimate', 'aggressive'],
		          ['income', 'month_rate', 'customer_rate', 'conservative', 'estimate', 'aggressive'])
		self.fixed_costs    = read_files(settings.fixed_costs_file,    rec_array=True, fields=fields[0])
		self.variable_costs = read_files(settings.variable_costs_file, rec_array=True, fields=fields[1])
		self.income         = read_files(settings.income_file,         rec_array=True, fields=fields[2])
		del fields


	def net(self, months, customers, estimate_type='estimate'):
		net = gross_income(months, customers, estimate_type) - expenses(months, customers, estimate_type)
		return net


	def gross_income(self, months, customers, estimate_type):
		if estimate_type == 'estimate':
			income = self.income.estimate
		elif estimate_type == 'conservative':
			income = self.income.conservative
		elif estimate_type = 'aggressive':
			income = self.income.aggressive

		month_rate = self.income.month_rate
		customer_rate = self.income.customer_rate

		line_income = income * (months / month_rate) * np.floor(customers / customer_rate)
		gross_income = np.sum(line_income)
		return gross_income


	def expenses(self, months, customers, estimate_type):
		expenses = fixed_expenses(months, estimate_type) + variable_expenses(months, customers, estimate_type)
		return expenses


	def fixed_expenses(self, months, estimate_type):
		if estimate_type == 'estimate':
			fixed_costs = self.fixed_costs.estimate
		elif estimate_type == 'conservative':
			fixed_costs = self.fixed_costs.conservative
		elif estimate_type = 'aggressive':
			fixed_costs = self.fixed_costs.aggressive

		month_rate = self.fixed_costs.month_rate

		line_costs = fixed_costs * (months / month_rate)
		expenses = np.sum(line_costs)

		return expenses


	def variable_expenses(self, months, customers, estimate_type):
		if estimate_type == 'estimate':
			variable_costs = self.variable_costs.estimate
		elif estimate_type == 'conservative':
			variable_costs = self.variable_costs.conservative
		elif estimate_type = 'aggressive':
			variable_costs = self.variable_costs.aggressive

		month_rate = self.variable_costs.month_rate
		customer_rate = self.variable_costs.customer_rate

		line_costs = variable_costs * (months / month_rate) * np.floor(customers / customer_rate)
		expenses = np.sum(line_costs)

		return expenses



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



def read_files(files, header_line=None, comment_char='#', rec_array=False, fields=None):
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
			print("Header must start with a '%s'." % comment_char)
			sys.exit(4)
		header = header[1:]
		header = header.split()

	if rec_array and fields == None and header != None:
		fields = header

	for file in files:
		print("Reading file%s..." % (file))
		if data == None:
			if rec_array:
				data = np.genfromtxt(file, dtype=None, comments=comment_char, names=fields, deletechars='[]/|')
				data = data.view(np.recarray)
			else:
				data = np.genfromtxt(file, dtype=None, comments=comment_char)
		else:
			if rec_array:
				data = np.append(data, np.genfromtxt(file, dtype=None, comments=comment_char, names=fields, deletechars='[]/|'), axis=0)
				data = data.view(np.recarray)
			else:
				data = np.append(data, np.genfromtxt(file, dtype=None, comments=comment_char), axis=0)

	if header_line == None:
		return data
	else:
		return header, data



def make_plots(revenue, settings):
	fig = make_time_plot(revenue)
	fig = make_customer_plot(revenue)
	fig = make_customer_time_plot(revenue)



def make_time_plot(revenue, settings):
	fig, ax = setup_plot()
	ax = make_line_plot(ax, x_data, y_data)
	fig = save_plot(fig, plot_name)
	return fig



def make_customer_plot(revenue, settings):
	fig, ax = setup_plot()
	fig = save_plot(fig, plot_name)
	return fig



def make_customer_time_plot(revenue, settings):
	fig, ax = setup_plot()
	fig = save_plot(fig, plot_name)
	return fig



def setup_plot():
	print('Making plot...')
	fig = plt.figure(figsize = (9.0, 6.0))
	ax = fig.add_subplot(111, aspect='equal')
	return fig, ax



def save_plot(fig, name):
	print('Saving %s plot.' % name)
	fig.tight_layout()
	plt.savefig(name, bbox_inches='tight')
	return fig



def make_line_plot(ax, x_data, y_data, xlim=None, ylim=None):
	pass



if __name__ == '__main__':
	main()

