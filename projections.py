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
	if len(sys.argv) > 1:
		config_file = sys.argv[1]

	# read config file
	settings = Settings(config_file)

	# read in data
	revenue = Revenue(settings.data_settings)

	# generate plots
	make_plots(revenue, settings)

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


	def get_revenue_type(self, months, customers, estimate_type='estimate', revenue_type=None):
		if revenue_type == 'net':
			return self.net(months, customers, estimate_type=estimate_type)
		elif revenue_type == 'income':
			return self.gross_income(months, customers, estimate_type=estimate_type)
		elif revenue_type == 'expenses':
			return self.expenses(months, customers, estimate_type=estimate_type)
		else:
			print('Please provide a revenue type of "net", "income", or "expenses" in the settings section.')
			sys.exit(704198)


	def net(self, months, customers, estimate_type='estimate'):
		net = self.gross_income(months, customers, estimate_type) - self.expenses(months, customers, estimate_type)
		return net


	def gross_income(self, months, customers, estimate_type):
		if estimate_type == 'estimate':
			income = self.income.estimate[:, np.newaxis]
		elif estimate_type == 'conservative':
			income = self.income.conservative[:, np.newaxis]
		elif estimate_type == 'aggressive':
			income = self.income.aggressive[:, np.newaxis]

		month_rate = self.income.month_rate[:, np.newaxis]
		customer_rate = self.income.customer_rate[:, np.newaxis]

		line_income = income * np.divide(months, month_rate) * np.floor(np.divide(customers, customer_rate))
		gross_income = np.sum(line_income, axis=0)
		return gross_income


	def expenses(self, months, customers, estimate_type):
		expenses = self.fixed_expenses(months, estimate_type) + self.variable_expenses(months, customers, estimate_type)
		return expenses


	def fixed_expenses(self, months, estimate_type):
		if estimate_type == 'estimate':
			fixed_costs = self.fixed_costs.estimate[:, np.newaxis]
		elif estimate_type == 'conservative':
			fixed_costs = self.fixed_costs.conservative[:, np.newaxis]
		elif estimate_type == 'aggressive':
			fixed_costs = self.fixed_costs.aggressive[:, np.newaxis]

		month_rate = self.fixed_costs.month_rate[:, np.newaxis]

		line_costs = fixed_costs * (months / month_rate)
		expenses = np.sum(line_costs, axis=0)
		return expenses


	def variable_expenses(self, months, customers, estimate_type):
		if estimate_type == 'estimate':
			variable_costs = self.variable_costs.estimate[:, np.newaxis]
		elif estimate_type == 'conservative':
			variable_costs = self.variable_costs.conservative[:, np.newaxis]
		elif estimate_type == 'aggressive':
			variable_costs = self.variable_costs.aggressive[:, np.newaxis]

		month_rate = self.variable_costs.month_rate[:, np.newaxis]
		customer_rate = self.variable_costs.customer_rate[:, np.newaxis]

		line_costs = variable_costs * (months / month_rate) * np.floor(customers / customer_rate)
		expenses = np.sum(line_costs, axis=0)
		return expenses



class Settings:
	def __init__(self, config_file):
		config = configparser.ConfigParser()
		config.read(config_file)

		self.data_settings = self.get_section_settings(config, 'data')

		self.time_net_plot_settings      = self.get_section_settings(config, 'time_net_plot')
		self.time_income_plot_settings   = self.get_section_settings(config, 'time_income_plot')
		self.time_expenses_plot_settings = self.get_section_settings(config, 'time_expenses_plot')

		self.customer_net_plot_settings      = self.get_section_settings(config, 'customer_net_plot')
		self.customer_income_plot_settings   = self.get_section_settings(config, 'customer_income_plot')
		self.customer_expenses_plot_settings = self.get_section_settings(config, 'customer_expenses_plot')

		self.customer_time_net_plot_settings      = self.get_section_settings(config, 'customer_time_net_plot')
		self.customer_time_income_plot_settings   = self.get_section_settings(config, 'customer_time_income_plot')
		self.customer_time_expenses_plot_settings = self.get_section_settings(config, 'customer_time_expenses_plot')


	def get_section_settings(self, config, section):
		settings_list = config.items(section)
		settings_list = [self.convert_type(x) for x in settings_list]
		settings_dict = dict(settings_list)
		keys = settings_dict.keys()
		key_string = ' '.join(keys)
		SettingsStruct = namedtuple('SettingsStruct', key_string)
		settings = SettingsStruct(**settings_dict)
		return settings


	def convert_type(self, setting):
		key = setting[0]
		value = setting[1]
		if type(value) == str:
			if (value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"):
				value = value[1:-1]
			elif value[0] == 'r' and ((value[1] == '"' and value[-1] == '"') or (value[1] == "'" and value[-1] == "'")):
				value = r"%s" % value[2:-1]
			elif value == 'True' or value == 'true':
				value = True
			elif value == 'False' or value == 'false':
				value = False
			elif value == 'None' or value == 'none':
				value = None
			elif '.' in value:
				value = float(value)
			else:
				value = int(value)
		else:
			sys.exit(18235)

		return (key, value)



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
	if settings.time_net_plot_settings.make_plot:
		fig = make_time_plot(revenue, settings.time_net_plot_settings)
	if settings.time_income_plot_settings.make_plot:
		fig = make_time_plot(revenue, settings.time_income_plot_settings)
	if settings.time_expenses_plot_settings.make_plot:
		fig = make_time_plot(revenue, settings.time_expenses_plot_settings)

	if settings.customer_net_plot_settings.make_plot:
		fig = make_customer_plot(revenue, settings.customer_net_plot_settings)
	if settings.customer_income_plot_settings.make_plot:
		fig = make_customer_plot(revenue, settings.customer_income_plot_settings)
	if settings.customer_expenses_plot_settings.make_plot:
		fig = make_customer_plot(revenue, settings.customer_expenses_plot_settings)

	if settings.customer_time_net_plot_settings.make_plot:
		fig = make_customer_time_plot(revenue, settings.customer_time_net_plot_settings)
	if settings.customer_time_income_plot_settings.make_plot:
		fig = make_customer_time_plot(revenue, settings.customer_time_income_plot_settings)
	if settings.customer_time_expenses_plot_settings.make_plot:
		fig = make_customer_time_plot(revenue, settings.customer_time_expenses_plot_settings)



def make_time_plot(revenue, settings):
	fig, ax = setup_plot(settings)

	months = get_independent_variable(settings)
	for customer_function, line_color, fill_color, label in zip( \
			[constant_customers, linear_customers, exponential_customers], \
			[settings.constant_line_color, settings.linear_line_color, settings.exponential_line_color], \
			[settings.constant_fill_color, settings.linear_fill_color, settings.exponential_fill_color], \
			[settings.constant_label, settings.linear_label, settings.exponential_label] \
			):
		line, range_lines = add_time_plot_lines(ax, months, revenue, customer_function, line_color, fill_color, label, settings)
	ax.legend(loc='lower right', fontsize='x-large')
	set_line_plot_params(ax, settings)

	fig = save_plot(fig, settings)
	return fig



def add_time_plot_lines(ax, months, revenue, customer_function, line_color, fill_color, label, settings):
	revenue_estimate = revenue.get_revenue_type( \
			months, \
			customer_function(months, settings), \
			estimate_type='estimate', \
			revenue_type=settings.revenue_type \
			)
	revenue_conservative = revenue.get_revenue_type( \
			months, \
			customer_function(months, settings), \
			estimate_type='conservative', \
			revenue_type=settings.revenue_type \
			)
	revenue_aggressive = revenue.get_revenue_type( \
			months, \
			customer_function(months, settings), \
			estimate_type='aggressive', \
			revenue_type=settings.revenue_type \
			)

	lines = ax.fill_between(months, revenue_conservative, revenue_aggressive, \
			color=line_color, \
			facecolor=fill_color, \
			alpha=settings.transparency_alpha, \
			label=label \
			)
	line = ax.plot(months, revenue_estimate, \
			linestyle='-', \
			color=line_color \
			)
	return line, lines



def constant_customers(months, settings):
	customers = settings.constant_customers_slice
	return customers



def linear_customers(months, settings):
	customers = months * settings.linear_customers_rate
	return customers



def exponential_customers(months, settings):
	customers = np.exp(months**(settings.exponential_customers_month_power) / settings.exponential_customers_folding_time)
	return customers



def make_customer_plot(revenue, settings):
	fig, ax = setup_plot(settings)

	customers = get_independent_variable(settings)
	revenue_estimate = revenue.get_revenue_type( \
			settings.time_slice, \
			customers, estimate_type='estimate', \
			revenue_type=settings.revenue_type \
			)
	revenue_conservative = revenue.get_revenue_type( \
			settings.time_slice, \
			customers, \
			estimate_type='conservative', \
			revenue_type=settings.revenue_type \
			)
	revenue_aggressive = revenue.get_revenue_type( \
			settings.time_slice, \
			customers, \
			estimate_type='aggressive', \
			revenue_type=settings.revenue_type \
			)

	lines = ax.fill_between(customers, revenue_conservative, revenue_aggressive, facecolor=settings.fill_color, alpha=settings.transparency_alpha)
	line = ax.plot(customers, revenue_estimate, linestyle='-')
	set_line_plot_params(ax, settings)

	fig = save_plot(fig, settings)
	return fig



def make_customer_time_plot(revenue, settings):
	fig, ax = setup_plot(settings)
	fig = save_plot(fig, settings)
	return fig



def setup_plot(settings):
	print('Making plot...')
	fig = plt.figure(figsize = (9.0, 6.0))
	ax = fig.add_subplot(111)
	return fig, ax



def save_plot(fig, settings):
	name = settings.plot_name
	print('Saving %s plot.' % name)
	if settings.set_tight_layout:
		fig.tight_layout()
	plt.savefig(name, bbox_inches='tight')
	return fig



def get_independent_variable(settings):
	if settings.x_log:
		x = np.logspace(settings.x_min, settings.x_max, num=settings.npoints)
	else:
		x = np.linspace(settings.x_min, settings.x_max, num=settings.npoints)
	return x



def set_line_plot_params(ax, settings):
	if settings.x_log:
		ax.set_xscale('log')
	if settings.y_log:
		ax.set_yscale('log')

	null_list = [None, 'None', 'none', 'Auto', 'auto']
	if settings.y_min not in null_list and settings.y_max not in null_list:
		ax.set_ylim(settings.y_min, settings.y_max)

	ax.set_xlabel(settings.x_axis_label)
	ax.set_ylabel(settings.y_axis_label)



if __name__ == '__main__':
	main()

