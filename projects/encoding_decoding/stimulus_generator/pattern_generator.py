from numpy.random import choice
from random import randint, random
import numpy as np
from modules.parameters import ParameterSet


class StimulusPattern(object):
	"""
	Class to generate complex stimulus sets
	(specific to encoding_decoding project)
	"""

	def __init__(self, initializer):
		"""

		:param initializer: task parameter set
		"""
		assert(isinstance(initializer, ParameterSet)), "initializer must be ParameterSet object"

		if initializer.task == 1:
			self.task_description = 'identity mapping'
		elif initializer.task == 2:
			self.task_description = 'delayed identity mapping'
		elif initializer.task == 3:
			self.task_description = 'delayed identity mapping with distractors'
		elif initializer.task == 4:
			self.task_description = 'adjacent dependencies'
		elif initializer.task == 5:
			self.task_description = 'non-adjacent dependencies'
		elif initializer.task == 6:
			self.task_description = 'pattern mapping with cross dependencies'
		elif initializer.task == 7:
			self.task_description = 'hierachical dependencies'
		else:
			raise TypeError("Incorrect task specification!")
		self.parameters = initializer
		self.lexicon = ['A'+str(i) for i in range(initializer.lexicon_size+1)]
		if initializer.task in [4, 5, 7]:
			self.lexicon = [str(i) for i in range(initializer.lexicon_size+5)]
		self.lexicon = self.lexicon[:initializer.lexicon_size]
		self.EOS = ['.']
		self.FileName = 'Task_' + str(self.task_description) + '_Delay_' + str(initializer.dt) + '_LexSize_' + str(
			initializer.lexicon_size) + '.txt'

		self.Input = []
		self.Output = {'ID': [], 'Accepted': []}
		self.Language = {}

	def filler(self, number=-1):
		if number != -1:
			return ['x' + str(randint(0, self.parameters.n_distractors)) for _ in range(number)]
		else:
			return 'x'

	def ran_dt(self):
		if self.parameters.random_dt:
			return randint(0, self.parameters.dt)
		else:
			return self.parameters.dt

	def accepted_pattern(self, task, pattern):
		if task == 4:
			AC = ['A'+str(i)+'B'+str(i) for i in range(self.parameters.lexicon_size+1)]
			#AC = ['A1B1', 'A2B2', 'A3B3', 'A4B4', 'A5B5', 'A6B6']
			# AC = ['A1A2','B1B2','C1C2','D1D2','E1E2','F1F2']
			# AC = ['AB','CD','EF']
			if pattern in AC:
				return 'A'
			else:
				return 'R'
		if task == 5:
			AC = ['A'+str(i)+'B'+str(i) for i in range(self.parameters.lexicon_size+1)]
			#AC = ['A1B1', 'A2B2', 'A3B3', 'A4B4', 'A5B5', 'A6B6']
			# AC = ['A1A2','B1B2','C1C2','D1D2','E1E2','F1F2']
			if pattern in AC:
				return 'A'
			else:
				return 'R'

	def AddHierarchy(self, length):
		if length == 0:
			return [], [], []
		else:
			item = choice(self.lexicon)
			inp, O_id, O_acc = self.AddHierarchy(length - 2)
			if random() < 0.7:
				return ['A' + item] + inp + ['B' + item], [self.filler()] + O_id + ['A' + item + 'B' + item], [
					'R'] + O_acc + ['A']
			else:
				f_item = choice(self.lexicon)
				while f_item == item:
					f_item = choice(self.lexicon)
				return ['A' + f_item] + inp + ['B' + item], [self.filler()] + O_id + ['A' + f_item + 'B' + item], [
					'R'] + O_acc + ['R']

	def AddPattern(self, task, Input, Output):

		if task == 1:
			item = choice(self.lexicon)
			Input.append(item)
			Output['ID'].append(item)
			Output['Accepted'].append('A')
		elif task == 2:
			item = choice(self.lexicon)
			Dist = self.parameters.dt
			Input += [item] + self.filler(Dist) + self.filler(self.parameters.pause_t)
			Output['ID'] += [self.filler()] * Dist + [item] + self.EOS + [self.filler()] * self.parameters.pause_t
			Output['Accepted'] += ['R'] * Dist + ['A'] + ['R'] + ['R'] * self.parameters.pause_t
		elif task == 3:
			if (len(Input)+1)%20==0:
				Input += self.EOS
				Output['ID'] += self.EOS
				Output['Accepted'].append('R')
			else:
				item = choice(self.lexicon)
				Input.append(item)
				if (len(Input))%20 < self.parameters.dt+1:
					Output['ID'].append(self.filler())
					Output['Accepted'].append('R')
				else:
					Output['ID'].append(Input[-self.parameters.dt - 1])
					Output['Accepted'].append('A')
		elif task == 4:
			current_pattern = ['A' + choice(self.lexicon), 'B' + choice(self.lexicon)]
			Input += current_pattern + self.EOS + self.filler(self.parameters.pause_t)
			current_pattern = ''.join(current_pattern)
			Output['ID'] += [self.filler()] * (self.parameters.C_len - 1) + [current_pattern] + self.EOS + [
				            self.filler()] * self.parameters.pause_t
			Output['Accepted'] += ['R'] * (self.parameters.C_len - 1) + [self.accepted_pattern(task, current_pattern)]\
			                      + ['R'] + ['R'] * self.parameters.pause_t
		elif task == 5:
			current_pattern = ['A' + choice(self.lexicon), 'B' + choice(self.lexicon)]
			Dist = self.ran_dt()
			newinp = [item for subl in [[item] + self.filler(Dist) for item in current_pattern] for item in subl][
			         :-Dist]
			newid = [self.filler()] * len(newinp)
			newid[-1] = ''.join(current_pattern)
			newacc = ['R'] * len(newinp)
			newacc[-1] = self.accepted_pattern(task, ''.join(current_pattern))
			Input += newinp + self.EOS + self.filler(self.parameters.pause_t)
			Output['ID'] += newid + self.EOS + [self.filler()] * self.parameters.pause_t
			Output['Accepted'] += newacc + ['R'] + ['R'] * self.parameters.pause_t

		elif task == 7:
			item = choice(self.lexicon)
			Dist = self.ran_dt()
			newinp, newid, newacc = self.AddHierarchy(Dist * 2)
			i = 0
			while i < len(newinp):
				if random() < 0.1:
					i += 1
					newinp.insert(i, self.filler(1)[0])
					newid.insert(i, self.filler())
					newacc.insert(i, 'R')
				i += 1

			Input += newinp + self.EOS + self.filler(self.parameters.pause_t)
			Output['ID'] += newid + self.EOS + [self.filler()] * self.parameters.pause_t
			Output['Accepted'] += newacc + ['R'] + ['R'] * self.parameters.pause_t

		return Input, Output

	def generate(self, T=0):

		print "\n*****\n Generating stimulus set \n*****"
		Input = self.Input
		Output = self.Output
		Lang = self.Language
		if T == 0:
			T = self.parameters.T
		for t in range(T):
			Input, Output = self.AddPattern(self.parameters.task, Input, Output)
		self.Input = Input[:T]
		self.Output['ID'] = Output['ID'][:T]
		self.Output['Accepted'] = Output['Accepted'][:T]

		self.Language['Task number'] = self.parameters.task
		self.Language['Task description'] = self.task_description
		self.Language['Lexicon size'] = self.parameters.lexicon_size
		self.Language['Filler size'] = self.parameters.n_distractors
		self.Language['Sequence length'] = self.parameters.T
		self.Language['Maximum delay distance'] = self.parameters.dt
		self.Language['Pause length'] = self.parameters.pause_t
		self.Language['Input set'] = list(set(self.Input))
		self.Language['Output ID set'] = list(set(self.Output['ID']))
		self.Language['Output Accepted set'] = list(set(self.Output['Accepted']))

		print 'Input     :', self.Input[:10]
		print 'Output ID :', self.Output['ID'][:10]
		print 'Output Accepted :', self.Output['Accepted'][:10]

		return self.Input, self.Output, self.Language

	def as_index(self):
		"""
		Return the Input and Output sequences as sequences of ints
		:return:
		"""
		symbols_in = np.unique(self.Input)
		symbols_out = np.unique(self.Output['ID'])
		# print [np.where(k == symbols_in)[0] for k in self.Input], [np.where(k == symbols_out)[0] for k in self.Output[
		# 	'ID']]
		return [int(np.where(k == symbols_in)[0]) for k in self.Input], [int(np.where(k == symbols_out)[0]) for k in
		                                                              self.Output['ID']]

	def save(self, path=''):
		if path:
			self.FileName = path + self.FileName
		with open(self.FileName, 'w') as outfile:
			outfile.write('Lang = ' + str(self.Language) + '\n')
			outfile.write('Input' + '\n')
			outfile.write(' '.join(self.Input) + '\n')
			outfile.write('Output ID' + '\n')
			outfile.write(' '.join(self.Output['ID']) + '\n')
			outfile.write('Output Ac' + '\n')
			outfile.write(' '.join(self.Output['Accepted']) + '\n')

	def load(self, task=1, delay=0, lexicon_size=0, FileName='', as_index=True):
		parameters = {'task': task, 'lexicon_size': lexicon_size, 'T': 0, 'random_dt': None, 'dt': delay, 'pause_t':
			None, 'C_len': None}
		self.__init__(ParameterSet(parameters))
		if not FileName:
			FileName = self.FileName

		with open(FileName, 'r') as infile:
			i = 0
			Output = {}
			for line in infile:
				if i == 0:
					exec (line)
				elif i == 2:
					Input = line.strip().split(' ')
				elif i == 4:
					Output['ID'] = line.strip().split(' ')
				elif i == 6:
					Output['Accepted'] = line.strip().split(' ')
				i += 1

		if as_index:
			Input_index = []
			Output_index = {}
			Output_index['ID'] = []
			Output_index['Accepted'] = []
			for i in range(len(Input)):
				Input_index.append(self.Language['Input set'].index(Input[i]))
				Output_index['ID'].append(self.Language['Output ID set'].index(Output['ID'][i]))
				Output_index['Accepted'].append(self.Language['Output Accepted set'].index(Output['Accepted'][i]))
			(self.Language, self.Input, self.Output) = (self.Language, Input_index, Output_index)
			return self.Language, Input_index, Output_index
		else:
			(self.Language, self.Input, self.Output) = (self.Language, Input, Output)
			return self.Language, Input, Output



