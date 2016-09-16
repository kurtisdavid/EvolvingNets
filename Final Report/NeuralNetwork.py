# -*- coding: UTF-8 -*-
from math import *
import random
import copy
from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt


# implementation for GA
class NeuralNetwork:

	innovation_number = 3

	def __init__(self, input_vals):

		self.inputs = []

		for i in range(len(input_vals)):

			self.inputs.append(input_vals[i])

		self.neurons = []

		self.connections = []

		self.fitness = None

		bias_neuron = Neuron("bias")

		in_neuron1 = Neuron("input")
		in_neuron1.input = self.inputs[0]
		in_neuron1.output = in_neuron1.calculateOutput()

		in_neuron2 = Neuron("input")
		in_neuron2.input = self.inputs[1]
		in_neuron2.output = in_neuron2.calculateOutput()

		out_neuron = Neuron("output")
		

		self.neurons.append(bias_neuron)
		self.neurons.append(in_neuron1)
		self.neurons.append(in_neuron2)
		self.neurons.append(out_neuron)


		primary_connection = Connection(random.uniform(-10,10), in_neuron1, out_neuron, 1)
		primary_connection1 = Connection(random.uniform(-10,10), in_neuron2, out_neuron, 2)
		bias_connection = Connection(random.uniform(-10,10), bias_neuron, out_neuron, 3)

		self.neurons[1].connections.append(primary_connection)
		self.neurons[2].connections.append(primary_connection1)
		self.neurons[0].connections.append(bias_connection)

		self.neurons[3].connections.append(primary_connection)
		self.neurons[3].connections.append(primary_connection1)
		self.neurons[3].connections.append(bias_connection)


		for i in range(len(self.neurons)):

			current = self.neurons[i]

			for j in range(len(current.connections)):

				if current.connections[j] not in self.connections:

					self.connections.append(current.connections[j])

	def __str__(self):

		answer = "\n"

		list_c = []
		connect_list = []


		for i in range(len(self.connections)):

			if self.connections[i].enabled == True:

				list_c.append(str(self.neurons.index(self.connections[i].neurons[0])) + "->" + str(self.neurons.index(self.connections[i].neurons[1])) + " [weight: " + str(self.connections[i].weight) + ", innovation number: " + str(self.connections[i].counter) + "]")

		for i in range(len(list_c)):

			answer += list_c[i] +"\n"

		answer += "Fitness: " + str(self.fitness)


		return answer

	# for mutation
	def addConnection(self):

		index1 = random.randint(0,len(self.neurons)-1)

		while index1==3:

			index1 = random.randint(0,len(self.neurons)-1)

		validNeurons = []

		for i in range(3,len(self.neurons)):

			c_neuron = self.neurons[i]

			found = False

			for j in range(len(self.neurons[index1].connections)):

				if c_neuron in self.neurons[index1].connections[j].neurons:

					found = True

			if not found:

				validNeurons.append(c_neuron)

		# if not possible, we can look for a possible connection to enable
		if len(validNeurons) == 0:

			for i in range(len(self.connections)):

				if self.connections[i].enabled == False:

					neuron_1 = self.connections[i].neurons[0]
					neuron_2 = self.connections[i].neurons[1]

					works = True

					for j in range(len(neuron_1.connections)):

						if neuron_1.connections[j]!=self.connections[i] and neuron_2 in neuron_1.connections[j].neurons:

							works = False
							break

					if works == True:

						self.connections[i].enabled = True

					return


			return

		index2 = random.randint(0,len(validNeurons)-1)

		list_tried = []

		while not isValid(self.neurons[index1], validNeurons[index2]) and len(list_tried)<len(validNeurons):

			if validNeurons[index2] not in list_tried:

				list_tried.append(validNeurons[index2])

			index2 = random.randint(0,len(validNeurons)-1)

		if len(list_tried)==len(validNeurons):
			return


		chosen_neurons = [self.neurons[index1],validNeurons[index2]]

		# adds new connection
		new_connect = Connection(random.uniform(-5,5),chosen_neurons[0],chosen_neurons[1], NeuralNetwork.innovation_number+1)
		chosen_neurons[0].connections.append(new_connect)
		chosen_neurons[1].connections.append(new_connect)

		NeuralNetwork.innovation_number += 1

		self.connections.append(new_connect)

	# adds new neuron to ANN
	def addNeuron(self):

		new_neuron = Neuron("hidden")

		index1 = random.randint(0,len(self.neurons)-1)

		connected = False

		neuron1 = self.neurons[index1]
		neuron2 = self.neurons[0]

		while connected == False:

			index2 = generateSecond(len(self.neurons),index1, self.neurons)
			neuron2 = self.neurons[index2]

			for i in range(len(neuron1.connections)):

				current = neuron1.connections[i]

				if neuron2 in current.neurons:

					connected = True
					break

		shared_c = []

		for i in range(len(neuron1.connections)):

			if neuron2 in neuron1.connections[i].neurons:

				shared_c.append(neuron1.connections[i])


		chosen_connection = shared_c[0]
		chosen_connection.enabled = False

		self.neurons.append(new_neuron)

		# UPDATE THIS
		new_connection1 = Connection(random.uniform(-10,10), chosen_connection.neurons[0], new_neuron, NeuralNetwork.innovation_number+1)
		chosen_connection.neurons[0].connections.append(new_connection1)
		new_neuron.connections.append(new_connection1)
		NeuralNetwork.innovation_number += 1

		new_connection2 = Connection(random.uniform(-10,10), new_neuron, chosen_connection.neurons[1], NeuralNetwork.innovation_number+1)
		new_neuron.connections.append(new_connection2)
		chosen_connection.neurons[1].connections.append(new_connection2)
		NeuralNetwork.innovation_number += 1

		self.connections.append(new_connection1)
		self.connections.append(new_connection2)


	# Copy ANN for GA methods
	def copyNetwork(self):
		
		new_neurons = []

		for i in range(len(self.neurons)):

			current = self.neurons[i]

			new_copy = Neuron(current.type)

			if current.type == "input":

				new_copy.input = current.input
				new_copy.output = current.output

			new_neurons.append(new_copy)

		new_net = NeuralNetwork(self.inputs)

		new_connections = []


		for i in range(len(self.neurons)):

			current = self.neurons[i]
			c_connections = current.connections

			for j in range(len(c_connections)):

				current_con = c_connections[j]

				index1 = self.neurons.index(current_con.neurons[0])
				index2 = self.neurons.index(current_con.neurons[1])

				new_connection = Connection(current_con.weight, new_neurons[index1], new_neurons[index2], current_con.counter)
				new_connection.enabled = current_con.enabled

				if new_connection not in new_connections:

					new_connections.append(new_connection)
					new_neurons[index1].connections.append(new_connection)
					new_neurons[index2].connections.append(new_connection)

		new_net.neurons = new_neurons
		new_net.connections = new_connections
		new_net.fitness = self.fitness

		return new_net

	def finalOutput(self):

		output_neuron = Neuron("")

		for i in range(len(self.neurons)):

			if self.neurons[i].type == "output":
				output_neuron = self.neurons[i]
				break

		check = 0
		try:
			check = recurOutput(output_neuron)
		except RuntimeError:
			print("UH OH")
			print(self)
		
		return check

	def resetOutput(self):

		for i in range(len(self.neurons)):

			if self.neurons[i].type != "input":

				self.neurons[i].input = 0

				if self.neurons[i].type == "bias":

					self.neurons[i].output = 1

				else:

					self.neurons[i].output = 0


def recurOutput(neuron):

	if neuron.output != 0:

		return neuron.output

	else:

		sum_t = 0
		for i in range(len(neuron.connections)):
			current = neuron.connections[i]

			if current.enabled:
				if current.neurons[1] == neuron:
					sum_t += recurOutput(current.neurons[0]) * current.weight

		neuron.input = sum_t

		neuron.output = neuron.calculateOutput()
		return neuron.output

def generateSecond(length,index1, neurons):

	index2 = random.randint(0,length-1)

	while index2==index1:

		index2 = random.randint(0,length-1)

	return index2

# checks to see if a connection can be made between neuron1 and neuron2
def isValid(neuron1, neuron2):

	ans = True

	for i in range(len(neuron1.connections)):

		current = neuron1.connections[i]

		if neuron1 == current.neurons[1]:

			ans = not isFound(current.neurons[0], neuron2)

	return ans

# helper method for isValid
def isFound(neuron, wanted):

	if neuron.type == "input":

		return False

	else:

		ans = False
		for i in range(len(neuron.connections)):

			current_c = neuron.connections[i]
			if neuron == current_c.neurons[1]:

				if current_c.neurons[0] == wanted:
					return True
				else:
					ans = isFound(current_c.neurons[0],wanted)

		return ans


# object for Neuron
class Neuron:

	def __init__(self, type_neuron):
		self.connections = []
		self.type = type_neuron
		self.input = 0
		self.output = 0

		if self.type == "bias":
			self.output = 1


	def calculateOutput(self):

		check = sigmoid(self.input)

		return check


# object for Connection
class Connection:

	def __init__(self, weight, neuron1, neuron2, innovation_number):
		self.weight = weight

		self.neurons = [neuron1,neuron2]
		self.counter = innovation_number
		self.enabled = True


	def __eq__(self, other):

		if self.weight == other.weight and self.neurons[0] == other.neurons[0] and self.neurons[1] == other.neurons[1] and self.counter == other.counter and self.enabled==other.enabled:

			return True

		return False		

def sigmoid(x):

	ans = ""
	try:
		ans = 1/(1+exp(x))
	except OverflowError:
		print("This x value is too big:" + str(x))
		ans = None

	return ans


# mutation that either adds a Neuron to a net individual, or a Connection.
def mutate(individual, indpb):

	net = individual[0]
	test = random.random()
	if test < indpb:
		p = random.random()

		if p < .6:

			for i in range(len(net.connections)):

				n = random.uniform(-1,1)
				net.connections[i].weight += n

		else:

			x = random.random()

			if x<.4:

				net.addNeuron()

			else:

				net.addConnection()

# how to crossover indviduals
def crossover(ind1, ind2):

	new_net1 = ind1[0].copyNetwork()
	new_net2 = ind2[0].copyNetwork()

	if new_net1.fitness > new_net2.fitness:

		for i in range(len(new_net1.connections)):

			current = new_net1.connections[i]

			for j in range(len(new_net2.connections)):

				if new_net2.connections[j].counter == current.counter:

					x = random.random()

					temp = new_net2.connections[j].weight

					new_net2.connections[j].weight = new_net2.connections[j].weight*x + current.weight*(1-x)
					current.weight = current.weight*x + temp*(1-x)

	elif new_net2.fitness >= new_net1.fitness:

		for i in range(len(new_net2.connections)):

			current = new_net2.connections[i]

			for j in range(len(new_net1.connections)):

				if new_net1.connections[j].counter == current.counter:

					x = random.random()

					temp = new_net1.connections[j].weight

					new_net1.connections[j].weight = new_net1.connections[j].weight*x + current.weight*(1-x)
					current.weight = current.weight*x + temp*(1-x)

	child1 = creator.Individual()
	child2 = creator.Individual()

	child1.append(new_net1)
	child2.append(new_net2)

	return child1, child2




# fitness function for GA
def evalFitness(individual):

	net = individual[0]

	input_data = [[0,0],[0,1],[1,0],[1,1]]
	output_data = [0,1,1,0]

	total_error = 0

	for i in range(0,len(input_data)):

		net.resetOutput()

		net.neurons[1].input = input_data[i][0]
		net.neurons[1].output = net.neurons[1].calculateOutput()
		net.neurons[2].input = input_data[i][1]
		net.neurons[2].output = net.neurons[2].calculateOutput()

		output = net.finalOutput()

		total_error += (output_data[i]-output)**2

	return total_error



# tournament selection for GA
def selection(pop, tournsize):

	random_index = random.randint(0,len(pop)-1)
	best = pop[random_index]
	
	for i in range(1, tournsize):
		random_index2 = random.randint(0, len(pop)-1)
		newcomer = pop[random_index2]
		if newcomer.fitness[0]<best.fitness[0]:
			best = newcomer
	return best



def main():

	notes = [[],[],[]]
	notes2 = []
	# First GA and stores data
	for trial in range(3):

		# get DEAP ready to run GA
		creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
		creator.create("Individual", list, fitness=creator.FitnessMin)
		toolbox = base.Toolbox()
		toolbox.register("attr_bool", NeuralNetwork, [0,0])
		toolbox.register("individual", tools.initRepeat, creator.Individual,
	                     toolbox.attr_bool, n=1)
		toolbox.register("population", tools.initRepeat, list, 
	                     toolbox.individual)
		toolbox.register("evaluate", evalFitness)
		toolbox.register("mate", crossover)
		toolbox.register("mutate", mutate, indpb=0.2)
		toolbox.register("select", selection, tournsize=5)

		pop = toolbox.population(n=50)

		trial_results = []

		result = geneticAlgorithm(pop, toolbox, trial_results, cxpb=.5, mutpb = .2, ngen=350)

		notes[trial] = trial_results

		answer = tools.selWorst(pop, k=1)[0]

		net = answer[0].copyNetwork()

		# starts GA2
		# needed since implementation does not optimize weights fast enough in the first GA
		new_net = optimizeWeights(net, notes2)

		input_data = [[0,0],[0,1],[1,0],[1,1]]
		output_data = [0,1,1,0]

		print(new_net)

		for i in range(len(input_data)):

			new_net.neurons[1].input = input_data[i][0]
			new_net.neurons[1].output = new_net.neurons[1].calculateOutput()
			new_net.neurons[2].input = input_data[i][1]
			new_net.neurons[2].output = new_net.neurons[2].calculateOutput()

			output = new_net.finalOutput()

			new_net = new_net.copyNetwork()

		 	print(str(input_data[i][0]) + "⊕" + str(input_data[i][1]) + " = " + str(output))

	generation1 = []
	for i in range(351):
		generation1.append(i)

	# plotting
	plt.figure(1)
	hn0 = notes[0][0]
	hn1 = notes[1][0]
	hn2 = notes[2][0]
	conn0 = notes[0][1]
	conn1 = notes[1][1]
	conn2 = notes[2][1]
	plt.plot(generation1,hn0,"b-.", label="Trial 1 Hidden Neurons")
	plt.plot(generation1,hn1,"g-.", label="Trial 2 Hidden Neurons")
	plt.plot(generation1,hn2,"r-.", label="Trial 3 Hidden Neurons")
	plt.plot(generation1,conn0,"b-", label="Trial 1 Connections")
	plt.plot(generation1,conn1,"g-", label="Trial 2 Connections")
	plt.plot(generation1,conn2,"r-", label="Trial 3 Connections")
	plt.legend(loc="upper left")
	plt.axis([0,350,0,12])
	plt.xlabel("Generation")
	plt.ylabel("Mean Values")
	plt.title("Topological Data per Generation")

	generation2 = []
	for i in range(501):
		generation2.append(i)
	print("x=" + str(len(generation2)), "y=" + str(len(notes2[0])))

	plt.figure(2)
	meanFit0 = notes2[0]
	meanFit1 = notes2[1]
	meanFit2 = notes2[2]
	plt.plot(generation2, meanFit0, "b-", label="Trial 1")
	plt.plot(generation2, meanFit1, "g-", label="Trial 2")
	plt.plot(generation2, meanFit2, "r-", label="Trial 3")
	plt.legend(loc="uppder left")
	plt.axis([0,500,0,1.2])
	plt.xlabel("Generation")
	plt.ylabel("Mean Fitness")
	plt.title("Optimizing Weights of Chosen Topology")


	plt.show()

# GA based on EASimple by DEAP
# altered to match the graphs
def geneticAlgorithm(population, toolbox, notes, cxpb, mutpb, ngen):

	print("Generation 0:")

	# initialize initial fitnesses for first population

	avg_neurons = [] # records hidden neurons per generation
	avg_connections = [] # records active connections per generation
	total_n = 0
	total_c = 0
	for i in range(len(population)):

		population[i].fitness = (toolbox.evaluate(population[i]),)
		population[i][0].fitness = population[i].fitness[0]
		for j in range(len(population[i][0].neurons)):

			if population[i][0].neurons[j].type == "hidden":

				total_n += 1

		for j in range(len(population[i][0].connections)):

			if population[i][0].connections[j].enabled == True:

				total_c += 1

	avg_neurons.append(total_n/(len(population)))
	avg_connections.append(total_c/(len(population)))

	for gen in range(1, ngen+1):

		print("Generation " + str(gen) + ":")
		offspring = []

		total_hidden = 0
		total_conn = 0

		for i in range(len(population)/2):

			parent1 = toolbox.select(population)
			parent2 = toolbox.select(population)

			child1 = creator.Individual()
			child2 = creator.Individual()

			child1.append(parent1[0].copyNetwork())
			child2.append(parent2[0].copyNetwork())

			children = [child1, child2]

			if random.random()<cxpb:

				children[0], children[1] = toolbox.mate(children[0],children[0])

			for j in range(len(children)):

				if random.random()<mutpb:

					toolbox.mutate(children[j])

				fitness_val = toolbox.evaluate(children[j])

				net = children[j][0]
				hidden_counter = 0

				for k in range(len(net.neurons)):

					if net.neurons[k].type == "hidden":

						hidden_counter += 1

				fitness_val += .000125*(hidden_counter**2)
				total_hidden += hidden_counter

				n_connections = 0
				for k in range(len(net.connections)):

					if net.connections[k].enabled == True:

						n_connections += 1

				fitness_val += .0001*(n_connections-3)
				total_conn += n_connections


				children[j].fitness = (fitness_val,)
				children[j][0].fitness = fitness_val

				offspring.append(children[j])

		avg_neurons.append(total_hidden/(len(offspring)))
		avg_connections.append(total_conn/len(offspring))

		population[:] = offspring

		# for i in range(len(population)):

		# 	print(str(i),str(population[i][0].fitness))

		if gen%50 == 0:
			print("Generation " + str(i) +":")
			print(tools.selWorst(population, k=1)[0][0])


		print("~~~~~~~~~~~~~")

	notes.append(avg_neurons)
	notes.append(avg_connections)
	return population
	# 	print("Generation " + str(i) + ":")

# Fitness function for GA2
def evalFitnessWeights(ind, ANN):

	new_net = ANN

	individual = ind
	index = 0

	for i in range(len(new_net.connections)):

		if new_net.connections[i].enabled == True:

			new_net.connections[i].weight = individual[index]
			index += 1

	return evalFitness([new_net])

# crossover for GA2
def crossWeights(ind1, ind2):

	offspring1 = creator.Individual()
	offspring2 = creator.Individual()

	for i in range(len(ind1)):
		offspring1.append(ind1[i])

	for i in range(len(ind2)):
		offspring2.append(ind2[i])

	for i in range(len(ind1)):

		x = random.random()

		offspring1[i] = ind1[i] - x * (ind1[i] - ind2[i])
		offspring2[i] = ind2[i] + x * (ind1[i] - ind2[i])

	return offspring1, offspring2

def mutateWeights(ind, indpb):

	individual = ind

	for i in range(len(individual)):

		if random.random() < indpb:

			n = random.uniform(-1,1)
			individual[i] += n

	return individual

def tournamentSelect(pop, tournsize):

	x = random.randint(0,len(pop)-1)
	best = pop[x]

	for i in range(1, tournsize):

		y = random.randint(0,len(pop)-1)
		if pop[y].fitness < best.fitness:

			best = pop[y]

	return best


# sends the mutated topology and optimizes weights
def optimizeWeights(network, notes2):

	optimum = network
	print(optimum)

	trial = []

	count = 0
	for i in range(len(optimum.connections)):

		if optimum.connections[i].enabled == True:

			count += 1

	creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMin)
	toolbox = base.Toolbox()
	toolbox.register("attr_bool", random.uniform, -10, 10)
	toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, n=count)
	toolbox.register("population", tools.initRepeat, list, 
                     toolbox.individual)
	toolbox.register("evaluate", evalFitnessWeights)
	toolbox.register("mate", crossWeights)
	toolbox.register("mutate", mutateWeights, indpb=0.2)
	toolbox.register("select", tournamentSelect, tournsize=5)

	pop = toolbox.population(n=50)

	data = []

	result = weightsGA(pop, toolbox, optimum, data, cxpb=.5, mutpb=.2, ngen=500)

	notes2.append(data)

	best = tools.selWorst(pop, k=1)[0]

	print(best)

	new_ANN = optimum.copyNetwork()

	index = 0

	for i in range(len(new_ANN.connections)):

		if new_ANN.connections[i].enabled == True:

			new_ANN.connections[i].weight = best[index]
			index += 1

	new_ANN.fitness = best.fitness[0]

	print("BEFORE PASSING:")
	toolbox.evaluate(best, new_ANN)
	print(new_ANN)

	print("~~~~~~~~~~~~~~~~~~~~~~~~~")

	return new_ANN
	


# similar to GA1, but uses the different mutation/crossover (only for weights)
def weightsGA(population, toolbox, ANN, data, cxpb, mutpb, ngen):

	initmean = 0
	for i in range(len(population)):

		population[i].fitness = (toolbox.evaluate(population[i], ANN),)
		print(population[i].fitness)
		initmean += population[i].fitness[0]

	initmean = initmean/(len(population))

	data.append(initmean)

	for gen in range(1,ngen+1):

		print("Generation " + str(gen) +":")

		offspring = []

		mean = 0

		for i in range(len(population)/2):

			parent1 = toolbox.select(population)
			parent2 = toolbox.select(population)

			offspring1 = creator.Individual()
			offspring2 = creator.Individual()

			for j in range(len(parent1)):

				offspring1.append(parent1[j])

			for j in range(len(parent2)):

				offspring2.append(parent2[j])

			children = [offspring1, offspring2]

			if random.random()<cxpb:

				children[0],children[1] = toolbox.mate(children[0],children[1])

			for j in range(len(children)):

				if random.random()<mutpb:

					children[j] = toolbox.mutate(children[j])

				children[j].fitness = (toolbox.evaluate(children[j], ANN),)
				print(children[j].fitness)
				offspring.append(children[j])
				mean += children[j].fitness[0]



		mean = mean/len(offspring)

		data.append(mean)

		population[:] = offspring

	return population


main()

