import numpy as np, os
import mod_grumb as mod, sys
from random import randint
from torch.autograd import Variable
import torch
import random


class Tracker(): #Tracker
    def __init__(self, parameters):
        self.foldername = parameters.save_foldername + '/0000_CSV'
        self.fitnesses = []; self.avg_fitness = 0; self.tr_avg_fit = []
        self.hof_fitnesses = []; self.hof_avg_fitness = 0; self.hof_tr_avg_fit = []
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)
        self.file_save = 'PyTorch_Simulator.csv'

    def add_fitness(self, fitness, generation):
        self.fitnesses.append(fitness)
        if len(self.fitnesses) > 100:
            self.fitnesses.pop(0)
        self.avg_fitness = sum(self.fitnesses)/len(self.fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/train_' + self.file_save
            self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
            np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

    def add_hof_fitness(self, hof_fitness, generation):
        self.hof_fitnesses.append(hof_fitness)
        if len(self.hof_fitnesses) > 100:
            self.hof_fitnesses.pop(0)
        self.hof_avg_fitness = sum(self.hof_fitnesses)/len(self.hof_fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/champ_real_' + self.file_save
            self.hof_tr_avg_fit.append(np.array([generation, self.hof_avg_fitness]))
            np.savetxt(filename, np.array(self.hof_tr_avg_fit), fmt='%.3f', delimiter=',')

    def save_csv(self, generation, filename):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

class SSNE_param:
    def __init__(self):
        self.num_input = 1
        self.num_hnodes = 20
        self.num_output = 1

        self.elite_fraction = 0.1
        self.crossover_prob = 0.1
        self.mutation_prob = 0.9
        self.extinction_prob = 0.000 # Probability of extinction event
        self.extinction_magnituide = 0.9  # Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 1000000000
        # self.mut_distribution = 0 #1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

class Parameters:
    def __init__(self):
            #BackProp
            self.bprop_max_gens = 10
            self.bprop_train_examples = 1000
            self.skip_bprop = False
            self.load_seed = False #Loads a seed population from the save_foldername
                                   # IF FALSE: Runs Backpropagation, saves it and uses that
            self.is_random = False

            #SSNE stuff
            self.population_size = 1000
            self.ssne_param = SSNE_param()
            self.total_gens = 100
            #Determine the nerual archiecture
            self.arch_type = 2 #1 FEEDFORWARD
                               #2 GRU-MB
                               #3 LSTM

            #Task Params
            self.depth_train = 3
            self.depth_test = self.depth_train
            self.num_train_examples = 10
            self.num_test_examples = 100
            self.corridors = [10, 20]

            self.is_dynamic = False  # Makes the task depth dynamic
            self.dynamic_limit = 50


            #Network params
            self.output_activation = 'tanh'
            if self.arch_type == 1: self.arch_type = 'FF'
            elif self.arch_type ==2: self.arch_type = 'GRUMB'
            elif self.arch_type == 3: self.arch_type = 'LSTM'
            else: sys.exit('Invalid choice of neural architecture')

            #Reward scheme
            #1 Test Reward
            #2 Absolute value forces 0 on noise
            #3 GECCO style reward
            self.reward_scheme = 3

            self.save_foldername = 'Seq_Classifier/'

class Task_Seq_Classifier: #Bindary Sequence Classifier
    def __init__(self, parameters):
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param
        self.num_input = self.ssne_param.num_input; self.num_hidden = self.ssne_param.num_hnodes; self.num_output = self.ssne_param.num_output

        self.ssne = mod.Fast_SSNE(parameters) #nitialize SSNE engine

        # Simulator save folder for checkpoints
        self.save_foldername = self.parameters.save_foldername
        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

        #####CREATE POPULATION
        self.pop = []
        for i in range(self.parameters.population_size):
            if self.parameters.arch_type == 'GRUMB':
                self.pop.append(mod.PT_GRUMB(self.num_input, self.num_hidden, self.num_output, output_activation=self.parameters.output_activation))
            elif self.parameters.arch_type == 'LSTM':
                self.pop.append(mod.PT_LSTM(self.num_input, self.num_hidden, self.num_output, output_activation=self.parameters.output_activation))
            elif self.parameters.arch_type == 'FF':
                self.pop.append(mod.PT_FF(self.num_input, self.num_hidden, self.num_output, output_activation=self.parameters.output_activation))

        ###Init population
        if self.parameters.load_seed: #Load seed population
            self.load('bprop_bsc')
        else: #Run Backprop
            self.run_bprop(self.pop[0])
            self.pop[0].to_fast_net()

        #Turn off grad for evolution
        for net in self.pop:
            net.turn_grad_off()



    def save(self, individual, filename ):
        torch.save(individual, filename)
        #return individual.saver.save(individual.sess, self.save_foldername + filename)

    def load(self, filename):
        return torch.load(self.save_foldername + filename)

    def predict(self, individual, input): #Runs the individual net and computes and output by feedforwarding
        return individual.predict(input)

    def run_bprop(self, model):
        if self.parameters.skip_bprop: return
        criterion = torch.nn.L1Loss()
        #criterion = torch.nn.MSELoss()
        #criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        train_x, train_y = self.test_sequence_data(self.parameters.bprop_train_examples, self.parameters.depth_train)
        model.cuda()
        for epoch in range(1, self.parameters.bprop_max_gens):
            epoch_loss = 0.0

            for example_x, example_y in zip(train_x, train_y):  # For each examples
                relevant_out = []
                model.reset() #Reset memory and recurrent out for the model
                for item in example_x: #For all temporal items in input
                    net_out = model.forward([item])
                    if item != 0: #Positions of signal introduction
                        relevant_out.append(net_out)

                # Compare with target and compute loss
                for net_pred, target in zip(relevant_out, example_y):
                    target_T = Variable(torch.Tensor([target]).cuda())
                    target_T = target_T.unsqueeze(0)
                    loss = criterion(net_pred, target_T)
                    #loss = (net_pred - target_T) * 1000 + 100
                    loss.backward(retain_variables=True)
                    epoch_loss += loss.cpu().data.numpy()[0]
                optimizer.step() #Perform the gradient updates to weights for the entire set of collected gradients
                optimizer.zero_grad()

            print 'Epoch: ', epoch, ' Loss: ', epoch_loss / len(train_x)



        #self.save(model, self.save_foldername + 'bprop_simulator') #Save individual

    def compute_fitness(self, individual, data_x, data_y, reward_scheme):
        reward = 0.0

        for example_x, example_y in zip(data_x, data_y):  # For each examples
            out = []
            individual.fast_net.reset()  # Reset memory and recurrent out for the model
            for item in example_x:  # For all temporal items in input

                if self.parameters.is_random: #Random guess
                    if random.random() < 0.5: out.append(1)
                    else: out.append(-1)
                else:
                    net_out = individual.fast_net.forward(item)
                    out.append(net_out[0][0])

            reward += self.get_reward(example_y, out, reward_scheme)


        return reward/len(data_x)


    def get_reward(self, target, prediction, reward_scheme):


        if reward_scheme == 1:  #Test
            reward = 1.0
            for y, pred in zip(target, prediction):
                if y != 0:
                    point_reward = y * pred
                    if point_reward < 0 or pred == 0:
                        reward = 0.0
                        break

        if reward_scheme == 2:  #Coarse
            error = 0.0
            for i, j in zip(target, prediction):
                error += abs(i - j)
            reward = - error/len(target)

        if reward_scheme == 3:  #GECCO
            real_class = 0.0; reward = 0.0
            for y, pred in zip(target, prediction):
                if y != 0: real_class = y

                point_reward = real_class * pred
                if point_reward > 1: point_reward = 1
                if point_reward < -1: point_reward = -1
                reward += point_reward
                if y!= 0: reward += point_reward
            #reward /= len(target)

        return reward


    def evolve(self, gen):

        #Fitness evaluation list for the generation
        fitness_evals = [[] for x in xrange(self.parameters.population_size)]

        #Get task training examples for the epoch
        train_x, train_y = self.test_sequence_data(self.parameters.num_train_examples, self.parameters.depth_train)

        #Test all individuals and assign fitness
        for index, individual in enumerate(self.pop): #Test all genomes/individuals
            fitness = self.compute_fitness(individual, train_x, train_y, reward_scheme=self.parameters.reward_scheme)
            fitness_evals[index] = fitness
        gen_best_fitness = max(fitness_evals)

        #Champion Individual
        champion_index = fitness_evals.index(max(fitness_evals))
        test_x, test_y = self.test_sequence_data(self.parameters.num_test_examples, self.parameters.depth_test)
        champ_real_score = self.compute_fitness(self.pop[champion_index], test_x, test_y, reward_scheme=1) #Reward Scheme just test

        #Save population and HOF
        if gen % 100 == 0:
            for index, individual in enumerate(self.pop): #Save population
                self.save(individual, self.save_foldername + 'Simulator_' + str(index))
            self.save(self.pop[champion_index], self.save_foldername + 'Champion_Simulator') #Save champion
            np.savetxt(self.save_foldername + '/gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')

        #SSNE Epoch: Selection and Mutation/Crossover step
        self.ssne.epoch(self.pop, fitness_evals)

        return gen_best_fitness, champ_real_score

    def test_sequence_data(self, num_examples, depth):

        train_x = []; train_y = []
        for example in range(num_examples):
            x = []; y = []
            for i in range(depth):
                #Encode the signal (1 or -1s)
                if random.random() < 0.5: x.append(-1)
                else: x.append(1)
                if sum(x) >= 0: y.append(1)
                else: y.append(-1)
                if i == depth - 1: continue

                #Encdoe the noise (0's)
                num_noise = randint(self.parameters.corridors[0], self.parameters.corridors[1])
                for i in range(num_noise):
                    x.append(0); y.append(0)
            train_x.append(x); train_y.append(y)
        return train_x, train_y





if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    tracker = Tracker(parameters)  # Initiate tracker
    print 'Running Simulator Training ', parameters.arch_type

    sim_task = Task_Seq_Classifier(parameters)
    for gen in range(1, parameters.total_gens):
        gen_best_fitness, champ_real_score = sim_task.evolve(gen)
        print 'Generation:', gen, ' Best_Epoch:', "%0.2f" % gen_best_fitness, ' Champ Real:', "%0.2f" % champ_real_score, '  Cumul_Real_Score:', "%0.2f" % tracker.hof_avg_fitness
        tracker.add_fitness(gen_best_fitness, gen)  # Add average global performance to tracker
        tracker.add_hof_fitness(champ_real_score, gen)  # Add best global performance to tracker














