import traceback
from collections import defaultdict, deque
from copy import deepcopy
from functools import partial
from math import prod
from os.path import exists, join
from pickle import dump, load
from random import choice
from time import time
from itertools import cycle

import psutil
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from einspace.search_spaces import EinSpace
from einspace.utils import (
    ArchitectureCompilationError,
    SearchSpaceSamplingError,
    get_average_branching_factor,
    get_max_depth,
    get_size,
    millify,
    recurse_count_nodes,
)


class Individual(object):
    """A class representing a model containing an architecture, its modules and its accuracy."""

    def __init__(
        self,
        id,
        parent_id,
        arch=None,
        modules=None,
        accuracy=None,
        age=0,
        hpo_dict=None,
    ):
        self.id = id
        self.parent_id = parent_id
        self.arch = arch
        # self.modules = modules
        self.accuracy = accuracy
        self.age = age
        self.hpo_dict = hpo_dict

        self.alive = True

        self.feature_shape = arch["output_shape"]
        self.num_parameters = sum([p.numel() for p in modules.parameters()])
        self.num_terminals = get_size(arch, "terminal")
        self.num_nonterminals = get_size(arch, "nonterminal")
        self.average_branching_factor = get_average_branching_factor(arch)
        self.max_depth = get_max_depth(arch)

        self.nodes = {
            node.__name__: 0
            for node in (
                EinSpace.modules
                + EinSpace.branching_fns
                + EinSpace.aggregation_fns
                + EinSpace.prerouting_fns
                + EinSpace.postrouting_fns
                + EinSpace.computation_fns
            )
        }

    def get_descriptor(self):
        # return a simple vector representation of the individual including all numerical features
        return [
            self.feature_shape[1],
            self.num_parameters,
            self.num_terminals,
            self.num_nonterminals,
            self.average_branching_factor,
            self.max_depth,
            *recurse_count_nodes(self.arch, self.nodes).values(),
        ]

    def get_named_descriptor(self):
        # return a simple vector representation of the individual including all numerical features
        named_descriptor = {
            "feature_shape": self.feature_shape[1],
            "num_parameters": self.num_parameters,
            "num_terminals": self.num_terminals,
            "num_nonterminals": self.num_nonterminals,
            "average_branching_factor": self.average_branching_factor,
            "max_depth": self.max_depth,
        }
        named_descriptor.update(recurse_count_nodes(self.arch, self.nodes))
        return named_descriptor

    def __repr__(self):
        """Prints a readable version of this bitstring."""
        return f"Individual(accuracy={self.accuracy}, age={self.age}, feature_shape={self.feature_shape}, num_parameters={millify(self.num_parameters)}, num_terminals={self.num_terminals}, num_nonterminals={self.num_nonterminals}, average_branching_factor={self.average_branching_factor}, hpo_dict={self.hpo_dict})"


class Population(deque):
    """A class representing a population of models."""

    def __init__(self, individuals):
        self.individuals = individuals

    def __repr__(self):
        """Prints a readable version of this bitstring."""
        return f"Population(individuals={self.individuals})"

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]

    def __setitem__(self, idx, value):
        self.individuals[idx] = value

    def append(self, individual):
        self.individuals.append(individual)

    def popleft(self):
        individual = self.individuals.pop(0)
        individual.alive = False

    def max(self, key):
        return max(self.individuals, key=key)

    def sample(self, k):
        return choice(self.individuals, k=k)

    def extend(self, individuals):
        self.individuals.extend(individuals)

    def sort(self, key):
        self.individuals.sort(key=key)

    def __iter__(self):
        return iter(self.individuals)

    def __next__(self):
        return next(self.individuals)

    def __contains__(self, item):
        return item in self.individuals

    def index(self, item):
        return self.individuals.index(item)

    def remove(self, item):
        self.individuals.remove(item)

    def tournament_selection(self, k, key):
        sample = []
        while len(sample) < k:
            candidate = choice(list(self.individuals))
            sample.append(candidate)
        return max(sample, key=key)

    def age(self):
        for individual in self.individuals:
            individual.age += 1

    def tolist(self):
        return self.individuals


class RandomSearch:
    """Random search strategy"""

    def __init__(
        self,
        search_space,
        compiler,
        evaluation_fn,
        num_samples,
        save_name,
        continue_search=False,
    ):
        self.search_space = search_space
        self.compiler = compiler
        self.evaluation_fn = evaluation_fn
        self.num_samples = num_samples
        self.save_name = save_name

        if continue_search:
            # check if the file exists
            if exists(join("results", self.save_name + ".pkl")):
                print(join("results", self.save_name + ".pkl"))
                self.history = Population(
                    load(open(join("results", self.save_name + ".pkl"), "rb"))
                )
                print(
                    f"Continuing search from previous results at {self.save_name}.",
                    flush=True,
                )
            else:
                self.history = Population([])
                print(
                    f"No previous results found at {self.save_name}. Starting new search."
                )
        else:
            self.history = Population([])

    def create_and_evaluate_individual(
        self, architecture, modules, id, parent_id
    ):
        best_model = self.evaluation_fn(architecture, modules)
        individual = Individual(id, parent_id, architecture, modules)
        individual.accuracy = best_model["val_score"]
        individual.duration = best_model["duration"]
        if "lr" in best_model:
            individual.hpo_dict = {
                key: best_model[key]
                for key in ["lr", "momentum", "weight_decay", "epoch"]
            }
        else:
            individual.hpo_dict = {}
        return individual

    def new_individual(self, iteration):
        architecture = self.search_space.sample()
        modules = self.compiler.compile(architecture)
        individual = self.create_and_evaluate_individual(
            architecture, modules, iteration, None
        )
        return individual

    def search(self):
        """Perform random search"""
        print("####################################")
        print("Started searching with Random Search")
        print("####################################")
        while len(self.history) < self.num_samples:
            print(f"Training architecture {len(self.history) + 1}", flush=True)
            try:
                individual = self.new_individual(len(self.history))
                self.history.append(individual)
                print(individual, flush=True)
            except Exception as e:
                print(e)
            # Save history
            dump(
                self.history.tolist(),
                open(join("results", self.save_name + ".pkl"), "wb"),
            )
        return self.history


class RegularisedEvolution:
    def __init__(
        self,
        search_space,
        compiler,
        evaluation_fn,
        num_samples,
        init_pop_size,
        sample_size,
        save_name,
        continue_search=False,
        architecture_seed=[],  # list of architectures to start with
        update_population=True, # if True, the population will be updated
    ):
        """Algorithm for regularized evolution (i.e. aging evolution).

        Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
        Classifier Architecture Search".

        Args:
        num_samples: the number of cycles the algorithm should run for.
        population_size: the number of individuals to keep in the population.
        sample_size: the number of individuals that should participate in each
            tournament.

        Returns:
        history: a list of `Model` instances, representing all the models computed
            during the evolution experiment.
        """
        self.search_space = search_space
        self.compiler = compiler
        self.evaluation_fn = evaluation_fn
        self.num_samples = num_samples
        self.init_pop_size = init_pop_size
        self.sample_size = sample_size
        self.save_name = save_name
        self.architecture_seed = architecture_seed
        self.update_population = update_population

        if continue_search:
            if exists(join("results", self.save_name + ".pkl")):
                print(join("results", self.save_name + ".pkl"))
                self.history = Population(
                    load(open(join("results", self.save_name + ".pkl"), "rb"))
                )
                print(
                    f"Continuing search from previous results at {self.save_name}.",
                    flush=True,
                )
                # try:
                #     self.population = Population(
                #         [individual for individual in self.history if individual.alive]
                #     )
                #     print(f"Loaded {len(self.population)} alive individuals from previous search.", flush=True)
                # except:
                # print("No alive individuals in the population. Loading legacy way.", flush=True)
                if self.update_population:
                    self.population = Population(
                        self.history[-self.init_pop_size:]
                    )
                else:
                    self.population = Population(
                        self.history[:self.init_pop_size]
                    )
                print(
                    f"Loaded {len(self.population)} alive individuals from previous search.",
                    flush=True,
                )
            else:
                self.history = Population([])
                self.population = Population([])
                print(
                    f"No previous results found at {self.save_name}. Starting new search.",
                    flush=True,
                )
        else:
            self.population = Population([])
            self.history = Population([])

    def create_and_evaluate_individual(
        self, architecture, modules, id, parent_id
    ):
        best_model = self.evaluation_fn(architecture, modules)
        individual = Individual(id, parent_id, architecture, modules)
        individual.accuracy = best_model["val_score"]
        individual.duration = best_model["duration"]
        individual.hpo_dict = {
            key: best_model[key]
            for key in ["lr", "momentum", "weight_decay", "epoch"]
        }
        return individual

    def new_individual(self, iteration, mode="sample"):
        id = iteration
        if mode == "seed":
            if len(self.architecture_seed) > 0:
                print(
                    f"Creating individual {len(self.history) + 1} from architecture seed",
                    flush=True,
                )
                pre_architecture = self.architecture_seed.pop(0)
                try:
                    architecture = self.search_space.recurse_state(
                        pre_architecture,
                        input_shape=self.search_space.input_shape,
                        input_mode=self.search_space.input_mode,
                    )
                    parent_id = None
                except Exception as e:
                    print(
                        f"Error while creating individual from seed: {e}",
                        flush=True,
                    )
                    self.architecture_seed.append(pre_architecture)
            else:
                print(
                    f"No more architectures in seed. Sampling individual {len(self.history) + 1}",
                    flush=True,
                )
                mode == "sample"
        if mode == "sample":
            architecture = self.search_space.sample()
            parent_id = None
        elif mode == "mutate":
            parent = self.population.tournament_selection(
                k=self.sample_size, key=lambda i: i.accuracy
            )
            architecture = self.search_space.mutate(parent.arch)
            try:
                parent_id = parent.id
            except:
                # legacy version
                parent_id = None
        modules = self.compiler.compile(architecture)
        individual = self.create_and_evaluate_individual(
            architecture, modules, id, parent_id
        )
        return individual

    def search(self):
        """Perform regularised evolution"""
        print("############################################")
        if self.update_population:
            print("Started searching with Regularised Evolution")
        else:
            print("Started searching with Random Mutation")
        print("############################################")

        # Initialize the population with random models.
        while len(self.population) < self.init_pop_size:
            print(f"Training architecture {len(self.history) + 1}", flush=True)
            try:
                if len(self.architecture_seed) > 0:
                    population_seed = []
                    while len(self.architecture_seed) > 0:
                        individual = self.new_individual(
                            len(self.history), mode="seed"
                        )
                        print(individual)
                        population_seed.append(individual)
                    k = self.init_pop_size // len(population_seed)
                    for _, individual in zip(range(self.init_pop_size), cycle(population_seed)):
                        print(
                            f"Adding individual {len(self.history)} to population: {individual}"
                        )
                        self.population.append(deepcopy(individual))
                        self.history.append(deepcopy(individual))
                    print("Seed population created.")
                else:
                    individual = self.new_individual(
                        len(self.history), mode="sample"
                    )
                    self.population.append(individual)
                    self.history.append(individual)
                    print(individual)
            except ArchitectureCompilationError as e:
                print(
                    f"Error while creating the population at iteration {len(self.history) + 1}: {e}",
                    flush=True,
                )
            except SearchSpaceSamplingError as e:
                print(
                    f"Error while creating the population at iteration {len(self.history) + 1}: {e}",
                    flush=True,
                )
            except TimeoutError as e:
                print(
                    f"Error while creating the population at iteration {len(self.history) + 1}: {e}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"Error while creating the population at iteration {len(self.history) + 1}: {e}",
                    flush=True,
                )
                print(traceback.format_exc())
            # Save history
            dump(
                self.history.tolist(),
                open(join("results", self.save_name + ".pkl"), "wb"),
            )
            # track memory usage
            memory_usage = psutil.virtual_memory()
            print(f"Memory Usage: {memory_usage.percent}%", flush=True)

        # Carry out evolution in cycles. Each cycle produces a model and removes
        # another.
        while len(self.history) < self.num_samples:
            # Create and evaluate new individuals via mutation.
            print(f"Training architecture {len(self.history) + 1}", flush=True)
            try:
                child = self.new_individual(len(self.history), mode="mutate")
                if self.update_population:
                    self.population.append(child)
                self.history.append(child)
                print(child, flush=True)
            except Exception as e:
                print(e, flush=True)

            # Age each induvidual in the population by 1
            self.population.age()
            if self.update_population:
                # kill oldest individuals
                while len(self.population) > self.init_pop_size:
                    self.population.popleft()
            # Save history
            dump(
                self.history.tolist(),
                open(join("results", self.save_name + ".pkl"), "wb"),
            )
            # track memory usage
            memory_usage = psutil.virtual_memory()
            print(f"Memory Usage: {memory_usage.percent}%", flush=True)

        return self.history


# class AsyncRegularisedEvolution:
#     def __init__(
#         self,
#         search_space,
#         compiler,
#         evaluation_fn,
#         num_samples,
#         init_pop_size,
#         sample_size,
#         save_name,
#         gpus=["0", "1", "2", "3"],
#         pool_size=4,
#         continue_search=False,
#     ):
#         """Algorithm for regularized evolution (i.e. aging evolution).

#         Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
#         Classifier Architecture Search".

#         Args:
#         num_samples: the number of cycles the algorithm should run for.
#         population_size: the number of individuals to keep in the population.
#         sample_size: the number of individuals that should participate in each
#             tournament.

#         Returns:
#         history: a list of `Model` instances, representing all the models computed
#             during the evolution experiment.
#         """
#         self.search_space = search_space
#         self.compiler = compiler
#         self.evaluation_fn = evaluation_fn
#         self.num_samples = num_samples
#         self.init_pop_size = init_pop_size
#         self.sample_size = sample_size
#         self.save_name = save_name
#         # multiprocessing parameters
#         self.gpus = gpus
#         self.pool_size = pool_size
#         mp.set_start_method("spawn", force=True)
#         self.manager = mp.Manager()
#         self.queue = self.manager.Queue()
#         self.pool = mp.Pool(processes=self.pool_size)

#         if continue_search:
#             if join("results", self.save_name + ".pkl"):
#                 self.history = Population(
#                     load(open(join("results", self.save_name + ".pkl"), "rb"))
#                 )
#                 print(
#                     f"Continuing search from previous results at {self.save_name}."
#                 )
#                 try:
#                     self.population = Population(
#                         [individual for individual in self.history if individual.alive]
#                     )
#                 except:
#                     print("No alive individuals in the population. Loading legacy way.")
#                     self.population = Population(self.history[-self.init_pop_size:])
#             else:
#                 self.history = Population([])
#                 self.population = Population([])
#                 print(
#                     f"No previous results found at {self.save_name}. Starting new search."
#                 )
#         else:
#             self.population = Population([])
#             self.history = Population([])

#     def create_and_evaluate_individual(self, architecture, modules, id, parent_id):
#         best_model = self.evaluation_fn(architecture, modules)
#         individual = Individual(id, parent_id, architecture, modules)
#         individual.accuracy = best_model["val_score"]
#         individual.hpo_dict = {
#             key: best_model[key]
#             for key in ["lr", "momentum", "weight_decay", "epoch"]
#         }
#         return individual

#     def new_individual(self, iteration, mode="sample"):
#         id = iteration
#         if mode == "sample":
#             architecture = self.search_space.sample()
#             parent_id = None
#         elif mode == "mutate":
#             parent = self.population.tournament_selection(
#                 k=self.sample_size, key=lambda i: i.accuracy
#             )
#             architecture = self.search_space.mutate(parent.arch)
#             try:
#                 parent_id = parent.id
#             except:
#                 # legacy version
#                 parent_id = None
#         modules = self.compiler.compile(architecture)
#         individual = self.create_and_evaluate_individual(architecture, modules, id, parent_id)
#         return individual

#     def run_worker(self, fn, queue, job_index):
#         try:
#             start_time = time()
#             with open(
#                 join("logs", f"{str(job_index)}.txt"), "w", buffering=1
#             ) as logger:
#                 logger.write(f"Process {job_index} started\n")
#                 individual = fn()
#                 queue.put(
#                     {
#                         "start_time": start_time,
#                         "individual": individual,
#                     }
#                 )
#                 logger.write(f"Process {job_index} ended\n")
#         except Exception as e:
#             print(
#                 f"Process {job_index + 1}: crash with the following Exception:\n"
#             )
#             print(e)
#             traceback.print_exc()
#             raise e

#     def handle_results(self, x):
#         results = self.queue.get()
#         duration = time() - results["start_time"]
#         results["duration"] = duration
#         individual = results["individual"]
#         self.population.append(individual)
#         self.history.append(individual)
#         dump(
#             self.history.tolist(),
#             open(join("results", self.save_name + ".pkl"), "wb"),
#         )
#         print(f"Process {results['job_index'] + 1} completed.", flush=True)
#         print(individual)
#         # track memory usage
#         memory_usage = psutil.virtual_memory()
#         print(f"Memory Usage: {memory_usage.percent}%")

#     def kill_pool(self, err_msg):
#         print(err_msg, flush=True)
#         print("Terminating all processes.", flush=True)
#         self.pool.terminate()

#     def search(self):
#         """Perform regularised evolution"""
#         print("############################################")
#         print("Started searching with Regularised Evolution")
#         print("############################################")

#         # Initialize the population with random models.
#         while len(self.population) < self.init_pop_size:
#             for i in range(self.pool_size):
#                 device = f"cuda:{self.gpus[i % len(self.gpus)]}"
#                 self.pool.apply_async(
#                     self.run_worker,
#                     args=(
#                         partial(
#                             self.run_worker,
#                             partial(self.new_individual, iteration=len(self.history), mode="mutate"),
#                         ),
#                         device,
#                         self.queue,
#                         i,
#                     ),
#                     callback=self.handle_results,
#                     error_callback=self.kill_pool,
#                 )
#                 print(f"Process {i + 1} launched on {device}.", flush=True)
#             # synchronise processes
#             self.pool.join()

#         # Carry out evolution in cycles. Each cycle produces a model and removes
#         # another.
#         while len(self.history) < self.num_samples:
#             for i in range(self.pool_size):
#                 device = f"cuda:{self.gpus[i % len(self.gpus)]}"
#                 self.pool.apply_async(
#                     self.run_worker,
#                     args=(
#                         partial(
#                             self.run_worker,
#                             partial(self.new_individual, iteration=len(self.history), mode="mutate"),
#                         ),
#                         device,
#                         self.queue,
#                         i,
#                     ),
#                     callback=self.handle_results,
#                     error_callback=self.kill_pool,
#                 )
#                 print(f"Process {i + 1} launched on {device}.", flush=True)
#             # synchronise processes
#             self.pool.join()
#             # Age each individual in the population by 1
#             self.population.age()
#             # kill oldest individuals
#             while len(self.population) > self.init_pop_size:
#                 self.population.popleft()
#         self.pool.close()
#         self.pool.join()
#         self.manager.shutdown()

#         return self.history