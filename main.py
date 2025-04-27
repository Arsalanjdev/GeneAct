# from simple_term_menu import TerminalMenu
import questionary

import models
from genetic_algo import GeneticAlgorithm
from mnist.base_msnit import MLP
from models import *

ga_msnit = GeneticAlgorithm(
    population_size=5, model=models.MLPModel, fitness_fn=None, max_generations=10
)
print(ga_msnit.run())
