import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Callable, Optional

import torch
from numpy.random import choice

from genetic_utils import Operation, OPERATIONS

MAX_DEPTH = 6  # Max depth of expression tree


@dataclass
class OperationGene:
    operation: Operation
    left: Optional["OperationGene"] = None
    right: Optional["OperationGene"] = None

    def arity(self) -> int:
        """:returns the arity of the root operation."""
        for ar, ops in OPERATIONS.items():
            if self.operation in ops:
                return ar
        raise ValueError(f"Unknown operation {self.operation.name}")

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Recursively evaluates the final value of the operation tree"""
        if self.operation in OPERATIONS[0]:  # 0-ary
            return self.operation.func(x)

        elif self.operation in OPERATIONS[1]:  # 1-ary
            if self.left is None:
                raise ValueError(f"Missing operand")
            return self.operation.func(self.left.evaluate(x))

        elif self.operation in OPERATIONS[2]:  # 2-ary
            if self.right is None or self.left is None:
                raise ValueError(f"Missing operand")
            return self.operation.func(self.left.evaluate(x), self.right.evaluate(x))

        else:
            raise ValueError(f"Unknown operation: {self.operation}")

    def clone(self) -> "OperationGene":
        """
        :return: Returns a deep copy of the operation gene
        """
        return deepcopy(self)


class Chromosome:
    def __init__(
            self,
            root: OperationGene = None,
            fitness_fn: Optional[Callable[["Chromosome"], float]] = None,
            max_depth: int = 6,
    ) -> None:
        """
        If no root is provided, a random tree up to maximum depth is constructed.
        :param root:
        :param fitness_fn:
        :param max_depth:
        """
        self.max_depth = max_depth
        self.root = root or random_gene(max_depth)
        self.fitness_fn = fitness_fn
        self.fitness: Optional[float] = 0

    def flatten(self) -> List["OperationGene"]:
        """
        Flattens the whole operation tree into a list.
        :return: The list of flattened operation trees.
        """
        flattened = []

        def traverse(node):
            if node is None:
                return
            flattened.append(node)
            traverse(node.left)
            traverse(node.right)

        traverse(self.root)
        return flattened

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the chromosome's activation function on tensor X"""
        return self.root.evaluate(x)

    def compute_fitness(self) -> float:
        """Compute and store the fitness of the chromosome with the help of provided fitness function"""
        if self.fitness is None:
            raise ValueError("No fitness function is provided.")
        self.fitness = self.fitness_fn(self)
        return self.fitness

    def mutate(self):
        """Mutating a randomly chosen n-ary operation with another n-ary operation"""
        genes: List[OperationGene] = self.flatten()
        chosen_gene: OperationGene = random.choice(genes)
        current_operation = chosen_gene.operation
        arity = chosen_gene.arity()
        candidates = [op for op in OPERATIONS[arity] if op != current_operation]

        if not candidates:
            return

        new_operation = random.choice(candidates)
        chosen_gene.operation = new_operation


class GeneticAlgorithm:

    def __init__(
            self,
            population_size: int,
            fitness_fn=Callable,
            max_generations: int = 20,
            max_depth: int = MAX_DEPTH,
            mutation_rate: float = 0.1,
    ):
        self.population_size = population_size
        self.fitness_fn = fitness_fn
        self.max_generations = max_generations
        self.max_depth = max_depth

    def crossover(self, parent1: "Chromosome", parent2: "Chromosome") -> "Chromosome":
        """Perform subtree crossover between parent1 and parent2,
        ensuring offspring has at most MAX_NODES nodes.
        If the limit would be exceeded, return a mutated copy of parent1 instead."""

        # Clone the parent roots
        tree1 = parent1.root.clone()
        tree2 = parent2.root.clone()

        # Wrap the cloned trees into temporary Chromosome instances to use flatten
        temp1 = Chromosome(
            root=tree1, fitness_fn=parent1.fitness_fn, max_depth=parent1.max_depth
        )
        temp2 = Chromosome(
            root=tree2, fitness_fn=parent2.fitness_fn, max_depth=parent2.max_depth
        )

        # Group nodes by arity
        nodes1 = {}
        for n in temp1.flatten():
            nodes1.setdefault(n.arity(), []).append(n)

        nodes2 = {}
        for n in temp2.flatten():
            nodes2.setdefault(n.arity(), []).append(n)

        # Find common arities
        common = set(nodes1.keys()) & set(nodes2.keys())
        if not common:
            return parent1.mutate()

        a = random.choice(list(common))

        # Choose random nodes of matching arity
        choosen_node1 = random.choice(nodes1[a])
        choosen_node2 = random.choice(nodes2[a])

        # Swap operations and children
        choosen_node1.operation, choosen_node2.operation = (
            choosen_node2.operation,
            choosen_node1.operation,
        )
        choosen_node1.left, choosen_node2.left = choosen_node2.left, choosen_node1.left
        choosen_node1.right, choosen_node2.right = (
            choosen_node2.right,
            choosen_node1.right,
        )

        # After crossover, enforce maximum node count
        new_child = Chromosome(
            root=tree1, fitness_fn=parent1.fitness_fn, max_depth=parent1.max_depth
        )
        if len(new_child.flatten()) > self.max_nodes:
            return parent1.mutate()

        return new_child

    def mate(self, partner: "Chromosome") -> "Chromosome":
        """Selecting parents for crossover and then applying crossover."""
        pass


def __init__(
        self,
        population_size: int,
        fitness_function: Callable,
        min_length: int = 3,
        max_length: int = 6,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.1,
):
    self.population_size = population_size
    self.fitness_function = fitness_function
    self.min_length = min_length
    self.max_length = max_length
    self.crossover_rate = crossover_rate
    self.mutation_rate = mutation_rate

    self.population: List[Chromosome] = []


def fitness_function(chromosome: Chromosome) -> float:
    pass


def random_gene() -> OperationGene:
    """Randomly constrcuts and returns a non-ary or unary or binary gene"""
    chance = random.random()
    X_tensor = OPERATIONS[0][-1]  #
    non_ary_op = random.choice(OPERATIONS[0])
    unary_op = random.choice(OPERATIONS[1])
    if chance < 1 / 3:
        return OperationGene(non_ary_op)
    elif chance < 2 / 3:
        return OperationGene(unary_op, X_tensor)
    else:
        binary_op = random.choice(OPERATIONS[2])
        chance = random.random()  # XXX
        if chance < 1 / 2:
            return OperationGene(binary_op, X_tensor, unary_op)
        else:
            return OperationGene(binary_op, X_tensor, non_ary_op)

# for i in range(100):
#     print(random_gene())

# def random_gene(min_depth: int = 2, max_depth: int = 4) -> OperationGene:
#     """Create a random expression tree with controlled depth."""
#     # depth = random.randint(min_depth, max_depth)
#     # return _build_tree(depth)
#     pass
