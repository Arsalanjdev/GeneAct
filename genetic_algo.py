import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Callable, Optional

import torch

from genetic_utils import Operation, OPERATIONS

MAX_DEPTH = 6  # Max depth of an expression tree
MIN_DEPTH = 3  # Minimum depth of an expression tree


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

    def _traverse(self, node: "OperationGene", func: Callable) -> None:
        """Traverses the expression tree and apply a function on it"""
        if node is None:
            return
        func(node)
        if node.left:
            self._traverse(node.left, func)
        if node.right:
            self._traverse(node.right, func)

    def __str__(self):
        parts = []
        self._traverse(node=self, func=lambda node: parts.append(str(node.operation)))
        return " ".join(parts)


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
        self.root = root or random_gene()
        self.fitness_fn = fitness_fn
        self.fitness: Optional[float] = 0

    def _traverse(self, node: OperationGene, func: Callable) -> None:
        """Traverses the expression tree and apply a function on it"""
        if node is None:
            return
        func(node)
        if node.left:
            self._traverse(node.left, func)
        if node.right:
            self._traverse(node.right, func)

    def flatten(self) -> List["OperationGene"]:
        """
        Flattens the whole operation tree into a list.
        :return: The list of flattened operation trees.
        """
        flattened = []
        self._traverse(self.root, lambda node: flattened.append(node))
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

    def add_gene(self, gene: OperationGene) -> None:
        """Adds a gene to the chromosome"""

    def to_activation_function(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Convert the expression tree of the chromosome back to an activation function"""
        root = self.root.clone()

        def activation(x: torch.Tensor) -> torch.Tensor:
            return root.evaluate(x)

        return activation

    def __str__(self) -> str:
        # parts = []
        # self._traverse(self.root, lambda node: parts.append(str(node)))
        # return " ".join(parts)
        return str(self.root)

    def __repr__(self):
        return str(self)


class GeneticAlgorithm:

    def __init__(
            self,
            population_size: int,
            model_factory: Callable,
            fitness_fn=Callable,
            max_generations: int = 20,
            max_depth: int = MAX_DEPTH,
            mutation_rate: float = 0.2,
            crossover_rate: float = 0.8,
    ):
        self.population_size = population_size
        self.fitness_fn = fitness_fn
        self.max_generations = max_generations
        self.max_depth = max_depth
        self.mutation_rate = mutation_rate

        self.population: List[Chromosome] = [
            Chromosome() for _ in range(population_size)
        ]
        self.crossover_rate = crossover_rate
        self.generation = 0
        self.model = model_factory()

    def evaluate_population(self) -> None:
        for chromosome in self.population:
            chromosome.compute_fitness()

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

    def _tournament_selection(self, tournament_size=3) -> Chromosome:
        """Choosing a chromosome based on fitness"""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def mate(self, selection_method: Callable[[Optional[int]], Chromosome]):
        """Breeding new chromosomes based on the selection method provided"""
        if self.generation >= self.max_generations:
            return max(self.population, key=x.fitness)

        new_population: List[Chromosome] = []

        # Elitism strategy
        new_population.append(max(self.population, key=lambda x: x.fitness))

        while len(new_population) < len(self.population):
            p1 = selection_method()
            p2 = selection_method()

            if random.random() > self.crossover_rate:
                new_population.append(random.choice([p1, p2]))
            else:
                child = self.crossover(p1, p2)
                new_population.append(child)

        self.population = new_population
        self.generation += 1

    def mutate(self):
        for candidate in self.population:
            chance = random.random()
            if chance < self.mutation_rate:
                candidate.mutate()

    def __str__(self):
        return str(self.population)

    # def fitness_population(self):
    #     """Computes a simple normalization [0-10] score of each individual chromosome based on the score of max and min.
    #     Might change it later."""
    #     max_chromosome = max(self.population, key=lambda x: x.fitness)
    #     min_chromosome = min(self.population, key=lambda x: x.fitness)
    #     for chromosome in self.population:
    #         score = (chromosome.fitness - min_chromosome) / (
    #             max_chromosome.fitness - min_chromosome
    #         )


def random_gene() -> OperationGene:
    """Randomly constructs and returns a non-ary or unary or binary gene"""
    chance = random.random()
    X_tensor = OPERATIONS[0][-1]  #
    X_tensor = OperationGene(X_tensor)
    non_ary_op = random.choice(OPERATIONS[0])
    non_ary_op = OperationGene(non_ary_op)
    unary_op = random.choice(OPERATIONS[1])
    unary_op = OperationGene(unary_op, X_tensor)
    if chance < 1 / 2:
        return unary_op
    else:
        binary_op = random.choice(OPERATIONS[2])
        chance = random.random()  # XXX
        if chance < 1 / 2:
            return OperationGene(binary_op, X_tensor, unary_op)
        else:
            return OperationGene(binary_op, X_tensor, non_ary_op)

# def random_chromosome(min_depth=MIN_DEPTH, max_depth=MAX_DEPTH) -> Chromosome:
#     """Constrcuts and returns a randomly generated chromosome with restriction of depth in mind"""
#     return Chromosome(random_gene(), min_depth, max_depth)

#
# ga = GeneticAlgorithm(population_size=50)
#
# print(str(ga))
# for i in range(100):
#     print(Chromosome())

# def random_gene(min_depth: int = 2, max_depth: int = 4) -> OperationGene:
#     """Create a random expression tree with controlled depth."""
#     # depth = random.randint(min_depth, max_depth)
#     # return _build_tree(depth)
#     pass
