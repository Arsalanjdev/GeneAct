from dataclasses import dataclass
from typing import Dict, List, Callable

import torch
import torch.nn.functional as F


@dataclass
class Operation:
    name: str
    role: str
    func: Callable

    def __str__(self):
        return self.name


X_Operation = Operation("X", "X", lambda x: x)
OPERATIONS: Dict[int, List[Operation]] = {
    0: [
           Operation(
               name,
               "terminal",
               lambda x, n=name: (
                   torch.full_like(x, float(n)) if n.replace(".", "", 1).isdigit() else x
               ),
           )
           for name in ["0.5", "1.0", "1.5", "2.0"]
       ]
       + [X_Operation],
    1: [
        Operation("relu", "ACTIVATION", F.relu),
        Operation("sigmoid", "GATING", F.sigmoid),
        Operation("tanh", "GATING", F.tanh),
        Operation("sin", "PERIODIC", torch.sin),
        Operation("cos", "PERIODIC", torch.cos),
        Operation(
            "safe_exp", "TRANSFORM", lambda x: torch.exp(torch.clamp(x, -10, 10))
        ),  # Prevents explosion
        Operation("ln", "TRANSFORM", lambda x: torch.log(x + 1e-8)),
        Operation("sqrt", "TRANSFORM", lambda x: torch.sqrt(torch.abs(x) + 1e-8)),
        Operation("abs", "TRANSFORM", torch.abs),
        Operation("softplus", "ACTIVATION", F.softplus),
        Operation("heavyside", "ACTIVATION", torch.heaviside),
    ],
    2: [
        Operation("add", "COMBINER", lambda x, y: x + y),
        Operation("mul", "GATING", lambda x, y: x * y),
        Operation("sub", "COMBINER", lambda x, y: x - y),
        Operation("div", "COMBINER", lambda x, y: x / (y + 1e-8)),
    ],
}
