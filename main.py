"""

Author: Frederic Mrozinski
Date: August 4th, 2024

This Python script benchmarks neural weight pruning, comparing different
pruning criteria. It serves research purposes and may be used and extended,
as desired. For questions and issue reports, please contact fm.public@tuta.com.

================================================================================
The MIT License

Copyright (c) 2024 Frederic Mrozinski

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
================================================================================

"""




from typing import Callable, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


class PruningObjective:
    def __init__(
            self,
            prunable_weights: List[Tuple[nn.Module, str, str]],
            pruning_ratio: float,
            pruning_steps: int,
            saliency_criterion: str,
            convergence_patience: int):

        self.prunable_weights = prunable_weights
        self.pruning_ratio = pruning_ratio
        self.pruning_steps = pruning_steps
        self.saliency_criterion = saliency_criterion
        self.convergence_patience = convergence_patience


class TrainingObjective:
    # TODO doc
    def __init__(
            self,
            model: nn.Module,
            train_dataset: Dataset,
            val_dataset: Dataset,
            test_dataset: Dataset,
            loss: Callable[[torch.Tensor, torch.Tensor], float]):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.loss = loss


class TrainingArguments:
    # TODO doc
    def __init__(
            self,
            batch_size: int,
            learning_rate: float,
            eval_steps: int,
            device: Optional[str] = None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.eval_steps = eval_steps
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_pruning_loop(
        pruning_objective: PruningObjective,
        training_objective: TrainingObjective,
        training_arguments: TrainingArguments) -> None:
    """
    This function runs the pruning loop, i.e. it prunes a small set of weights,
    then fine-tunes the network until convergence, and repeats until a given
    level of sparsity is reached.

    # TODO Add unlisted parameters here
    :arg: pruning_ratio: float
    The pruning ratio to be reached (0.9 means 90% of weights to be pruned)
    :arg: pruning_steps: int
    The number of pruning steps to perform until pruning_ratio is reached.

    :return: TODO add return description
    """

    writer = SummaryWriter()
    global_training_step = 0

    for prune_step in range(pruning_objective.pruning_steps):

        # 1. Compute current weight saliency
        print("computing saliencies...")
        saliencies = compute_weight_saliencies(
            pruning_objective, training_objective, training_arguments)

        # 2. Prune the next set of weights
        print("pruning...")
        prune_next_weights(
            training_objective.model, pruning_objective, saliencies)
        writer.add_scalar(
            "pruning / global step", 1, global_training_step)

        # 3. Fine-tune network until convergence
        print("training...")

        train_dataloader = DataLoader(
            training_objective.train_dataset,
            batch_size=training_arguments.batch_size,
            shuffle=True
        )
        val_dataloader = DataLoader(
            training_objective.val_dataset,
            batch_size=training_arguments.batch_size,
            shuffle=False
        )

        model = training_objective.model
        optimizer = torch.optim.Adam(
            model.parameters(), lr=training_arguments.learning_rate)
        first_batch = True
        local_training_step = 0

        def eval_model() -> float:
            print("evaluating...")
            model.eval()
            with torch.no_grad():
                v_loss = 0
                for v_inpt, v_target in val_dataloader:
                    v_inpt = v_inpt.to(training_arguments.device)
                    v_target = v_target.to(training_arguments.device)
                    v_pred = model(v_inpt)
                    v_loss += training_objective.loss(v_pred, v_target)
                v_loss /= len(val_dataloader.dataset)
                writer.add_scalar(
                    "val-loss / global step", v_loss, global_training_step)
            print("end evaluating...")
            return v_loss

        best_val_score = eval_model()
        patience_count = 0
        for inpt, target in train_dataloader:
            if first_batch:
                first_batch = False
            else:
                writer.add_scalar(
                    "pruning / global step", 0, global_training_step)

            optimizer.zero_grad()

            inpt = inpt.to(training_arguments.device)
            target = target.to(training_arguments.device)

            model.train()
            pred = model(inpt)
            loss = training_objective.loss(pred, target)
            writer.add_scalar(
                "train-loss / global step", loss, global_training_step)
            loss.backward()
            optimizer.step()

            global_training_step += 1
            local_training_step += 1
            if local_training_step % training_arguments.eval_steps == 0:
                val_score = eval_model()
                if val_score >= best_val_score:
                    patience_count += 1
                    if patience_count >= pruning_objective.convergence_patience:
                        break
                else:
                    patience_count = 0
                    best_val_score = val_score


def compute_weight_saliencies(
        pruning_objective: PruningObjective,
        training_objective: TrainingObjective,
        training_arguments: TrainingArguments) -> dict[str, torch.Tensor]:
    # TODO doc

    saliencies = {}

    match pruning_objective.saliency_criterion:
        case 'magnitude':
            for pw_module, pw_name, pw_fullname \
                    in pruning_objective.prunable_weights:
                if hasattr(pw_module, pw_name):
                    p = getattr(pw_module, pw_name)
                    saliencies[pw_fullname] = p.abs().clone().detach()
                else:
                    p = getattr(pw_module, pw_name + "_orig")
                    saliencies[pw_fullname] = p.abs().clone().detach()
                    mask = getattr(pw_module, pw_name + "_mask")
                    saliencies[pw_fullname] *= mask


        case 'fisher-information':
            dataloader = None  # TODO, set batch size to 1!
            raise NotImplementedError()
            model = training_objective.model

            for _, _, pw_fullname \
                    in pruning_objective.prunable_weights:
                saliencies[pw_fullname] = 0

            # Compute accumulated squared gradients for all weights
            model.train()
            for inpt, target in dataloader:
                model.zero_grad()

                inpt = inpt.to(training_arguments.device)
                target = target.to(training_arguments.device)

                pred = model(inpt)
                loss = training_objective.loss(pred, target)
                loss.backward()

                # Compute Fisher-information for all prunable weights

                if hasattr(pw_module, pw_name):
                    p = getattr(pw_module, pw_name)
                    saliencies[pw_fullname] += p.grad.square()
                else:
                    p = getattr(pw_module, pw_name + "_orig")
                    saliencies[pw_fullname] += p.grad.square()

            # Scale Fisher-information by squared weight values
            if hasattr(pw_module, pw_name):
                p = getattr(pw_module, pw_name)
                saliencies[pw_fullname] *= p.square()
            else:
                p = getattr(pw_module, pw_name + "_orig")
                saliencies[pw_fullname] *= p.square()
                mask = getattr(pw_module, pw_name + "_mask")
                saliencies[pw_fullname] *= mask

        case _:
            raise ValueError(f"Saliency criterion "
                             f"'{pruning_objective.saliency_criterion}' "
                             f"is not supported.'")

    return saliencies


def prune_next_weights(
        model: nn.Module,
        pruning_objective: PruningObjective,
        saliencies: dict[str, torch.Tensor]) -> None:
    # TODO doc

    final_unpruned_ratio = 1 - pruning_objective.pruning_ratio
    step_unpruned_ratio = (final_unpruned_ratio **
                           (1 / pruning_objective.pruning_steps))
    step_pruning_ratio = 1 - step_unpruned_ratio

    for module, name, fullname in pruning_objective.prunable_weights:
        prune.l1_unstructured(
            module, name, step_pruning_ratio, saliencies[fullname])


alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010])
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    normalize,
])
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform,
)
train_dataset = Subset(train_dataset, range(1000))
val_test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False,
    download=True, transform=transform,
)
val_test_dataset = Subset(val_test_dataset, range(1000))
val_dataset, test_dataset = torch.utils.data.random_split(val_test_dataset, [600, 400])

_prunable_weights = [
    (alexnet.classifier[1], 'weight', 'classifier.1.weight'),
    (alexnet.classifier[4], 'weight', 'classifier.4.weight'),
    (alexnet.classifier[6], 'weight', 'classifier.6.weight'),
]

_training_objective = TrainingObjective(alexnet, train_dataset, val_dataset, test_dataset, nn.CrossEntropyLoss())
_training_arguments = TrainingArguments(16, 5e-4, 10)
_pruning_objective = PruningObjective(_prunable_weights, 0.5, 30, 'magnitude', 0)

run_pruning_loop(_pruning_objective, _training_objective, _training_arguments)