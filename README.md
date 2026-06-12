# Overview

This project exlores security vulnerabilites in d=federated learning systems by 
simulating malicious clients and evaluating their impact on a global model
trained using FedAvg aggregation.

(experimaents were conducted using ResNet18 and CIFAR-10 dataset)

# Implemented attacks

## Label Poisoning:

A malicious client modifies training labels before local training.

### Objective:

corrupt model behaviour for a targeted class.

increase targeted misclassification rates.

### Findings:

Poisoning influence increased gradually across communication rounds.

Global model became more llikely to classify class 0 samples as class 1.

## Weight Manipulation attacks:

A malicious client amplifies model parameters before aggregation.

### e.g:

weights *= k

### tested values:

1.5x - caused moderate model degradation.

2x - caused severe model degradation.

5x - resulted in catastrophic loss growth and model instability.

## Sign-Flipping attacks:

A malicious client reverses model parameters before aggregation.

### e.g:

weights *= -1

### objective:
oppose honest client updates.

prevernt global convergence.

### Findings:
GLobal accuracy remained near random-guess performance (10%).

Global loss remained near initial baselines values.

The attack appeared to oppose learning rather than destabilized the model.
