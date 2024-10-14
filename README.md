# adversarial_mnist
Adversarial examples for image recognition

## Model Info

This example use a [multi-layer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) for image recognition.
THe default architecture uses one hidden layer with 32 hidden nodes and sigmoid activations, and a 10-dimensional softmax output layer.

The model is trained using Adam for 32 epochs.
Settings such as learning rates, number of eopchs, etc... can be edited in the source code.
Training the model takes a couple minutes on a CPU machine.

## Serving the Genie app

To serve tha app in dev mode:
```bash
julia --project=. -i -e 'using GenieFramework; Genie.loadapp(); Genie.up()'
```

To serve the app in production mode:
```bash
GENIE_ENV=prod
julia --project -e "using GenieFramework; Genie.loadapp(); up(async=false);"
```
