using Flux
using MLDatasets
using Statistics

"""
    initialize_mnist_mlp(h)

Initialize a multi-layer perceptron (MLP) for MNIST digit recognition.

The MLP model has input size 784 (28*28), hidden layer sizes defined as `h`, 
    and a 10-dimensional output layer with softmax activation.
"""
function initialize_mnist_mlp(h=[32])
    _h = [28*28; h]
    model = Chain(
        [
            Dense(_h[i] => _h[i+1], sigmoid)
            for i in 1:length(h)
        ]...,
        Dense(h[end] => 10), 
        softmax,
    )
    return model
end

function load_mnist_data()
    train_data = MLDatasets.MNIST(split=:train)  # i.e. split=:train
    test_data = MLDatasets.MNIST(split=:test)
    return train_data, test_data
end

function simple_loader(data::MNIST; batchsize::Int=64)
    x2dim = reshape(data.features, 28^2, :)
    yhot = Flux.onehotbatch(data.targets, 0:9)
    Flux.DataLoader((x2dim, yhot); batchsize, shuffle=true)
end

"""
    simple_accuracy(model, data)

Compute model accuracy, returned in percentage with 0.01% precision.
"""
function simple_accuracy(model, data::MNIST)
    (x, y) = only(simple_loader(data; batchsize=length(data)))  # make one big batch
    y_hat = model(x)
    iscorrect = Flux.onecold(y_hat) .== Flux.onecold(y)  # BitVector
    acc = round(100 * mean(iscorrect); digits=2)
    return acc
end

