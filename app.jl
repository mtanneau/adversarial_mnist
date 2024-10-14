module App
using PlotlyJS
include("app/mlp.jl")
using GenieFramework
@genietools

using Random
using LinearAlgebra
BLAS.set_num_threads(2)

"""

Plot gray-scale MNIST image
"""
function plot_mnist_grayscale(x; title="")
    tr = heatmap(
        z=reshape(x, 28, 28)'[28:-1:1, :],
        colorscale=PlotlyJS.colors.grays,
        showscale=false,
    )
    l = PlotlyJS.Layout(
        title = title,
        autosize = false,
        width = 512,
        height = 512,
        xaxis = attr(showgrid = false, zeroline = false, showticklabels = false),
        yaxis = attr(showgrid = false, zeroline = false, showticklabels = false),
    )
    return [tr], l
end

@app begin
    model = initialize_mnist_mlp()
    _state = Flux.state(model)

    # MLP 
    @out traces = []
    @in train = false
    @in training = false

    @out layout = PlotlyJS.Layout(
        title="Neural network training progress",
        autosize = false,
        width = 800,
        height = 600,
        xaxis_title = "# training epochs",
        xaxis_range=[1, 32],
        yaxis_title = "Accuracy (%)",
        yaxis_range=[75, 100],
    )

    @onchange train begin
        training = true

        # model = initialize_model()
        train_data, test_data = load_mnist_data()
        
        acc_train_all = []
        acc_test_all  = []

        train_loader = simple_loader(train_data, batchsize = 256)
        opt_state = Flux.setup(Adam(3e-4), model);

        for epoch in 1:32
            loss = 0.0
            for (x, y) in train_loader
                # Compute the loss and the gradients:
                l, gs = Flux.withgradient(m -> Flux.crossentropy(m(x), y), model)
                # Update the model parameters (and the Adam momenta):
                Flux.update!(opt_state, model, gs[1])
                # Accumulate the mean loss, just for logging:
                loss += l / length(train_loader)
            end
        
            # if mod(epoch, 2) == 1
            # Train / test accuracy
            train_acc = simple_accuracy(model, train_data)
            test_acc  = simple_accuracy(model, test_data)

            push!(acc_train_all, train_acc)
            push!(acc_test_all, test_acc)
            @info "After epoch = $epoch" loss train_acc test_acc

            tr_train = PlotlyJS.scatter(x=collect(1:epoch), y=acc_train_all, mode="lines", name="train")
            tr_test  = PlotlyJS.scatter(x=collect(1:epoch), y=acc_test_all, mode="lines", name="test")

            traces = [tr_train, tr_test]
        end
        training = false
    end

    # adversarial examples code stuff
    @out img_idx = 1
    @out p_true = 0  # true label
    @out p_pred = 0  # original prediction
    @out p_adv  = 0  # prediction on adversarial example

    @out tr_img_raw = []  # image 
    @out layout_img_raw = PlotlyJS.Layout(
        title = "Original Image",
        autosize = false,
        width = 512,
        height = 512,
        xaxis = attr(showgrid = false, zeroline = false, showticklabels = false),
        yaxis = attr(showgrid = false, zeroline = false, showticklabels = false),
    )
    @out tr_img_adv = []  # adversarial image
    @out layout_img_adv = PlotlyJS.Layout(
        title = "Noisy Image",
        autosize = false,
        width = 512,
        height = 512,
        xaxis = attr(showgrid = false, zeroline = false, showticklabels = false),
        yaxis = attr(showgrid = false, zeroline = false, showticklabels = false),
    )

    @in sample_img = false
    @in noise_level = 0

    _x = zeros(Float32, 28*28)  # test image, in memory
    _x_adv = zeros(Float32, 28*28)
    _eps = zeros(Float32, 28*28)  # adversarial noise

    @onchange sample_img begin
        img_idx = rand(1:10_000)

        # Re-set the seed so that every image has the same ensures that we always re-generate the same noise
        Random.seed!(img_idx)  
        _eps .= rand(Float32, 28*28)

        test_data = MLDatasets.MNIST(split=:test)
        xtest = reshape(test_data.features, 28*28, :)
        ytest = test_data.targets

        _x .= xtest[:, img_idx]
        _x_adv .= clamp.(
            _x .+ (noise_level ./ 1f3) .* _eps,
            0f0,
            1f0,
        )
        p_true = ytest[img_idx]
        p_pred = argmax(model(_x)) - 1
        p_adv = argmax(model(_x_adv)) - 1
        tr_img_raw, layout_img_raw = plot_mnist_grayscale(_x;
            title="Image #$(img_idx) (original): Output: $(p_pred)"
        )
        tr_img_adv, layout_img_adv = plot_mnist_grayscale(_x_adv;
            title="Image #$(img_idx) (noisy); Output: $(p_adv)"
        )
    end

    @onchange noise_level begin
        # Only update 
        _x_adv .= clamp.(
            _x .+ (noise_level ./ 1f3) .* _eps,
            0f0,
            1f0,
        )
        p_adv = argmax(model(_x_adv)) - 1
        tr_img_adv, layout_img_adv = plot_mnist_grayscale(_x_adv;
            title="Image #$(img_idx) (noisy); Output: $(p_adv)"
        )
    end
end

@page("/","app.jl.html")
end
