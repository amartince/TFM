using Statistics
using Flux
using MAT
using CairoMakie
using Random
using FileIO
using Images
using BSON
include("MELTNET_Jan_aproach.jl")
include("weight_init_Jan_aproach.jl")



file = matopen("matlab/params.mat")
params = read(file, "new_params")
close(file)

file = matopen("matlab/state.mat")
state = read(file, "new_state")
close(file)

file = matopen("matlab/DAEnet.mat")
DAEnet = read(file, "new_DAEnet")
close(file)

# The last convolutional layer need some changes in the parameters
# due to the fact that the output number of dimensions is 1
DAEnet["Layers"][16]["Weights"] = reshape(DAEnet["Layers"][16]["Weights"], (3, 3, 16, 1))
DAEnet["Layers"][16]["Bias"   ] = [DAEnet["Layers"][16]["Bias"]]

# parameters obtained from Rossier 2020
c_min = -150
c_max = 10


# In order to have more data to train, we do some transformations
function more_data_train(data_matrix)
    

    maps_default      = data_matrix[:,:,:,1:85  ]
    maps_reverse      = reverse(data_matrix[:,:,:,1:85  ], dims = 1)

    maps_default_r90  = mapslices(rotr90, maps_default, dims=[1, 2])
    maps_reverse_r90  = mapslices(rotr90, maps_reverse, dims=[1, 2])

    maps_default_r180 = mapslices(rot180, maps_default, dims=[1, 2])
    maps_reverse_r180 = mapslices(rot180, maps_reverse, dims=[1, 2])

    maps_default_r270 = mapslices(rotl90, maps_default, dims=[1, 2])
    maps_reverse_r270 = mapslices(rotl90, maps_reverse, dims=[1, 2])

    all_default   = cat(maps_default, maps_default_r90, maps_default_r180, maps_default_r270, dims = 4)
    all_reverse   = cat(maps_reverse, maps_reverse_r90, maps_reverse_r180, maps_reverse_r270, dims = 4)

    all = cat(all_default, all_reverse, dims = 4)

    return(all)
end

# Load the input data
path = "../shrrosier-MELTNET-d8b077a/NClass_10/validation_inputs/"
for file_name in readdir(path)
    file_open = matopen(path*file_name)
    data = read(file_open, "input_out")
    close(file_open)
    data_matrix = reshape(reinterpret(Int8, data), 64, 64, 4, 1)
    if file_name == "input_000014.mat"
        data_all = data_matrix
    else
        global data_all = cat(data_matrix, data_all, dims = 4)
    end
end

X_train = more_data_train(data_all[:,:,:,1:85])
X_val   = data_all[:,:,:,86:end]

# Load the NEMO data
path = "../shrrosier-MELTNET-d8b077a/NClass_10/validation_meltrates/"
for file_name in readdir(path)
    file_open = matopen(path*file_name)
    data = read(file_open, "ab")
    close(file_open)
    data_matrix = reshape(data, 64, 64, 1, 1)
    if file_name == "ab_000014.mat"
        data_all_DAE = data_matrix
    else
        global data_all_DAE = cat(data_matrix, data_all_DAE, dims = 4)
    end
end

# Normalization of the values for the NN
data_all_DAE = Float32.(255/(c_min-c_max).*(data_all_DAE.-c_max))
replace!(data_all_DAE, NaN => 0)

Y_train = more_data_train(data_all_DAE[:,:,:,1:85])
Y_val   = data_all_DAE[:,:,:,86:end]



function loss(model, x, y)
    Flux.Losses.huber_loss(model(x), y)
end

# Initialise the weights
nClasses = 10
nfilt = 3
init_method = Flux.glorot_normal()

params_2 = get_weights(params, nClasses, nfilt, init_method)
state_2  = initialise_state(state)
DAEnet_2 = initialise_DAE_params(DAEnet, init_method)

# Create the dataloadet and the model
dataloader_yelmo = Flux.DataLoader((X_train, Y_train), batchsize = 20, shuffle = true)
model_yelmo = MELTNET_Jan_aproach(params_2, nClasses, state_2, true)

# Set the learning rate (lr) to 0.0005
opt_state = Flux.setup(Adam(0.0005), model_yelmo)

println("Starting training...")

epoch = 100
loss_train = fill(NaN32, epoch)
loss_val   = fill(NaN32, epoch)

for i in 1:epoch
    Flux.train!(model_yelmo, dataloader_yelmo, opt_state) do m, x, y
    loss(m, x, y)
    end
    loss_train[i] = loss(model_yelmo, X_train, Y_train)
    loss_val[i]   = loss(model_yelmo, X_val, Y_val)
    println("Epoch ", i, " clompleted")
end

# Obtain the model state and save it
estado_modelo = Flux.state(model_yelmo)
BSON.@save "MELTNET_Jan_aproach_100epoch_randomweight_lr0005.bson" model_state = estado_modelo

with_theme(theme_latexfonts()) do
    f = Figure(size = (900, 600),
           fontsize = 25
           )

    lines(f[1, 1], 
          1:epoch, 
          loss_train, 
          label = "Training data", 
          color = :red, axis = (xlabel = "Epoch", 
                                ylabel = "Loss function (Logaritmic sacale)", 
                                yscale = log10,
                                ytickformat = "{:.3f}"
                               )
         )
    lines!(f[1, 1], 1:epoch, loss_val, label = "Validation data", color = :blue)
    axislegend(position =:rt)
    current_figure()


    #CairoMakie.save("imagenes/MELTNET_Jan_aproach_100epoch_randomweight_lr0005.png", f)
    f
end