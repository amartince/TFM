using Statistics
using Flux
using MAT
using CairoMakie
using Random
using FileIO
using Images
using BSON
include("matconv.jl")
include("MELTNET.jl")



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

X_train = data_all[:,:,:,1:85  ]
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

Y_train = data_all_DAE[:,:,:,1:85  ]
Y_val   = data_all_DAE[:,:,:,86:end]

# Crate the network
model = MELTNET(params, DAEnet, 10, state, false)

# Load the parameters
Flux.loadmodel!(model, BSON.load("MELTNET_100epoch_randomweight_lr0005.bson")[:model_state])

# Eval the network
y_model = model(X_val)

# Undo the normalization
c_min = -150
c_max = 10
y_model = y_model.*(c_min - c_max)/255 .+ c_max

# This is done to put NaN in the same positions as the Y_val matrices
y_model = y_model+Y_val-Y_val

# Set the color range
color_range_1 = (-150, 10)

# Visualization of the validation geometries

idx_image = round.(Int, LinRange(1, 15, 15))
idx_image = reshape(idx_image, (3, 5)) #This is only to have the images in order

with_theme(theme_latexfonts()) do
    f = Figure(size = (2000, 2000),
           fontsize = 25
           )

    figure_grid     = f[1:8, 1:5] = GridLayout()
    colorbar_layout = f[1:8, 6]   = GridLayout()

    for i in 1:3
        Label(figure_grid[Int(1+2*(i-1)), 0], text = "Predicted\nmelt rates", alignmode = Outside(200, 0, 100, 100))
        Label(figure_grid[Int(2*i)      , 0], text = "Real\nmelt rates"     , alignmode = Outside(200, 0, 100, 100))
    end

    for (idx, value) in pairs(idx_image)
        heatmap(figure_grid[Int(1+2*(idx[1]-1)), Int(idx[2])], y_model[:, :, 1, value], colorrange = color_range_1)
        heatmap(figure_grid[Int(2*idx[1])      , Int(idx[2])], Y_val[  :, :, 1, value], colorrange = color_range_1)
    end

    Colorbar(colorbar_layout[1:8, 1], colorrange = color_range_1, label = L"$\mathrm{m \, yr^{-1}}$")

    #CairoMakie.save("imagenes/comparativa_xl_NaN.png", f)
    f
end


# Creation of a histogram

box_vector = filter(!isnan, (vec(y_model).-vec(Y_val))./abs.(vec(Y_val)))
with_theme(theme_latexfonts()) do
    f = Figure(size = (600, 600),
           fontsize = 25
           )

    rainclouds(f[1, 1], ones(length(box_vector)), box_vector;
        axis = (; ylabel = "",
        xlabel = "melt rate difference", title = "MELTNET - NEMO normalized\n(validation set)",
        xminorticks = IntervalsBetween(5), xminorticksvisible = true, xminorgridvisible = true),
        orientation = :horizontal,
        plot_boxplots = true, cloud_width=0.75, clouds = hist, 
        show_median = false,
        hist_bins = 10000
    )

    hideydecorations!(grid = false)
    xlims!(-5, 5)

    #CairoMakie.save("imagenes/hist_norm_meltnet-nemo.png", f)
    f
end