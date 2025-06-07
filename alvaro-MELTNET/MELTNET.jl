using Flux
include("matconv.jl")


function squeeze(A::AbstractArray)
    singleton_dims = tuple((size(A, d) for d in 1:ndims(A) if size(A, d) != 1)...) 
    # los 3 puntos permiten que la funcion tuple reciba un número variable de argumentos
    reduced_A = reshape(A, singleton_dims)
    return reduced_A
end



function masked_matrix(x1, x2, cte)
    z = x1.*x2 + (ones(size(x2))-x2*1)*cte
    return(z)
end



function batchnorm_state(params, idx, state, training)
    if training

        out = BatchNorm(size(params["Offset"][idx])[1]; affine = true)

        out.β  .= squeeze(params["Offset"][idx])
        out.γ  .= squeeze(params["Scale" ][idx])  
        out.μ  .= state["batchnorm"][idx]["TrainedMean"    ]
        out.σ² .= state["batchnorm"][idx]["TrainedVariance"]

        trainmode!(out)

        return(out)
    
    else

        # β desplazamiento (offset)
        # γ factor de escala
        out = BatchNorm(size(params["Offset"][idx])[1]; affine = true)

        out.β  .= squeeze(params["Offset"][idx])
        out.γ  .= squeeze(params["Scale" ][idx])
        out.μ  .= state["batchnorm"][idx]["TrainedMean"    ]
        out.σ² .= state["batchnorm"][idx]["TrainedVariance"]

        testmode!(out)

        return(out)

    end
end


function batchnorm_DAE(params_DAE, idx, training)
    if training

        out = BatchNorm(size(params_DAE["Layers"][idx]["Offset"])[3]; affine = true)

        out.β  .= squeeze(params_DAE["Layers"][idx]["Offset"         ])
        out.γ  .= squeeze(params_DAE["Layers"][idx]["Scale"          ])  
        out.μ  .= squeeze(params_DAE["Layers"][idx]["TrainedMean"    ])
        out.σ² .= squeeze(params_DAE["Layers"][idx]["TrainedVariance"])

        trainmode!(out)

        return(out)
    
    else

        # β desplazamiento (offset)
        # γ factor de escala
        out = BatchNorm(size(params_DAE["Layers"][idx]["Offset"])[3]; affine = true)

        out.β  .= squeeze(params_DAE["Layers"][idx]["Offset"         ])
        out.γ  .= squeeze(params_DAE["Layers"][idx]["Scale"          ])  
        out.μ  .= squeeze(params_DAE["Layers"][idx]["TrainedMean"    ])
        out.σ² .= squeeze(params_DAE["Layers"][idx]["TrainedVariance"])

        testmode!(out)

        return(out)

    end
end


function SE_encoder(idx, params, state, training)
    # Squeeze and excited encoder block

    nf = collect(2^x for x in 5:8)
    sidx1 = (idx-1)*3 .+ collect(3:5) # batch norm state index
    
    model = Chain(

        se1out = Parallel(

            .* ,

            Chain(
                # squeeze and excite block
                se1a  = GlobalMeanPool(),
                # Squeeze elimina las dimensiones de tamaño 1 porque dan problemas
                se1as = x -> squeeze(x),

                se1b  = Dense(squeeze(params["seWeights1"][idx]), zeros(Float32, Int(nf[idx]/8))),
                se1c  = x -> x.*sigmoid(x),
                se1d  = Dense(squeeze(params["seWeights2"][idx]), zeros(Float32, nf[idx])),
                se1e  = x -> x.*sigmoid(x),
                se1er = x -> reshape(x, (1, 1, size(x)...)),
            )  ,

            x -> x

        ),

        out = Parallel(

            +,

            Chain(
                # main convolutional block
                bn2a  = batchnorm_state(params, sidx1[1], state, training),
                r2a   = x -> x.*sigmoid(x),
                c2a   = MatConv(params["enWeights1"][idx+1], squeeze(params["enBias1"][idx+1]); stride = 2, pad = SamePad()),
                bn2b  = batchnorm_state(params, sidx1[2], state, training),
                r2b   = x -> x.*sigmoid(x),
                c2b   = MatConv(params["enWeights2"][idx+1], squeeze(params["enBias2"][idx+1]); stride = 1, pad = SamePad()),
            ),

            # shortcut
            Chain(
                c2c   = MatConv(params["enWeights3"][idx+1], squeeze(params["enBias3"][idx+1]); stride = 2, pad = SamePad()),
                bn2c  = batchnorm_state(params, sidx1[3], state, training)
                
            )

        )

    )

    
    return(model)

end


function SE_decoder_subfunction(idx, params)
     # Squeeze and excited decoder block
     # First layer of the block due to Julia syntaxis

    layer = MatConvTranspose(params["deWeights1"][idx], squeeze(params["deBias1"][idx]); stride = 2, pad = 0)

    return(layer)

end


function SE_decoder(idx, params, state, training)
    # Squeeze and excited decoder block

    sidx1 = (idx-1)*3 .+ collect(16:18)
    
    model = Chain(
        out = Parallel(
            
            +,

            Chain(
                # main convolutional block
                bn5a  = batchnorm_state(params, sidx1[1], state, training), # Entra el q5
                r5a   = x -> x.*sigmoid(x),
                c5a   = MatConv(params["deWeights2"][idx], squeeze(params["deBias1"][idx]); stride = 1, pad = SamePad()),
                bn5b  = batchnorm_state(params, sidx1[2], state, training),
                c5b   = MatConv(params["deWeights3"][idx], squeeze(params["deBias2"][idx]); stride = 1, pad = SamePad()),
            ),

            Chain(
                c5c   = MatConv(params["deWeights4"][idx], squeeze(params["deBias3"][idx]); stride = 1, pad = SamePad()),
                bn5c  = batchnorm_state(params, sidx1[3], state, training)
            )

        )

    )

    return(model)

end



function lossMELTNET_SEG(params, nC, state, training)

    model = Parallel(
        
        (x1, x2) -> masked_matrix(x1, x2, 0),

        Chain(
            add1 = Parallel(

                +,

                Chain(
                    c1a   = MatConv(params["enWeights1"][1], squeeze(params["enBias1"][1]); stride = 1, pad = SamePad()),
                    bn1a  = batchnorm_state(params, 1, state, training),
                    r1b   = x -> x.*sigmoid(x),
                    c1b   = MatConv(params["enWeights2"][1], squeeze(params["enBias2"][1]); stride = 1, pad = SamePad()),
                ),

                Chain(
                    c1c   = MatConv(params["enWeights3"][1], squeeze(params["enBias3"][1]); stride = 1, pad = SamePad()),
                    bn1c  = batchnorm_state(params, 2, state, training)
                )
            ),

            add7_q5 = Parallel(

                (x1, x2) -> cat(x1, x2; dims = 3),

                Chain(
                    add2     = SE_encoder(1, params, state, training),
                    add6_q5  = Parallel(

                        (x1, x2) -> cat(x1, x2; dims = 3),

                        Chain(
                            add3     = SE_encoder(2, params, state, training),
                            add5_q5  = Parallel(

                                (x1, x2) -> cat(x1, x2; dims = 3),

                                Chain(
                                    add4     = SE_encoder(3, params, state, training),
                                    aspp1    = Parallel(

                                    +,

                                    Chain(
                                        x1a = MatConv(params["ASPPWeights1"], squeeze(params["ASPPBias1"]); pad = SamePad(), dilation = (6, 6)),
                                        x1b = batchnorm_state(params, 12, state, training)
                                    ),

                                    Chain(
                                        x2a = MatConv(params["ASPPWeights2"], squeeze(params["ASPPBias2"]); pad = SamePad(), dilation = (12, 12)),
                                        x2b = batchnorm_state(params, 13, state, training)
                                    ),

                                    Chain(
                                        x3a = MatConv(params["ASPPWeights3"], squeeze(params["ASPPBias3"]); pad = SamePad(), dilation = (18, 18)),
                                        x3b = batchnorm_state(params, 14, state, training)
                                    ),

                                    Chain(
                                        x4a = MatConv(params["ASPPWeights4"], squeeze(params["ASPPBias4"]); pad = SamePad()),
                                        x4b = batchnorm_state(params, 15, state, training)
                                    )


                                    ),

                                    aspp2    = MatConv(params["ASPPWeights5"], squeeze(params["ASPPBias5"]); pad = SamePad()),
                                    add5_c5u = SE_decoder_subfunction(1, params)
                                ),

                                x -> x
                            ),

                            add5     = SE_decoder(1, params, state, training),
                            add6_c5u = SE_decoder_subfunction(2, params)

                        ),

                        x -> x
                        
                    ),

                    add6     = SE_decoder(2, params, state, training),
                    add7_c5u = SE_decoder_subfunction(3, params)

                ),

                x -> x

            ),

            add7 = SE_decoder(3, params, state, training),

            aspp3 = Parallel(

                +,

                Chain(
                    y1a = MatConv(params["ASPPWeights6"], squeeze(params["ASPPBias6"]); pad = SamePad(), dilation = (6, 6)),
                    y1b = batchnorm_state(params, 25, state, training)
                ),

                Chain(
                    y2a = MatConv(params["ASPPWeights7"], squeeze(params["ASPPBias7"]); pad = SamePad(), dilation = (12, 12)),
                    y2b = batchnorm_state(params, 26, state, training)
                ),

                Chain(
                    y3a = MatConv(params["ASPPWeights8"], squeeze(params["ASPPBias8"]); pad = SamePad(), dilation = (18, 18)),
                    y3b = batchnorm_state(params, 27, state, training)
                ),

                Chain(
                    y4a = MatConv(params["ASPPWeights9"], squeeze(params["ASPPBias9"]); pad = SamePad()),
                    y4b = batchnorm_state(params, 28, state, training)
                )

            ),

            aspp4 = MatConv(params["ASPPWeights10"], squeeze(params["ASPPBias10"]); pad = SamePad()),
            cfin = MatConv(params["WeightsOut"], squeeze(params["BiasOut"]); stride = 1, pad = SamePad()),
            sm = softmax,
            yPred1 = x -> squeeze(maximum(x[:, :, 1:nC-1, :], dims = 3)),

        ),

        x -> squeeze(x[:, :, 2, :] .!= 0)

    )

    return(model)

end



function DAE_network(params_DAE, training)

    model = Chain(
        Conv1   = MatConv(params_DAE["Layers"][ 2]["Weights"], squeeze(params_DAE["Layers"][ 2]["Bias"]); pad = (1, 1, 1, 1)),
        ReLU1   = x -> swish(x),
        Conv2   = MatConv(params_DAE["Layers"][ 4]["Weights"], false; pad = (1, 1, 1, 1)),
        BNorm2  = batchnorm_DAE(params_DAE,  5, training),
        ReLU2   = x -> swish(x),
        Conv3   = MatConv(params_DAE["Layers"][ 7]["Weights"], false; pad = (1, 1, 1, 1)),
        BNorm3  = batchnorm_DAE(params_DAE,  8, training),
        ReLU3   = x -> swish(x),
        Conv4   = MatConv(params_DAE["Layers"][10]["Weights"], false; pad = (1, 1, 1, 1)),
        BNorm4  = batchnorm_DAE(params_DAE, 11, training),
        ReLU4   = x -> swish(x),
        Conv5   = MatConv(params_DAE["Layers"][13]["Weights"], false; pad = (1, 1, 1, 1)),
        BNorm5  = batchnorm_DAE(params_DAE, 14, training),
        ReLU5   = x -> swish(x),
        Conv6   = MatConv(params_DAE["Layers"][16]["Weights"], params_DAE["Layers"][16]["Bias"]; pad = (1, 1, 1, 1))
    )

    return(model)
    
end


function MELTNET(params_SEG, params_DAE, nC, state, training)

    model = Chain(
        lossMELTNET_SEG(params_SEG, nC, state, training),
        x -> reshape(Float32.(x), 64, 64, 1, :),
        DAE_network(params_DAE, training)
    )

    return(model)

end