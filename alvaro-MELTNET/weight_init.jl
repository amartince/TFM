function get_weights(params, nClasses, nfilt, init_method)

    # initialise network weights and biases

    nf = collect(2^x for x in 5:8)
    nfb = [32,32,32,64,64,64,128,128,128,256,256,256,256,256,256,256,128,128,128,64,64,64,32,32,32,32,32,32];
    se_rat = 8;

    params["enWeights1"][1] = init_method(nfilt, nfilt,  4, 32)
    params["enWeights2"][1] = init_method(nfilt, nfilt, 32, 32)
    params["enWeights3"][1] = init_method(    1,     1,  4, 32)

    params["Offset"][1] = Flux.zeros32(32, 1)
    params["Scale" ][1] = Flux.ones32( 32, 1)
    params["Offset"][2] = Flux.zeros32(32, 1)
    params["Scale" ][2] = Flux.ones32( 32, 1)

    for ii in eachindex(nfb[3:end])
        params["Offset"][ii+2] = Flux.zeros32(nfb[3:end][ii], 1)
        params["Scale" ][ii+2] = Flux.ones32( nfb[3:end][ii], 1)
    end

    for ii in 1:4 
        params["enBias1"][ii] = Flux.zeros32(1, 1, nf[ii])
        params["enBias2"][ii] = Flux.zeros32(1, 1, nf[ii])
        params["enBias3"][ii] = Flux.zeros32(1, 1, nf[ii])
    end

    for ii in 1:3 
        
        params["deBias1"][ii] = Flux.zeros32(1, 1, nf[4-ii])
        params["deBias2"][ii] = Flux.zeros32(1, 1, nf[4-ii])
        params["deBias3"][ii] = Flux.zeros32(1, 1, nf[4-ii])
        
        params["enWeights1"][ii+1] = init_method(nfilt, nfilt, nf[ii  ], nf[ii+1])
        params["enWeights2"][ii+1] = init_method(nfilt, nfilt, nf[ii+1], nf[ii+1])
        params["enWeights3"][ii+1] = init_method(    1,     1, nf[ii  ], nf[ii+1])
        
        params["deWeights1"][ii] = init_method(    2,     2, nf[4-ii], nf[5-ii])
        params["deWeights2"][ii] = init_method(nfilt, nfilt, nf[5-ii], nf[4-ii])
        params["deWeights3"][ii] = init_method(nfilt, nfilt, nf[4-ii], nf[4-ii])
        params["deWeights4"][ii] = init_method(    1,     1, nf[5-ii], nf[4-ii])
        
        params["seWeights1"][ii] = Flux.ones32(Int(nf[ii]./se_rat), 1, 1,              nf[ii]) 
        params["seWeights2"][ii] = Flux.ones32(             nf[ii], 1, 1, Int(nf[ii]./se_rat))
    end

    params["BiasOut"   ] = Flux.zeros32(1, 1,     nClasses)
    params["WeightsOut"] = init_method( 1, 1, 32, nClasses)

    params["ASPPWeights1" ] = init_method(nfilt, nfilt, 256, 256)
    params["ASPPWeights2" ] = init_method(nfilt, nfilt, 256, 256)
    params["ASPPWeights3" ] = init_method(nfilt, nfilt, 256, 256)
    params["ASPPWeights4" ] = init_method(nfilt, nfilt, 256, 256)
    params["ASPPWeights5" ] = init_method(    1,     1, 256, 256)

    params["ASPPWeights6" ] = init_method(nfilt, nfilt, 32, 32)
    params["ASPPWeights7" ] = init_method(nfilt, nfilt, 32, 32)
    params["ASPPWeights8" ] = init_method(nfilt, nfilt, 32, 32)
    params["ASPPWeights9" ] = init_method(nfilt, nfilt, 32, 32)
    params["ASPPWeights10"] = init_method(    1,     1, 32, 32)

    params["ASPPBias1" ] = Flux.zeros32(1, 1, 256)
    params["ASPPBias2" ] = Flux.zeros32(1, 1, 256)
    params["ASPPBias3" ] = Flux.zeros32(1, 1, 256)
    params["ASPPBias4" ] = Flux.zeros32(1, 1, 256)
    params["ASPPBias5" ] = Flux.zeros32(1, 1, 256)

    params["ASPPBias6" ] = Flux.zeros32(1, 1, 32)
    params["ASPPBias7" ] = Flux.zeros32(1, 1, 32)
    params["ASPPBias8" ] = Flux.zeros32(1, 1, 32)
    params["ASPPBias9" ] = Flux.zeros32(1, 1, 32)
    params["ASPPBias10"] = Flux.zeros32(1, 1, 32)

    return(params)

end

function initialise_state(state)

    # initialise state for the batchnorm function

    nf = [32,32,32,64,64,64,128,128,128,256,256,256,256,256,256,256,128,128,128,64,64,64,32,32,32,32,32,32];

    for ii in eachindex(nf)
        state["batchnorm"][ii]["TrainedMean"]     = Flux.zeros32(nf[ii], 1)
        state["batchnorm"][ii]["TrainedVariance"] = Flux.ones32( nf[ii], 1)
    end

    return(state)

end

function initialise_DAE_params(params, init_method)
    
    for i in [5, 8, 11, 14]
        params["Layers"][i]["Offset"         ] = Flux.zeros32(1, 1, 16)
        params["Layers"][i]["Scale"          ] = Flux.ones32( 1, 1, 16)
        params["Layers"][i]["TrainedMean"    ] = Flux.zeros32(1, 1, 16)
        params["Layers"][i]["TrainedVariance"] = Flux.ones32( 1, 1, 16)
    end

    params["Layers"][ 2]["Weights"] = init_method(3, 3,  1, 16)
    params["Layers"][ 2]["Bias"   ] = Flux.zeros32(  1,  1, 16)
    params["Layers"][ 4]["Weights"] = init_method(3, 3, 16, 16)
    params["Layers"][ 7]["Weights"] = init_method(3, 3, 16, 16)
    params["Layers"][10]["Weights"] = init_method(3, 3, 16, 16)
    params["Layers"][13]["Weights"] = init_method(3, 3, 16, 16)
    params["Layers"][16]["Weights"] = init_method(3, 3, 16,  1)
    params["Layers"][16]["Bias"   ] = Flux.zeros32(1)

    return(params)

end