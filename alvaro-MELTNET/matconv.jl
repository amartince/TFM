using Flux


# Convolutional layer modified to get as matlab does

struct MatConv{N,M,F,A,V}
    σ::F
    weight::A
    bias::V
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    dilation::NTuple{N,Int}
    groups::Int
end

function MatConv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
    init = Flux.glorot_uniform, stride = 1, pad = 0, dilation = 1, groups = 1,
    bias = true) where N

    weight = Flux.convfilter(k, ch; init, groups)
    MatConv(weight, bias, σ; stride, pad, dilation, groups)
end

function MatConv(w::AbstractArray{T,N}, b = true, σ = identity;
    stride = 1, pad = 0, dilation = 1, groups = 1) where {T,N}
    c = Flux.Conv(w, b, σ; stride = stride, pad = pad, dilation = dilation, groups = groups)
    return MatConv(c.σ, c.weight, c.bias, c.stride, c.pad, c.dilation, c.groups)
end
Flux._channels_in(l::MatConv) = size(l.weight, ndims(l.weight)-1) * l.groups
Flux._channels_out(l::MatConv) = size(l.weight, ndims(l.weight))

Flux.conv_dims(c::MatConv, x::AbstractArray) =
  DenseConvDims(x, c.weight; stride = c.stride, padding = c.pad, dilation = c.dilation, groups = c.groups)



function (c::MatConv)(x::AbstractArray)
    Flux._conv_size_check(c, x)
    cdims = Flux.conv_dims(c, x)
    xT = Flux._match_eltype(c, x)
    M = Flux.conv(xT, c.weight, cdims)
    p = tuple((min(c.pad[i], size(c.weight)[1]) for i in 1:length(c.pad))...)
    Mmat_center = M[p[1]+1:end-p[1], p[2]+1:end-p[2], :, :]
    Mmat_left   = reverse(M[end-p[1]+1:end, p[2]+1:end-p[2], :, :], dims = (1, 2))
    Mmat_right  = reverse(M[1:p[1], p[2]+1:end-p[2], :, :], dims = (1, 2))
    Mmat_top    = reverse(M[:, end-p[2]+1:end, :, :], dims = (1, 2))
    Mmat_bottom = reverse(M[:, 1:p[2], :, :], dims = (1, 2))
    Mmat_aux    = cat(Mmat_left, Mmat_center, Mmat_right, dims = 1)
    Mmat        = cat(Mmat_top, Mmat_aux, Mmat_bottom, dims = 2)
    NNlib.bias_act!(c.σ, Mmat, Flux.conv_reshape_bias(c))
end



# Convolucional transpuesta matlab

struct MatConvTranspose{N,M,F,A,V}
    σ::F
    weight::A
    bias::V
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    outpad::NTuple{N,Int}
    dilation::NTuple{N,Int}
    groups::Int
end
  
Flux._channels_in(l::MatConvTranspose)  = size(l.weight)[end]
Flux._channels_out(l::MatConvTranspose) = size(l.weight)[end-1]*l.groups

function MatConvTranspose(w::AbstractArray{T,N}, bias = true, σ = identity;
                    stride = 1, pad = 0, outpad = 0, dilation = 1, groups = 1) where {T,N}
    c = Flux.ConvTranspose(w, bias, σ; stride = stride, pad = pad, outpad = outpad, dilation = dilation, groups = groups)
    return MatConvTranspose(c.σ, c.weight, c.bias, c.stride, c.pad, c.outpad, c.dilation, c.groups)
end
  
function MatConvTranspose(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
                        init = glorot_uniform, stride = 1, pad = 0, outpad = 0, dilation = 1,
                        groups = 1,
                        bias = true,
                        ) where N

    weight = Flux.convfilter(k, reverse(ch); init, groups)                    
    MatConvTranspose(weight, bias, σ; stride, pad, outpad, dilation, groups)
end
  
  
function Flux.conv_transpose_dims(c::MatConvTranspose, x::AbstractArray)
    # Calculate combined pad in each dimension
    nd = ndims(x) - 2
    if length(c.pad) == nd
        # Handle symmetric non-constant padding
        combined_pad = ntuple(i -> 2 * c.pad[i], nd)
    else
        combined_pad = ntuple(i -> c.pad[2i-1] + c.pad[2i], nd)
    end

    # Calculate size of "input", from ∇conv_data()'s perspective...
    calc_dim(xsz, wsz, stride, dilation, pad, outpad) = (xsz - 1) * stride + 1 + (wsz - 1) * dilation - pad + outpad
    I = map(calc_dim, size(x)[1:end-2], size(c.weight)[1:end-2], c.stride, c.dilation, combined_pad, c.outpad)
    C_in = size(c.weight)[end-1] * c.groups
    batch_size = size(x)[end]

    # Create DenseConvDims() that looks like the corresponding conv()
    w_size = size(c.weight)
    return DenseConvDims((I..., C_in, batch_size), w_size;
                        stride=c.stride,
                        padding=c.pad,
                        dilation=c.dilation,
                        groups=c.groups,
    )
end
  
  
function (c::MatConvTranspose)(x::AbstractArray)
    Flux._conv_size_check(c, x)
    cdims = Flux.conv_transpose_dims(c, x)
    xT = Flux._match_eltype(c, x)
    M = Flux.∇conv_data(xT, c.weight, cdims)
    p = tuple((min(c.pad[i], size(c.weight)[1]) for i in 1:length(c.pad))...)
    Mmat_center = M[p[1]+1:end-p[1], p[2]+1:end-p[2], :, :]
    Mmat_left   = reverse(M[end-p[1]+1:end, p[2]+1:end-p[2], :, :], dims = (1, 2))
    Mmat_right  = reverse(M[1:p[1], p[2]+1:end-p[2], :, :], dims = (1, 2))
    Mmat_top    = reverse(M[:, end-p[2]+1:end, :, :], dims = (1, 2))
    Mmat_bottom = reverse(M[:, 1:p[2], :, :], dims = (1, 2))
    Mmat_aux    = cat(Mmat_left, Mmat_center, Mmat_right, dims = 1)
    Mmat        = cat(Mmat_top, Mmat_aux, Mmat_bottom, dims = 2)
    NNlib.bias_act!(c.σ, Mmat, Flux.conv_reshape_bias(c))
end