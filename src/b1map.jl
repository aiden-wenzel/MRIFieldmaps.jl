import ForwardDiff
using ADTypes: AutoForwardDiff
using Optim: optimize, LBFGS 
import Optim

export roughness_penalty, regularizer, log_loss, unpack, b1_fit, complex_gaussian_noise, DAM

# TODO: Change (N, D) to (N1, N2)

"""
Compute the roughness penalty for a 2d image as defined by Eq. 41 in regularized b1 mapping paper.

# In
- `z` 2D image
- `nl_mls` Tuple of tuple offsets which describe the neighbor pixels to compute roughness

# Out
- `penalty` The roughness penalty of an image
"""
function roughness_penalty(
        z::AbstractMatrix;
        nl_mls::Tuple = ((1,0), (0, 1)),
    )

    # TODO: Assuming that kappa = 1. Need to allow kappa to not be 1.
    # Offsets to compute neighboring pixel differences.
    # In this case, M denotes the width of the image, not the measurement number.
    N, M = size(z)

    penalty = 0.0
    for offset in nl_mls
        n_l = offset[1]
        m_l = offset[2]
        for n in 1+abs(n_l):N-abs(n_l)
            for m in 1+abs(m_l):M-abs(m_l)
                penalty += (2 * z[n, m] - z[n - n_l, m - m_l] - z[n + n_l, m+m_l])^2
            end
        end
    end

    return penalty
end

"""
Compute the total roughness penalty over each coil as defined by Eq. 6 in regularized b1 mapping paper.

# In
- `zks` an array of sizes (K, N, D) where K denotes the number of coils, and N and D are the height and width of each image respectively

# Out
- `regularized_cost` returns the sum of roughness penalties accross each coil.
"""
function regularizer(zks::AbstractArray)
    K = size(zks, 3)
    regularized_cost = 0.0
    for k=1:K
        regularized_cost += roughness_penalty(zks[:, :, k])
    end
    return regularized_cost
end

"""
Eq. 5 in regularized b1 mapping paper.
zks are of shape (K, N, D)
f is of shape (N, D)
"""

"""
Leftmost term in equation 3. 
"""
function signal_model(
        zks::AbstractArray,
        fjs::AbstractArray,
        index::Tuple,
        Chi::AbstractArray,
        F::Function
    )
    n, d, m = index[1], index[2], index[3]
    fj = fjs[n, d]

    K = size(zks, 3)
    K == size(Chi, 2) || throw(ArgumentError("K's don't match"))
    coil_sum = 0.0
    for k in 1:K
        coil_sum += Chi[m, k] * zks[n, d, k]
    end

    return fj * F(coil_sum)

end

"""
Eq. 5
"""
function log_loss(
        zks::AbstractArray, 
        fjs::AbstractArray, 
        Chi::AbstractArray,
        Y::AbstractArray,
        F::Function,
    )
    N, D, K = size(zks)
    K == size(Chi, 2) || throw(ArgumentError("K's don't match"))

    M = size(Chi, 1)
 
    loss_sum = 0.0
    for n in 1:N
        for d in 1:D
            for m in 1:M
                loss_sum += 1/2 * abs(Y[n, d, m] - signal_model(zks, fjs, (n, d, m), Chi, F)) ^ 2
            end
        end
    end

    return loss_sum
end

"""
Eq. 4 in regularized b1 mapping paper.
"""
function psi(
        params::AbstractVector, 
        zdims::Tuple, 
        fdims::Tuple, 
        Beta::Real, 
        Y::AbstractArray,
        Chi::AbstractArray,
        F::Function, 
    )
    zks, fjs = unpack(params, zdims, fdims)
    return log_loss(zks, fjs, Chi, Y, F) + Beta * regularizer(zks)
end

function unpack(
        params::AbstractVector,
        zdims::Tuple,
        fdims::Tuple 
    )
    N, D, K = zdims
    N == fdims[1] || throw(ArgumentError("N's don't match."))
    D == fdims[2] || throw(ArgumentError("D's don't match."))

    f_num_elements = N * D
    z_num_elements = N*D*K
    zks = reshape(params[1:z_num_elements], zdims)
    fjs = reshape(params[z_num_elements + 1:z_num_elements+f_num_elements], fdims)
    return zks, fjs
end

"""
- `Y` is (N, D, M)
- `Chi` is (M, K)
"""
function b1_fit(
        Beta::Real,
        Y::AbstractArray,
        Chi::Matrix,
        F::Function
    )
    # TODO: check that the bottom half of Chi is twice the top half of Chi.

    N, D, M = size(Y)
    K = size(Chi, 2) 
    M == size(Chi, 1) || throw(ArgumentError("M's don't match."))
    zdims = (N, D, K)
    fdims = (N, D)
    cost(x::AbstractVector) = psi(x, zdims, fdims, Beta, Y, Chi, F)
    # TODO:
    x0 =  ones(N*D*K + N*D) * 0.2# TODO: Define initial guess
    options = Optim.Options(store_trace=true)
    out = Optim.optimize(cost, x0, LBFGS(), options; autodiff=AutoForwardDiff())
    zk_opt, fj_opt = unpack(out.minimizer, zdims, fdims)
    return zk_opt, fj_opt
end

function DAM(
    fj::AbstractMatrix,
    aj::AbstractMatrix,
    std::Float64
)
    N, D = size(fj)
    N == size(aj, 1) || throw(ArgumentError("N's don't match."))
    D == size(aj, 2) || throw(ArgumentError("D's don't match."))

    yj1 = fj .* sin.(aj) + std * randn(ComplexF32, (N, D))
    yj2 = fj .* sin.(2*aj) + std * randn(ComplexF32, (N, D))

    ydata = ones(ComplexF64, (N, D, 2))
    ydata[:, :, 1] = yj1
    ydata[:, :, 2] = yj2
    return ydata
end

function complex_gaussian_noise(
    std::Float64,
    dims::Tuple
)
    return std .* (randn(dims) + im .* randn(dims)) / sqrt(2)
end