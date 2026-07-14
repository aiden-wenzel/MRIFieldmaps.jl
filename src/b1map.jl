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

function H(
    z::AbstractArray,
    F::Function
)
    return F.(z) ./ exp.(im .* z)
end

function compute_x_hat(
    Y::AbstractArray,
    K::Int,
    F::Function
)
    N, D, M = size(Y)
    x_hat_mags = zeros((N, D, K))
    for m in 1:K
        x_hat_mags[:, :, m] = acos.(0.5 .* abs.(Y[:, :, m + K] ./ Y[:, :, m]))
    end

    x_hat_angles = zeros((N, D, K))
    for m in 1:K
        x_hat_angles[:, :, m] = angle.(Y[:, :, m]) - angle.(H(abs.(x_hat_mags[:, :, K]), F))
    end

    return x_hat_mags .* exp.(im .* x_hat_angles)
end

function compute_composite_map(
    z_hat::AbstractArray,
    chi::AbstractMatrix
)
    N, D, K = size(z_hat)
    M = size(chi, 1)
    K == size(chi, 2) || throw(ArgumentError("K's don't match."))

    x = zeros(N, D, M)
    for k in 1:K
        z_flat = reshape(z_hat[:, :, k], :, 1) # N*D x 1
        x_perm = chi[:, k] * z_flat' # M x N*D
        x_perm = reshape(x_perm, M, N, D) # M x N x D
        x += permutedims(x_perm, (2, 3, 1)) # N, D, M
    end

    return x
end

function compute_f_hat(
    Y::AbstractArray,
    z_hat::AbstractArray,
    chi::AbstractMatrix,
    F::Function
)
    N, D, M = size(Y)
    xjm = compute_composite_map(z_hat, chi)

    top = zeros(N, D)
    bottom = zeros(N, D)

    for m in 1:M
        top += real.(conj.(Y[:, :, m]) .* F.(xjm[:, :, m]))
        bottom += abs.(F.(xjm[:, :, m])) .^ 2
    end

    return top ./ bottom
end

function b1_fit(
        Beta::Real,
        Y::AbstractArray,
        Chi::Matrix,
        F::Function
    )
    # TODO: check that the bottom half of Chi is twice the top half of Chi.

    N, D, M = size(Y)
    K = size(Chi, 2) 
    zdims = (N, D, K)
    fdims = (N, D)

    # Error checking
    M == size(Chi, 1) || throw(ArgumentError("M's don't match."))
    M == 2*K || throw(ArgumentError("Chi must be M x M/2."))
    Chi_tilda = Chi[1:Int32(M/2), :]
    isapprox(Chi[Int32(M/2) + 1:end, :], 2*Chi_tilda) || throw(ArgumentError("The bottom half of Chi must be twice the top half of Chi."))
    
    # Finding initial guess.
    x_hat = compute_x_hat(Y, K, F) # x_hat is N, D, K
    x_hat_perm = PermutedDimsArray(x_hat, (3, 1, 2))
    x_hat_flat = reshape(x_hat_perm, K, :)
    z_hat_flat = Chi_tilda \ x_hat_flat
    z_hat_perm = reshape(z_hat_flat, K, N, D)

    z_hat = permutedims(z_hat_perm, (2, 3, 1))
    f_hat = compute_f_hat(Y, z_hat, Chi, F)

    z_hat_flat = reshape(z_hat, :)
    f_hat_flat = reshape(f_hat, :)

    size(z_hat_flat) == (N*D*K,) || throw(ArgumentError(""))
    size(f_hat_flat) == (N*D,) || throw(ArgumentError(""))

    x0 = abs.([z_hat_flat; f_hat_flat])
    cost(x::AbstractVector) = psi(x, zdims, fdims, Beta, Y, Chi, F)
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