export roughness_penalty, regularizer, log_loss

"""
Eq 41 in regularized b1 mapping paper.
TODO: Assuming that kappa = 1. Need to allow kappa to not be 1.
z should be an image of shape (N, M)
"""
function roughness_penalty(
        z::AbstractMatrix;
        nl_mls::Tuple = ((1,0), (0, 1)),
    )

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
Eq. 6 in regularized b1 mapping paper.
zks are of shape (K, N, D)
"""
function regularizer(zks)
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
function signal_model(zks::AbstractArray, fjs::AbstractArray, index::Tuple, Chi::AbstractArray, F::Function)
    n, d, m = index[1], index[2], index[3]
    fj = fjs[n, d]

    K = size(zks)
    K == size(Chi, 2) || throw(ArgumentError("K's don't match"))
    coil_sum = 0.0
    for k in 1:K
        coil_sum += Chi[m, k] * z[n, d, k]
    end

    return fj * F(coil_sum)

end

"""
Eq. 5
"""
function log_loss(zks::AbstractArray, fjs::AbstractArray, Chi::AbstractArray, Y::AbstractArray, F::Function,)
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
        F::Function, 
        Y::AbstractArray,
        Chi::AbstractArray
    )
    # TODO: unpack zks, and fj's from params
    zks, fjs = 
    return log_loss(zks, fjs, F, Chi, Y) + Beta * regularizer(zks)
end

function unpack(
        params::AbstractVector,
        zdims::Tuple,
        fdims::Tuple 
    )
    N, D, K = zdims
    z_num_elements = N*D*K

    N == size(fdims, 1) || throw(ArgumentError("N's doin't match."))
    D == size(fdims, 2) || throw(ArgumentError("D's doin't match."))

    f_num_elements = N * D
    zks = reshape(params[1:z_num_elements], zdims)
    fjs = reshape(params[z_num_elements + 1:end], fdims)
    return zks, fjs
end

# function big_fit(yjk, chi, F, beta)
#     cost(x) = psi(x, F, beta, size(z), size(f))
#     z = Optim.optimize(cost, ...)
#     return z.minimizer
# end  

# This code will go in a test file.
#=
# load data
beta = 7
F = sin
=#