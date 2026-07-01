export roughness_penalty, regularizer, L

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
    K = size(zks, 1)
    regularized_cost = 0.0
    for k=1:K
        regularized_cost += roughness_penalty(zks[k, :, :])
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
function signal_model(f_j, F::Function, Chi::AbstractArray, zks::AbstractArray)
    K = size(zks, 1)
    coil_sum = 0.0
    for k in 1:K
        coil_sum += Chi[m, k] * z[k, j]
    end
    return f_j * coil_sum
end

# What should the shape of zks be?
function log_loss(zks::AbstractArray, f, F::Function, Chi::AbstractArray, Y::AbstractArray)
    K, N, D = size(zks)
    M = size(Chi, 1)
    K == size(Chi, 2) || throw(ArgumentError("size mismatch"))
 
    loss_sum = 0.0
    for m in 1:M
        for n in 1:N
            for d in 1:D
                loss_sum += 1/2 * abs(Y[n, d] - signal_model(f[n, d], F, Chi, zks)) ^ 2
            end
        end
    end
end

"""
Eq. 4 in regularized b1 mapping paper.
Cost function to optimze.
params = [zks, f] where size(zks) = (K, N, D)
and size(f) = (N, D) and beta is a real constant.
"""
function psi(params::AbstractVector, F::Function, Beta::Real, zdims::Tuple, fdims::Tuple)
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