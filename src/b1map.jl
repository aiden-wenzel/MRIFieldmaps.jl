using Zygote
using ADTypes: AutoReverseDiff, AutoZygote
using Optim: optimize, LBFGS 
import Optim

export roughness_penalty, regularizer, log_loss, unpack, b1_fit, complex_gaussian_noise, DAM

"""
Compute the roughness penalty for a 2d image using vectorized slice operations
to keep Zygote reverse-mode differentiation fast.
"""
function roughness_penalty(
        z::AbstractMatrix;
        nl_mls::Tuple = ((1, 0), (0, 1))
    )
    N, M = size(z)
    penalty = zero(real(eltype(z)))
    
    for offset in nl_mls
        n_l, m_l = offset[1], offset[2]
        abs_n, abs_m = abs(n_l), abs(m_l)

        # Slice views to compute 2*z[n,m] - z[n-n_l, m-m_l] - z[n+n_l, m+m_l] without loops
        center = @view z[1+abs_n:N-abs_n, 1+abs_m:M-abs_m]
        left   = @view z[1:N-2*abs_n,     1:M-2*abs_m]
        right  = @view z[1+2*abs_n:N,     1+2*abs_m:M]

        penalty += sum(abs2, 2 .* center .- left .- right)
    end

    return penalty
end

"""
Compute the total roughness penalty over each coil slice using sum generator syntax.
"""
function regularizer(zks::AbstractArray)
    K = size(zks, 3)
    return sum(roughness_penalty(@view zks[:, :, k]) for k in 1:K)
end

"""
Single-element evaluation helper for log-loss step. Fuses real/imag extractions.
"""
function l_equation_elem(y, f, chi_z_elem, F::Function)
    g_val = F(chi_z_elem)
    return -real(y) * f * real(g_val) - imag(y) * f * imag(g_val) + 0.5 * f^2 * abs2(g_val)
end

"""
Compute loss using direct matrix multiplication and fused elementwise operations.
"""
function log_loss(
        zks::AbstractArray, 
        fjs::AbstractArray, 
        Chi::AbstractArray,
        Y::AbstractArray,
        F::Function,
    )
    N, D, K = size(zks)
    M = size(Chi, 1)

    # Reshape and project via Chi
    Chi_z = reshape(reshape(zks, N * D, K) * Chi', N, D, M)

    # Fused calculation over broadcasted arrays
    l_array = l_equation_elem.(Y, fjs, Chi_z, F)
    return real(sum(l_array))
end

"""
Objective function for optimization combining log loss and regularization.
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

"""
Unpack optimized parameter vector into complex 3D and 2D arrays.
"""
function unpack(
        params::AbstractVector,
        zdims::Tuple,
        fdims::Tuple 
    )
    N, D, K = zdims
    N == fdims[1] || throw(ArgumentError("N's don't match."))
    D == fdims[2] || throw(ArgumentError("D's don't match."))

    fn = N * D
    zn = N * D * K

    # Direct vector slicing creates clean, Zygote-friendly adjoints
    zks_real = params[1:zn]
    zks_im   = params[zn+1 : 2*zn]
    zks      = reshape(complex.(zks_real, zks_im), zdims)

    fjs_real = params[2*zn+1 : 2*zn+fn]
    fjs_im   = params[2*zn+fn+1 : 2*zn+2*fn]
    fjs      = reshape(complex.(fjs_real, fjs_im), fdims)

    return zks, fjs
end

function H(z::AbstractArray, F::Function)
    return F.(z) ./ (exp.(im .* z) .+ 1e-12)
end

function compute_x_hat(
        Y::AbstractArray,
        K::Int,
        F::Function
    )
    T = eltype(Y)
    N, D, M = size(Y)

    x_hat_mags = zeros(T, N, D, K)
    for m in 1:K
        ratio = real(0.5 .* abs.(Y[:, :, m + K] ./ (Y[:, :, m] .+ 1e-12)))
        x_hat_mags[:, :, m] = acos.(clamp.(ratio, zero(real(T)), one(real(T)) - 1e-7))
    end

    x_hat_angles = zeros(eltype(Y), N, D, K)
    for m in 1:K
        x_hat_angles[:, :, m] = angle.(Y[:, :, m]) - angle.(H(abs.(x_hat_mags[:, :, K]), F))
    end

    return x_hat_mags .* exp.(im .* x_hat_angles)
end

function compute_composite_map(
        z_hat::AbstractArray,
        chi::AbstractArray
    )
    N, D, K = size(z_hat)
    M = size(chi, 1)
    K == size(chi, 2) || throw(ArgumentError("K's don't match."))

    x = zeros(eltype(z_hat), N, D, M)
    for k in 1:K
        z_flat = reshape(z_hat[:, :, k], :, 1)  # (N*D) x 1
        x_perm = chi[:, k] * z_flat'            # M x (N*D)
        x_perm = reshape(x_perm, M, N, D)       # M x N x D
        x += permutedims(x_perm, (2, 3, 1))      # N x D x M
    end

    return x
end

function compute_f_hat(
        Y::AbstractArray,
        z_hat::AbstractArray,
        chi::AbstractArray,
        F::Function
    )
    N, D, M = size(Y)
    xjm = compute_composite_map(z_hat, chi)

    top = zeros(eltype(Y), N, D)
    bottom = zeros(eltype(Y), N, D)

    for m in 1:M
        top += real.(conj.(Y[:, :, m]) .* F.(xjm[:, :, m]))
        bottom += abs.(F.(xjm[:, :, m])) .^ 2
    end

    return top ./ bottom
end

function b1_fit(
        Beta::Real,
        Y::AbstractArray,
        Chi::AbstractArray,
        F::Function
    )
    N, D, M = size(Y)
    K = size(Chi, 2) 
    zdims = (N, D, K)
    fdims = (N, D)

    # Error checking
    M == size(Chi, 1) || throw(ArgumentError("M's don't match."))
    M == 2 * K || throw(ArgumentError("Chi must be M x M/2."))
    
    half_M = div(M, 2)
    Chi_tilda = Chi[1:half_M, :]
    isapprox(Chi[half_M + 1:end, :], 2 * Chi_tilda) || throw(ArgumentError("The bottom half of Chi must be twice the top half of Chi."))
    
    # Initial guess estimation
    x_hat = compute_x_hat(Y, K, F)
    x_hat_perm = PermutedDimsArray(x_hat, (3, 1, 2))
    x_hat_flat = reshape(x_hat_perm, K, :)
    z_hat_flat = Chi_tilda \ x_hat_flat
    z_hat_perm = reshape(z_hat_flat, K, N, D)

    z_hat = permutedims(z_hat_perm, (2, 3, 1))
    f_hat = compute_f_hat(Y, z_hat, Chi, F)

    z_hat_flat = reshape(z_hat, :)
    f_hat_flat = reshape(f_hat, :)

    x0 = [real.(z_hat_flat); imag.(z_hat_flat); real.(f_hat_flat); imag.(f_hat_flat)]

    cost(x::AbstractVector) = psi(x, zdims, fdims, Beta, Y, Chi, F)
    options = Optim.Options(
        #iterations = 10,
        show_trace = true
    )
    
    println("Optimizing")
    out = Optim.optimize(cost, x0, LBFGS(), options; autodiff = AutoZygote())
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
    yj2 = fj .* sin.(2 .* aj) + std * randn(ComplexF32, (N, D))

    ydata = ones(ComplexF64, (N, D, 2))
    ydata[:, :, 1] = yj1
    ydata[:, :, 2] = yj2
    return ydata
end