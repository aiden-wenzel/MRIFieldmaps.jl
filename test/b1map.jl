# test/b1map.jl

using MRIFieldmaps: roughness_penalty, regularizer
using Test: @test, @testset, @test_throws, @inferred

@testset "b1map.jl" begin
    # We should expect no roughness since the matrix is full of the same values.
    z1 = ones(3, 3) * 5.0
    @test isapprox(roughness_penalty(z1), 0.0) 

    # Test with the middle pixel being different.
    z2 = zeros(3, 3)
    z2[2, 2] = 10.0
    expected = (-10)^2 + (-10)^2 + (2*10)^2 + (-10)^2 + (-10)^2;
    @test isapprox(roughness_penalty(z2), expected)

    # See if the regularizer function can sum up the roughness costs from z1 and z2.
    zdims = (3, 3, 1)
    zks = cat(reshape(z1, zdims), reshape(z2, zdims), dims=3)
    @test isapprox(regularizer(zks), expected)

    # Simulate data from paper

    K = 1
    M = 2
    Chi::Matrix = zeros((M, K))
    Chi[1, 1] = 1
    Chi[2, 1] = 2

    F = sin

    zdims = (N, D, K) # TODO: What are N and D?
    fdims = (N, D)

    Beta = 0.7
    z_hat, f_hat = b1_fit(zdims, fdims, Beta, Y, Chi, F)
end
