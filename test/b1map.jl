# test/b1map.jl

using MRIFieldmaps: roughness_penalty, regularizer, unpack
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

    # Test unpack
    N = 3
    D = 4
    K = 2
    zdims = (N, D, K)
    fdims = (N, D)
    params = Vector(1:N*D*K + N*D)
    zks, fjs = unpack(params, zdims, fdims)
    # We would expect the zks to contina the first N*D*K elements of params.
    coil_1::Array = [
        1 4 7 10;
        2 5 8 11;
        3 6 9 12;
    ]

    coil_2::Array = [
        13 16 19 22;
        14 17 20 23;
        15 18 21 24;
    ]

    expected_zks = cat(reshape(coil_1, (N, D, 1)), reshape(coil_2, (N, D, 1)), dims=3)
    @test size(expected_zks) == size(zks)
    @test expected_zks == zks

    expected_fjs::Array = [
        25 28 31 34;
        26 29 32 35;
        27 30 33 36;
    ]

    @test size(expected_fjs) == size(fjs)
    @test expected_fjs == fjs

    # Simulate data from paper
    """
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
    """
end
