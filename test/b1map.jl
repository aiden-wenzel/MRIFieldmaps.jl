# test/b1map.jl

using MRIFieldmaps: roughness_penalty, regularizer, L
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
    zks = cat(reshape(z1, 1, 3, 3), reshape(z2, 1, 3, 3), dims=1)
    @test isapprox(regularizer(zks), expected)
end
