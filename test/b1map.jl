# test/b1map.jl

using MRIFieldmaps: roughness_penalty, regularizer
using Test: @test, @testset, @test_throws, @inferred

@testset "b1map.jl" begin
    z1 = ones(3, 3) * 5.0
    @test isapprox(roughness_penalty(z1), 0.0)

    z2 = zeros(3, 3)
    z2[2, 2] = 10.0
    expected = (-10)^2 + (-10)^2 + (2*10)^2 + (-10)^2 + (-10)^2;
    
    @test isapprox(roughness_penalty(z2), expected)
    zks = [z1, z2]
    @test isapprox(regularizer(zks), expected)
end
