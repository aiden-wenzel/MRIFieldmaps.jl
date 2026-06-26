# test/b1map.jl

using MRIFieldmaps: roughness_penalty
using Test: @test, @testset, @test_throws, @inferred

@testset "b1map.jl" begin
    z = ones(3, 3) * 5.0
    @test roughness_penalty(z) == 0.0
end