using DataFrames
import CSV

include("../examples/basicExample.jl")

@testset "Submission Comparison" begin
    @testset "NEEC" begin
        basicExample()
        expected = CSV.read("test/test_results/NEEC_sampled_80.csv", DataFrame)
        actual = CSV.read("examples/results/NEEC_sampled_80.csv", DataFrame)

        @test size(expected) == size(actual)
        N = size(actual, 1)
        passing = zeros(N)
        for i = 1:N
            passing[i] = expected[i, "LowerBound"] <= actual[i, "Mean"] &&
                         actual[i, "Mean"] <= expected[i, "UpperBound"]
        end
        @test sum(passing) / N > 0.97
    end
end

@testset "Evaluation Comparison" begin
    # compare avg actual mean to avg original bounds
    # analysis/eval.jl
end