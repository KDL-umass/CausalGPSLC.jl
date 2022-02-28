using Pkg
Pkg.activate(".")

using GPSLC
using Test
using CSV
using DataFrames
using Random

Random.seed!(0)

include("../examples/basicExample.jl")

@testset "GPSLC.jl" begin
    # Is inference correct?

    # Is the algorithm doing the preconditions for inference?
    # (Are the individual mechanisms working properly)
    ## During inference are you changing each latent variable in your model? 

    # SBC: Run univariate sbc for each dimension
    # Apply Bonferroni correction to adjust p-value (multiple hypothesis correction)


    ### Test kernel functions

    ### Test utils

    ### Make sure things match submission
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
            @test sum(passing) / N > 0.95
        end
    end

    @testset "Evaluation Comparison" begin
        # compare avg actual mean to avg original bounds
        # analysis/eval.jl
    end
end
