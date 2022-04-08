function testInference()
    X, T, Y, SigmaU = prepareData("$(prefix)test_data/NEEC_sampled.csv")

    posteriorSample = samplePosterior(X, T, Y, SigmaU)
    ITEsamples = sampleITE(X, T, Y, SigmaU;
        posteriorSample=posteriorSample)
    summarizeITE(ITEsamples)
end

@testset "Submission Comparison" begin
    @testset "NEEC" begin
        expected = CSV.read("$(prefix)test_results/NEEC_sampled_80.csv", DataFrame)
        actual = testInference()

        @test size(expected) == size(actual)
        N = size(actual, 1)
        passing = zeros(N)
        for i = 1:N
            passing[i] = expected[i, "LowerBound"] <= actual[i, "Mean"] &&
                         actual[i, "Mean"] <= expected[i, "UpperBound"]
        end
        @test sum(passing) / N >= 0.93
    end
end

@testset "Evaluation Comparison" begin
    # compare avg actual mean to avg original bounds
    # analysis/eval.jl
end