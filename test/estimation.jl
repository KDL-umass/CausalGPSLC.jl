function testInference(fname, doT)
    X, T, Y, SigmaU = prepareData("$(prefix)test_data/$fname.csv")
    posteriorSample = samplePosterior(X, T, Y, SigmaU)
    ITEsamples = sampleITE(X, T, Y, SigmaU;
        posteriorSample=posteriorSample, doT=doT)
    summarizeITE(ITEsamples)
end


"""Count how many of the actual mean values are within the expected bounds"""
function countCloseEnough(expected, actual)
    @assert size(expected) == size(actual) "expected and actual aren't same size"
    n = size(actual, 1)
    passing = zeros(n)
    for i = 1:n
        passing[i] = expected[i, "LowerBound"] <= actual[i, "Mean"] &&
                     actual[i, "Mean"] <= expected[i, "UpperBound"]
    end
    return sum(passing) / n
end


@testset "Estimation Comparison" begin
    @testset "Additive Linear" begin
        @testset "0.0" begin
            expected = CSV.read("$(prefix)test_results/additive_linear_0.csv", DataFrame)
            actual = testInference("additive_linear", 0.0)
            @test countCloseEnough(expected, actual) >= 0.93
        end
        @testset "1.0" begin
            expected = CSV.read("$(prefix)test_results/additive_linear_1.csv", DataFrame)
            actual = testInference("additive_linear", 1.0)
            @test countCloseEnough(expected, actual) >= 0.93
        end
    end
    @testset "Additive Nonlinear" begin
        @testset "0.0" begin
            expected = CSV.read("$(prefix)test_results/additive_nonlinear_0.csv", DataFrame)
            actual = testInference("additive_nonlinear", 0.0)
            @test countCloseEnough(expected, actual) >= 0.93
        end
        @testset "1.0" begin
            expected = CSV.read("$(prefix)test_results/additive_nonlinear_1.csv", DataFrame)
            actual = testInference("additive_nonlinear", 1.0)
            @test countCloseEnough(expected, actual) >= 0.93
        end
    end
    @testset "Multiplicative Linear" begin
        @testset "0.0" begin
            expected = CSV.read("$(prefix)test_results/multiplicative_linear_0.csv", DataFrame)
            actual = testInference("multiplicative_linear", 0.0)
            @test countCloseEnough(expected, actual) >= 0.93
        end
        @testset "1.0" begin
            expected = CSV.read("$(prefix)test_results/multiplicative_linear_1.csv", DataFrame)
            actual = testInference("multiplicative_linear", 1.0)
            @test countCloseEnough(expected, actual) >= 0.93
        end
    end
    @testset "Multiplicative Nonlinear" begin
        @testset "0.0" begin
            expected = CSV.read("$(prefix)test_results/multiplicative_nonlinear_0", DataFrame)
            actual = testInference("multiplicative_nonlinear", 0.0)
            @test countCloseEnough(expected, actual) >= 0.93
        end
        @testset "1.0" begin
            expected = CSV.read("$(prefix)test_results/multiplicative_nonlinear_1", DataFrame)
            actual = testInference("multiplicative_nonlinear", 1.0)
            @test countCloseEnough(expected, actual) >= 0.93
        end
    end
    @testset "IHDP" begin
        @testset "false" begin
            expected = CSV.read("$(prefix)test_results/IHDP_sampled_false.csv", DataFrame)
            actual = testInference("IHDP_sampled", false)
            @test countCloseEnough(expected, actual) >= 0.93
        end
        @testset "true" begin
            expected = CSV.read("$(prefix)test_results/IHDP_sampled_true.csv", DataFrame)
            actual = testInference("IHDP_sampled", true)
            @test countCloseEnough(expected, actual) >= 0.93
        end
    end
    @testset "NEEC" begin
        @testset "0.0" begin
            expected = CSV.read("$(prefix)test_results/NEEC_sampled_0", DataFrame)
            actual = testInference("NEEC_sampled", 0.0)
            @test countCloseEnough(expected, actual) >= 0.93
        end
        @testset "1.0" begin
            expected = CSV.read("$(prefix)test_results/NEEC_sampled_1", DataFrame)
            actual = testInference("NEEC_sampled", 1.0)
            @test countCloseEnough(expected, actual) >= 0.93
        end
    end
end

@testset "Submission Comparison" begin
    @testset "NEEC" begin
        expected = CSV.read("$(prefix)test_results/NEEC_sampled_0.6.csv", DataFrame)
        actual = testInference("NEEC_sampled", 0.6)
        @test countCloseEnough(expected, actual) >= 0.93
    end
end

@testset "Evaluation Comparison" begin
    # compare avg actual mean to avg original bounds
    # analysis/eval.jl
end