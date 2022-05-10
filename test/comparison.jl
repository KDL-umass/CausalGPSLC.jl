function testInference(fname, doT)
    g = gpslc("$(prefix)test_data/" * fname * ".csv")
    ITEsamples = sampleITE(g; doT=doT)
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

print("[Running Comparison Tests]")

@testset "Comparison Tests" begin
    confidence = 0.93
    @testset "Additive Linear" begin
        @testset "AddLinear 0.0" begin
            expected = CSV.read("$(prefix)test_results/additive_linear_0.csv", DataFrame)
            actual = testInference("additive_linear", 0.0)
            @test countCloseEnough(expected, actual) >= confidence
        end
        @testset "AddLinear 1.0" begin
            expected = CSV.read("$(prefix)test_results/additive_linear_1.csv", DataFrame)
            actual = testInference("additive_linear", 1.0)
            @test countCloseEnough(expected, actual) >= confidence
        end
    end
    @testset "Additive Nonlinear" begin
        @testset "AddNonlinear 0.0" begin
            expected = CSV.read("$(prefix)test_results/additive_nonlinear_0.csv", DataFrame)
            actual = testInference("additive_nonlinear", 0.0)
            @test countCloseEnough(expected, actual) >= confidence
        end
        @testset "AddNonlinear 1.0" begin
            expected = CSV.read("$(prefix)test_results/additive_nonlinear_1.csv", DataFrame)
            actual = testInference("additive_nonlinear", 1.0)
            @test countCloseEnough(expected, actual) >= confidence
        end
    end
    @testset "Multiplicative Linear" begin
        @testset "MultiLinear 0.0" begin
            expected = CSV.read("$(prefix)test_results/multiplicative_linear_0.csv", DataFrame)
            actual = testInference("multiplicative_linear", 0.0)
            @test countCloseEnough(expected, actual) >= confidence
        end
        @testset "MultiLinear 1.0" begin
            expected = CSV.read("$(prefix)test_results/multiplicative_linear_1.csv", DataFrame)
            actual = testInference("multiplicative_linear", 1.0)
            @test countCloseEnough(expected, actual) >= confidence
        end
    end
    @testset "Multiplicative Nonlinear" begin
        @testset "MultiNonLinear 0.0" begin
            expected = CSV.read("$(prefix)test_results/multiplicative_nonlinear_0.csv", DataFrame)
            actual = testInference("multiplicative_nonlinear", 0.0)
            @test countCloseEnough(expected, actual) >= confidence
        end
        @testset "MultiNonLinear 1.0" begin
            expected = CSV.read("$(prefix)test_results/multiplicative_nonlinear_1.csv", DataFrame)
            actual = testInference("multiplicative_nonlinear", 1.0)
            @test countCloseEnough(expected, actual) >= confidence
        end
    end
    @testset "IHDP" begin
        @testset "IHDP false" begin
            expected = CSV.read("$(prefix)test_results/IHDP_sampled_false.csv", DataFrame)
            actual = testInference("IHDP_sampled", false)
            @test countCloseEnough(expected, actual) >= confidence
        end
        @testset "IHDP true" begin
            expected = CSV.read("$(prefix)test_results/IHDP_sampled_true.csv", DataFrame)
            actual = testInference("IHDP_sampled", true)
            @test countCloseEnough(expected, actual) >= confidence
        end
    end
    @testset "NEEC" begin
        @testset "NEEC 0.0" begin
            expected = CSV.read("$(prefix)test_results/NEEC_sampled_0.csv", DataFrame)
            actual = testInference("NEEC_sampled", 0.0)
            @test countCloseEnough(expected, actual) >= confidence
        end
        @testset "NEEC 0.6" begin
            expected = CSV.read("$(prefix)test_results/NEEC_sampled_0.6.csv", DataFrame)
            actual = testInference("NEEC_sampled", 0.6)
            @test countCloseEnough(expected, actual) >= confidence
        end
        @testset "NEEC 1.0" begin
            expected = CSV.read("$(prefix)test_results/NEEC_sampled_1.csv", DataFrame)
            actual = testInference("NEEC_sampled", 1.0)
            @test countCloseEnough(expected, actual) >= confidence
        end
    end
end