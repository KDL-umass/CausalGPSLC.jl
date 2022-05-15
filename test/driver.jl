@testset "gpslc" begin
    @testset "NEEC 0.6" begin
        expected = CSV.read("$(prefix)test_results/NEEC_sampled_0.csv", DataFrame)
        g = gpslc("$(prefix)test_data/NEEC_sampled.csv")
        ITEsamples = sampleITE(g; doT=0.0)
        actual = summarizeITE(ITEsamples)
        @test countCloseEnough(expected, actual) >= 0.93
    end
end


@testset "ITEsamples" begin

end

@testset "SATEsamples" begin

end

@testset "sampleITE" begin

end

@testset "SummarizeITE" begin

end