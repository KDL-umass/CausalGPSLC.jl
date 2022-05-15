@testset "gpslc" begin
    @testset "NEEC" begin
        expected = CSV.read("$(prefix)test_results/NEEC_sampled_0.6.csv", DataFrame)
        g = gpslc("$(prefix)test_data/NEEC_sampled.csv")
        ITEsamples = sampleITE(g; doT=0.6)
        actual = summarizeITE(ITEsamples; savetofile="tmp.csv")
        @test countCloseEnough(expected, actual) >= 0.85
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