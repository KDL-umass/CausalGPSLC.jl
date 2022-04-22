"""
Test to see if all the latent variables we are doing inference over
are changing from iteration to iteration.

This is a simple check to ensure that not all the iterations have the same value.

Observed variables (X,T,Y) are ignored.
"""
function testLatentVariablesChanging(posteriorSamples)
    @testset "Latent variables changing during inference" begin
        zip = Dict()
        for i in 1:length(posteriorSamples)
            for (addr, val) in get_values_shallow(posteriorSamples[i])
                if addr in keys(zip)
                    push!(zip[addr], val)
                else
                    zip[addr] = [val]
                end
            end
        end

        for addr in keys(zip)
            if !(addr in [:X, :T, :logitT, :Y]) # observed and don't change
                @test length(unique(zip[addr])) > 1
            end
        end
    end
end


@testset "Inference" begin
    hyperparams, n, nU, nX, X, binaryT, realT = getToyData(10, 2, 8)
    nOuter = 10
    nMHInner = 7
    nESInner = 6

    @testset "Binary Treatment, No U, No Cov" begin
        _, Y = getToyObservations(n)
        posteriorSamples, trace = Posterior(hyperparams, nothing, binaryT, Y, nothing, nOuter, nothing, nothing)
        testLatentVariablesChanging(posteriorSamples)
        @test true
    end

    @testset "Binary Treatment, No Cov" begin
        _, Y = getToyObservations(n)
        posteriorSamples, trace = Posterior(hyperparams, nothing, binaryT, Y, nU, nOuter, nMHInner, nESInner)
        testLatentVariablesChanging(posteriorSamples)
        @test true
    end

    @testset "Binary Treatment, No U" begin
        _, Y = getToyObservations(n)
        posteriorSamples, trace = Posterior(hyperparams, X, binaryT, Y, nothing, nOuter, nMHInner, nESInner)
        @test true
    end

    @testset "Binary Treatment" begin
        _, Y = getToyObservations(n)
        posteriorSamples, trace = Posterior(hyperparams, X, binaryT, Y, nU, nOuter, nMHInner, nESInner)
        @test true
    end

    @testset "Continuous Treatment, No U, No Cov" begin
        _, Y = getToyObservations(n)
        posteriorSamples, trace = Posterior(hyperparams, nothing, realT, Y, nothing, nOuter, nothing, nothing)
        @test true
    end

    @testset "Continuous Treatment, No Cov" begin
        _, Y = getToyObservations(n)
        posteriorSamples, trace = Posterior(hyperparams, nothing, realT, Y, nU, nOuter, nMHInner, nESInner)
        @test true
    end

    @testset "Continuous Treatment, No U" begin
        _, Y = getToyObservations(n)
        posteriorSamples, trace = Posterior(hyperparams, X, realT, Y, nothing, nOuter, nMHInner, nESInner)
        @test true
    end

    @testset "Continuous Treatment" begin
        _, Y = getToyObservations(n)
        posteriorSamples, trace = Posterior(hyperparams, X, realT, Y, nU, nOuter, nMHInner, nESInner)
        @test true
    end

end
