function comparePredictedBinary(predictedTreatment, binaryT, n)
    avg = mean(predictedTreatment, dims=2)
    avg = reshape(avg, (n,))
    rounded = round.(avg)
    booled = convert.(Bool, rounded)
    return booled .== binaryT
end

function approximatelyEqual(expected, actual)
    sd = Statistics.std(actual)
    nSTD = 2
    (actual .- nSTD * sd .<= expected) .& (expected .<= actual .+ nSTD * sd)
end

function comparePredictedReal(predictedTreatment, realT, n)
    avg = mean(predictedTreatment, dims=2)
    avg = reshape(avg, (n,))
    return approximatelyEqual(avg, realT)
end

function generateTreatment(model, sampleChoiceMap, priorparams, n, nU, nX)
    selectT = Gen.select(:T)
    notT = Gen.complement(selectT)
    obs = Gen.get_selected(sampleChoiceMap, notT)
    trace, _ = Gen.generate(model, (priorparams, n, nU, nX), obs)
    return get_choices(trace)
end

percentClose = 0.5

@testset "Posterior Predictive Checks" begin
    priorparams, n, nU, nX, X, binaryT, realT = getToyData(20, 2, 5)
    nOuter = 10
    nMHInner = 10
    nESInner = 15

    @testset "Binary Treatment, No U, No Cov" begin
        _, Y = getToyObservations(n)
        posteriorSamples, trace = Posterior(priorparams, nothing, binaryT, Y, nothing, nOuter, nothing, nothing)

        predictedTreatment = zeros(n, length(posteriorSamples))
        for (i, sampleChoiceMap) in enumerate(posteriorSamples)
            choicemap = generateTreatment(GPSLCNoUNoCovBinaryT, sampleChoiceMap, priorparams, n, nothing, nothing)
            predictedTreatment[:, i] = [choicemap[:T=>i=>:T] for i in 1:n]
        end
        comparison = comparePredictedBinary(predictedTreatment, binaryT, n)
        @test sum(comparison) >= percentClose * n
    end

    @testset "Binary Treatment, No Cov" begin
        _, Y = getToyObservations(n)
        posteriorSamples, trace = Posterior(priorparams, nothing, binaryT, Y, nU, nOuter, nMHInner, nESInner)

        predictedTreatment = zeros(n, length(posteriorSamples))
        for (i, sampleChoiceMap) in enumerate(posteriorSamples)
            choicemap = generateTreatment(GPSLCNoCovBinaryT, sampleChoiceMap, priorparams, n, nU, nothing)
            predictedTreatment[:, i] = [choicemap[:T=>i=>:T] for i in 1:n]
        end
        comparison = comparePredictedBinary(predictedTreatment, binaryT, n)
        @test sum(comparison) >= percentClose * n
    end

    @testset "Binary Treatment, No U" begin
        _, Y = getToyObservations(n)
        posteriorSamples, trace = Posterior(priorparams, X, binaryT, Y, nothing, nOuter, nMHInner, nESInner)

        predictedTreatment = zeros(n, length(posteriorSamples))
        for (i, sampleChoiceMap) in enumerate(posteriorSamples)
            choicemap = generateTreatment(GPSLCNoUBinaryT, sampleChoiceMap, priorparams, n, nothing, nX)
            predictedTreatment[:, i] = [choicemap[:T=>i=>:T] for i in 1:n]
        end
        comparison = comparePredictedBinary(predictedTreatment, binaryT, n)
        @test sum(comparison) >= percentClose * n
    end

    @testset "Binary Treatment" begin
        _, Y = getToyObservations(n)
        posteriorSamples, trace = Posterior(priorparams, X, binaryT, Y, nU, nOuter, nMHInner, nESInner)

        predictedTreatment = zeros(n, length(posteriorSamples))
        for (i, sampleChoiceMap) in enumerate(posteriorSamples)
            choicemap = generateTreatment(GPSLCBinaryT, sampleChoiceMap, priorparams, n, nU, nX)
            predictedTreatment[:, i] = [choicemap[:T=>i=>:T] for i in 1:n]
        end
        comparison = comparePredictedBinary(predictedTreatment, binaryT, n)
        @test sum(comparison) >= percentClose * n
    end

    @testset "Real Treatment, No U, No Cov" begin
        _, Y = getToyObservations(n)
        posteriorSamples, trace = Posterior(priorparams, nothing, realT, Y, nothing, nOuter, nothing, nothing)

        predictedTreatment = zeros(n, length(posteriorSamples))
        for (i, sampleChoiceMap) in enumerate(posteriorSamples)
            choicemap = generateTreatment(GPSLCNoUNoCovRealT, sampleChoiceMap, priorparams, n, nothing, nothing)
            predictedTreatment[:, i] = choicemap[:T]
        end
        comparison = comparePredictedReal(predictedTreatment, realT, n)
        @test sum(comparison) >= percentClose * n
    end

    @testset "Real Treatment, No Cov" begin
        _, Y = getToyObservations(n)
        posteriorSamples, trace = Posterior(priorparams, nothing, realT, Y, nU, nOuter, nMHInner, nESInner)

        predictedTreatment = zeros(n, length(posteriorSamples))
        for (i, sampleChoiceMap) in enumerate(posteriorSamples)
            choicemap = generateTreatment(GPSLCNoCovRealT, sampleChoiceMap, priorparams, n, nU, nothing)
            predictedTreatment[:, i] = choicemap[:T]
        end
        comparison = comparePredictedReal(predictedTreatment, realT, n)
        @test sum(comparison) >= percentClose * n
    end

    @testset "Real Treatment, No U" begin
        _, Y = getToyObservations(n)
        posteriorSamples, trace = Posterior(priorparams, X, realT, Y, nothing, nOuter, nMHInner, nESInner)

        predictedTreatment = zeros(n, length(posteriorSamples))
        for (i, sampleChoiceMap) in enumerate(posteriorSamples)
            choicemap = generateTreatment(GPSLCNoURealT, sampleChoiceMap, priorparams, n, nothing, nX)
            predictedTreatment[:, i] = choicemap[:T]
        end
        comparison = comparePredictedReal(predictedTreatment, realT, n)
        @test sum(comparison) >= percentClose * n
    end

    @testset "Real Treatment" begin
        _, Y = getToyObservations(n)
        posteriorSamples, trace = Posterior(priorparams, X, realT, Y, nU, nOuter, nMHInner, nESInner)

        predictedTreatment = zeros(n, length(posteriorSamples))
        for (i, sampleChoiceMap) in enumerate(posteriorSamples)
            choicemap = generateTreatment(GPSLCRealT, sampleChoiceMap, priorparams, n, nU, nX)
            predictedTreatment[:, i] = choicemap[:T]
        end
        comparison = comparePredictedReal(predictedTreatment, realT, n)
        @test sum(comparison) >= percentClose * n
    end
end