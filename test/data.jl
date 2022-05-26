
@testset "Data Input" begin
    @testset "File Parsing from CSV" begin
        @testset "No Covariates" begin
            SigmaU, obj, X, T, Y = prepareData("$(prefix)test_data/NEEC_sampled.csv")
            @test typeof(X) == Nothing
            @test typeof(SigmaU) <: GPSLC.ConfounderStructure
            @test typeof(obj) <: GPSLC.ObjectLabels
            @test typeof(T) <: GPSLC.Treatment
            @test typeof(Y) <: GPSLC.Outcome
        end

        @testset "With Covariates" begin
            SigmaU, obj, X, T, Y = prepareData("$(prefix)test_data/additive_linear.csv")
            @test typeof(SigmaU) <: GPSLC.ConfounderStructure
            @test typeof(obj) <: GPSLC.ObjectLabels
            @test typeof(X) <: GPSLC.Covariates
            @test typeof(T) <: GPSLC.Treatment
            @test typeof(Y) <: GPSLC.Outcome
        end

        @testset "No ObjectLabels" begin
            SigmaU, obj, X, T, Y = prepareData("$(prefix)test_data/no_objects.csv")
            @test typeof(SigmaU) == Nothing
            @test typeof(obj) == Nothing
            @test typeof(X) <: GPSLC.Covariates
            @test typeof(T) <: GPSLC.Treatment
            @test typeof(Y) <: GPSLC.Outcome
        end

        @testset "No Confounders, No Covariates" begin
            SigmaU, obj, X, T, Y = prepareData("$(prefix)test_data/no_objects_no_cov.csv")
            @test typeof(SigmaU) == Nothing
            @test typeof(obj) == Nothing
            @test typeof(X) == Nothing
            @test typeof(T) <: GPSLC.Treatment
            @test typeof(Y) <: GPSLC.Outcome
        end
    end
end