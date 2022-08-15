
@testset "Data Input" begin
    @testset "File Parsing from CSV" begin
        @testset "No Covariates" begin
            SigmaU, obj, X, T, Y = prepareData("$(prefix)test_data/NEEC_sampled.csv")
            @test typeof(X) == Nothing
            @test typeof(SigmaU) <: CausalGPSLC.ConfounderStructure
            @test typeof(obj) <: CausalGPSLC.ObjectLabels
            @test typeof(T) <: CausalGPSLC.Treatment
            @test typeof(Y) <: CausalGPSLC.Outcome
        end

        @testset "With Covariates" begin
            SigmaU, obj, X, T, Y = prepareData("$(prefix)test_data/additive_linear.csv")
            @test typeof(SigmaU) <: CausalGPSLC.ConfounderStructure
            @test typeof(obj) <: CausalGPSLC.ObjectLabels
            @test typeof(X) <: CausalGPSLC.Covariates
            @test typeof(T) <: CausalGPSLC.Treatment
            @test typeof(Y) <: CausalGPSLC.Outcome
        end

        @testset "No ObjectLabels" begin
            SigmaU, obj, X, T, Y = prepareData("$(prefix)test_data/no_objects.csv")
            @test typeof(SigmaU) == Nothing
            @test typeof(obj) == Nothing
            @test typeof(X) <: CausalGPSLC.Covariates
            @test typeof(T) <: CausalGPSLC.Treatment
            @test typeof(Y) <: CausalGPSLC.Outcome
        end

        @testset "No Confounders, No Covariates" begin
            SigmaU, obj, X, T, Y = prepareData("$(prefix)test_data/no_objects_no_cov.csv")
            @test typeof(SigmaU) == Nothing
            @test typeof(obj) == Nothing
            @test typeof(X) == Nothing
            @test typeof(T) <: CausalGPSLC.Treatment
            @test typeof(Y) <: CausalGPSLC.Outcome
        end
    end
end