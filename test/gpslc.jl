@testset "gpslc accepts each variety of input" begin
    hyperparams = getHyperParameters()
    hyperparams.nOuter = 5
    hyperparams.nMHInner = 1
    hyperparams.nESInner = 1
    @testset "Full model" begin
        gpslc("$(prefix)test_data/minimal.csv"; hyperparams=hyperparams)
        @test true
    end
    @testset "No Cov" begin
        gpslc("$(prefix)test_data/no_cov.csv"; hyperparams=hyperparams)
        @test true
    end
    @testset "No U" begin
        gpslc("$(prefix)test_data/no_objects.csv"; hyperparams=hyperparams)
        @test true
    end
    @testset "No Cov, No U" begin
        gpslc("$(prefix)test_data/no_objects_no_cov.csv"; hyperparams=hyperparams)
        @test true
    end
end
