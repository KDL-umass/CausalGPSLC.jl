@testset "gpslc" begin
    @testset "Full model" begin
        gpslc("$(prefix)test_data/additive_nonlinear.csv")
    end
    @testset "No Cov" begin
        gpslc("$(prefix)test_data/NEEC_sampled.csv")
    end
    @testset "No U" begin
        gpslc("$(prefix)test_data/no_objects.csv")
    end
    @testset "No Cov, No U" begin
        gpslc("$(prefix)test_data/no_objects_no_cov.csv")
    end
end
