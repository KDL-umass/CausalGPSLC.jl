@testset "Kernel Functions" begin
    @testset "rbfKernelLog" begin
        @testset "Int64" begin
            expected = [-(11 - 11)^2 / 0.1]
            actual = rbfKernelLog([11], [11], 0.1)
            @test expected ≈ actual
        end
        @testset "Float64" begin
            val = 11.1
            expected = [-(val - val)^2 / 0.1]
            actual = rbfKernelLog([val], [val], 0.1)
            @test expected ≈ actual
        end
        @testset "Bool true" begin
            val = true
            expected = [-(val - val)^2 / 0.1]
            actual = rbfKernelLog([val], [val], 0.1)
            @test expected ≈ actual
        end
        @testset "Bool false" begin
            val = false
            expected = [-(val - val)^2 / 0.1]
            actual = rbfKernelLog([val], [val], 0.1)
            @test expected ≈ actual
        end
        @testset "PersistentVector Bool true" begin
            val = true
            expected = [-(val - val)^2 / 0.1]
            actual = rbfKernelLog(FunctionalCollections.PersistentVector{}([val]), FunctionalCollections.PersistentVector{}([val]), 0.1)
            @test expected ≈ actual
        end
        @testset "PersistentVector Bool false" begin
            val = false
            expected = [-(val - val)^2 / 0.1]
            actual = rbfKernelLog(FunctionalCollections.PersistentVector{}([val]), FunctionalCollections.PersistentVector{}([val]), 0.1)
            @test expected ≈ actual
        end
    end
    @testset "processCov" begin

    end
    @testset "logit" begin
        @test logit(0.5) == 0
    end
    @testset "expit" begin
        @test expit(0) == 0.5
    end
end