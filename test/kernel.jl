@testset "Kernel Functions" begin
    @testset "rbfKernelLog" begin
        @testset "X1::SupportedRBFMatrix, X2::SupportedRBFMatrix, LS::Float64" begin

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