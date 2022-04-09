@testset "Utilities" begin
    @testset "generateSigmaU" begin

    end
    @testset "removeAdjacent" begin
        input = [1, 2, 2, 3, 4, 4, 5, 3, 4]
        expected = [1, 2, 3, 4, 5, 3, 4]
        actual = removeAdjacent(input)
        @test expected == actual
    end
    @testset "toMatrix" begin
        @testset "fill" begin
            U = fill(rand(5), 10)
            U = toMatrix(U, 10, 5)
            @test size(U) == (10, 5)
        end
        @testset "listcomp" begin
            U = [rand(5) for i = 1:10]
            U = toMatrix(U, 10, 5)
            @test size(U) == (10, 5)
        end
    end
end