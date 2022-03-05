@testset "Utilities" begin
    @testset "generateSigmaU" begin

    end
    @testset "removeAdjacent" begin
        input = [1, 2, 2, 3, 4, 4, 5, 3, 4]
        expected = [1, 2, 3, 4, 5, 3, 4]
        actual = removeAdjacent(input)
        @test expected == actual
    end
end