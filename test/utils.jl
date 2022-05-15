@testset "Utilities" begin
    @testset "generateSigmaU" begin
        nIndividualsArray = [2, 3]
        cov = 2.0
        eps = 0.1
        diag = 1 + eps
        expected = [
            diag cov 0 0 0
            cov diag 0 0 0
            0 0 diag cov cov
            0 0 cov diag cov
            0 0 cov cov diag
        ]
        actual = generateSigmaU(nIndividualsArray, eps, cov)
        @test expected == actual
    end
    @testset "removeAdjacent" begin
        input = [1, 2, 2, 3, 4, 4, 5, 3, 4]
        expected = [1, 2, 3, 4, 5, 3, 4]
        actual = removeAdjacent(input)
        @test expected ≈ actual
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
    @testset "toTupleOfVectors" begin
        data = ones(3, 2)
        expected = ([1, 1], [1, 1], [1, 1])
        actual = toTupleOfVectors(data)
        @test expected == actual
    end

    @testset "getChoiceAddresses" begin
        choices = choicemap()
        choices[:T] = 1.0
        actual = getAddresses(choices)
        expected = [:T]
        @test expected == actual
    end
end