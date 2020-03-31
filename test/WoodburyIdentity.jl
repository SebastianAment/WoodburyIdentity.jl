module TestWoodburyIdentity
using WoodburyIdentity
using WoodburyIdentity: Woodbury, eltype, issymmetric, ishermitian, factorize_logdet
using LazyInverse: inverse, Inverse
using LinearAlgebra
using Test

function getW(n, m; diagonal = true, symmetric = false)
    A = diagonal ? Diagonal(exp.(.1randn(n))) + I(n) : randn(n, n)
    A = symmetric && !diagonal ? A'A : A
    U = randn(n, m)
    C = diagonal ? Diagonal(exp.(.1randn(m))) + I(m) : randn(m, m)
    C = symmetric && !diagonal ? C'C : C
    V = symmetric ? U' : randn(m, n)
    W = Woodbury(A, U, C, V)
end

@testset "woodbury" begin
    n = 3
    m = 2
    W = getW(n, m, symmetric = true)
    MW = Matrix(W)
    @test size(W) == (n, n)
    x = randn(size(W, 2))
    @test W*x ≈ MW*x
    @test x'W ≈ x'MW
    @test dot(x, W, x) ≈ dot(x, MW*x)
    @test eltype(W) == Float64
    @test issymmetric(W)
    @test ishermitian(W)
    @test ishermitian(Woodbury(W.A, W.U, W.C, copy(W.U)')) # testing edge case
    @test !issymmetric(getW(n, m))
    @test !ishermitian(getW(n, m))

    # test solves
    @test W\(W*x) ≈ x
    @test (x'W)/W ≈ x'

    # factorization
    n = 1024
    m = 3
    W = getW(n, m, symmetric = true)
    MW = Matrix(W)
    F = factorize(W)
    @test F isa Inverse
    x = randn(n)
    @test W \ x ≈ F \ x
    @test MW \ x ≈ F \ x

    n = 4
    m = 3
    W = getW(n, m, symmetric = true)
    MW = Matrix(W)

    ## determinant
    @test det(W) ≈ det(MW)
    @test logdet(W) ≈ logdet(MW)
    @test det(inverse(W)) ≈ det(Inverse(W))
    @test det(inverse(W)) ≈ det(inv(MW))
    @test logdet(inverse(W)) ≈ logdet(Inverse(W))
    @test logdet(inverse(W)) ≈ logdet(inv(MW))
    @test all(logabsdet(inverse(W)) .≈ logabsdet(Inverse(W)))
    @test all(logabsdet(inverse(W)) .≈ logabsdet(inv(MW)))

    # indexing
    for i in 1:n, j in 1:n
        @test W[i, j] ≈ MW[i, j]
    end

    # trace
    @test tr(W) ≈ tr(MW)

    # factorize
    F = factorize(W)
    @test Matrix(inverse(F)) ≈ inv(MW)
    @test logdet(F) ≈ logdet(MW)

    # factorizing and logdet
    F, l = factorize_logdet(W)
    @test l ≈ logdet(MW)
    @test l ≈ logdet(F)
    @test Matrix(inverse(F)) ≈ inv(MW)
end

end # TestWoodburyIdentity
