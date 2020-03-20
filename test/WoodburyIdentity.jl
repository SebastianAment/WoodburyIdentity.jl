module TestWoodburyIdentity
using WoodburyIdentity
using WoodburyIdentity: Woodbury, eltype, issymmetric, ishermitian
using WoodburyIdentity: inverse, Inverse
using LinearAlgebra
using Test

function getW(n, m; diagonal = true, symmetric = false)
    A = diagonal ? Diagonal(randn(n)) : randn(n, n)
    A = symmetric ? A'A : A
    U = randn(n, m)
    C = diagonal ? Diagonal(randn(m)) : randn(m, m)
    C = symmetric ? C'C : C
    V = symmetric ? U' : randn(m, n)
    W = Woodbury(A, U, C, V)
end

@testset "woodbury" begin
    n = 3
    m = 2
    W = getW(n, m, symmetric = true)
    MatW = Matrix(W)
    @test size(W) == (n, n)
    x = randn(size(W, 2))
    @test W*x ≈ MatW*x
    @test x'W ≈ x'MatW
    @test dot(x, W, x) ≈ dot(x, MatW*x)
    @test eltype(W) == Float64
    @test issymmetric(W)
    @test ishermitian(W)
    @test ishermitian(Woodbury(W.A, W.U, W.C, copy(W.U)')) # testing edge case
    @test !issymmetric(getW(n, m))
    @test !ishermitian(getW(n, m))
    @test det(W) ≈ det(MatW)
    @test logdet(W) ≈ logdet(MatW)

    # test solves
    @test W\(W*x) ≈ x
    @test (x'W)/W ≈ x'

    # factorization
    n = 1024
    m = 3
    W = getW(n, m, symmetric = true)
    F = factorize(W)
    x = randn(n)
    @test W \ x ≈ F \ x
end

end # TestWoodburyIdentity
