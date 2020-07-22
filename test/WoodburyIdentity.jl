module TestWoodburyIdentity
using WoodburyIdentity
using WoodburyIdentity: Woodbury, eltype, issymmetric, ishermitian
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

@testset "Woodbury" begin
    n = 3
    m = 2
    W = getW(n, m, symmetric = true)
    MW = Matrix(W)
    @test size(W) == (n, n)
    x = randn(size(W, 2))
    # multiplication
    @test W*x ≈ MW*x
    @test x'W ≈ x'MW
    @test dot(x, W, x) ≈ dot(x, MW*x)
    X = randn(size(W, 2), 3)
    @test W*X ≈ MW*X
    @test X'*W ≈ X'*MW

    @test eltype(W) == Float64
    @test issymmetric(W)
    @test ishermitian(W)
    @test ishermitian(Woodbury(W.A, W.U, W.C, copy(W.U)')) # testing edge case
    @test !issymmetric(getW(n, m))
    @test !ishermitian(getW(n, m))

    # test solves
    @test W\(W*x) ≈ x
    @test (x'*W)/W ≈ x'

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
    @test logabsdet(W)[1] ≈ logabsdet(MW)[1]
    @test logabsdet(W)[2] ≈ logabsdet(MW)[2]
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
    @test isposdef(F)

    # factorize and logdet after factorization
    F = factorize(W)
    @test logabsdet(F)[1] ≈ logabsdet(MW)[1]
    @test logabsdet(F)[2] ≈ logabsdet(MW)[2]
    @test logdet(F) ≈ logdet(MW)
    @test Matrix(inverse(F)) ≈ inv(MW)

    # test recursive stacking of woodbury objects
    b = randn(n, 1)
    W2 = Woodbury(W, b, ones(1, 1), b')
    F2 = factorize(W2)
    M2 = Matrix(W2)
    @test logdet(W2) ≈ logdet(M2)
    @test logdet(F2) ≈ logdet(M2)

    # test factorize_D behavior on non-hermitian 1x1 matrix
    using WoodburyIdentity: factorize_D
    FD = factorize_D(W2, fill(-rand(), (1, 1)))
    @test FD isa Matrix
    @test size(FD) == (1, 1)

    # test rank 1 constructor
    A = exp(randn())*I(n)
    u = randn(n)
    W = Woodbury(A, u)
    @test Matrix(W) ≈ A + u*u'
    v = randn(n)
    W = Woodbury(A, u, v')
    @test Matrix(W) ≈ A + u*v'

    # constructor with cholesky pivoted
    A = randn(1, n)
    A = A'A
    C = cholesky(A, Val(true), check = false)
    W = Woodbury(A, C)
    @test W isa Woodbury
    @test Matrix(W) ≈ 2A
end

end # TestWoodburyIdentity
