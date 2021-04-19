module TestWoodburyIdentity
using WoodburyIdentity
using WoodburyIdentity: Woodbury, eltype, issymmetric, ishermitian
using LazyInverse: inverse, Inverse
using LinearAlgebra
using LinearAlgebraExtensions: LowRank
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
    n = 5
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

    # in-place multiplication
    y = randn(size(W, 1)) # with vector
    α, β = randn(2)
    b = α*(W*x) + β*y
    mul!(y, W, x, α, β)
    @test y ≈ b

    Y = randn(size(W, 1), size(X, 2)) # with matrix
    α, β = randn(2)
    B = α*(W*X) + β*Y
    mul!(Y, W, X, α, β)
    @test Y ≈ B

    @test α*W isa Woodbury
    @test Matrix(α*W) ≈ α*Matrix(W) # with scalar
    @test logdet(abs(α)*W) ≈ logdet(abs(α)*Matrix(W)) # checking that logdet works correctly

    @test eltype(W) == Float64
    @test issymmetric(W)
    @test ishermitian(W)
    # @test ishermitian(MW)
    # @test ishermitian(Woodbury(W.A, W.U, W.C, copy(W.U)')) # testing edge case
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
    i = 1:2
    j = 3:3
    Wij = W[i, j]
    @test Wij isa Woodbury
    @test Wij[1, 1] == W[1, 3]
    Wij = @view W[i, j]
    @test Wij isa Woodbury
    @test Wij.U isa SubArray

    # trace
    @test tr(W) ≈ tr(MW)

    # factorize
    F = factorize(W)
    @test Matrix(inverse(F)) ≈ inv(MW)
    @test inv(W) ≈ inv(MW)
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

    x = randn(n)
    b = W*x
    @test W \ b ≈ x

    # LowRank constructor
    U = randn(n, 1)
    L = LowRank(U, U')
    A = Matrix(L)
    W = Woodbury(I(n), L)

    # CholeskyPivoted constructor
    A = randn(1, n)
    A = A'A
    C = cholesky(A, Val(true), check = false)
    W = Woodbury(A, C)
    @test W isa Woodbury
    @test Matrix(W) ≈ 2A

    # tests with α = -
    W = Woodbury(I(n), C, -1)
    @test Matrix(W) ≈ I(n) - A
    @test inv(W) ≈ inv(I(n) - A)
    FW = factorize(W)
    @test Matrix(FW) ≈ I(n) - A
    @test Matrix(inverse(FW)) ≈ inv(I(n) - A)
    MW = I(n) - A
    # indexing
    for i in 1:n, j in 1:n
        @test W[i, j] ≈ MW[i, j]
    end
    @test tr(W) ≈ tr(MW)
    @test diag(W) ≈ diag(MW)

    # Woodbury structure with 1d outer dimension
    a, b, c, d = randn(4)
    W = Woodbury(a, b, c, d)
    MW = Matrix(W)
    @test size(MW) == (1, 1)
    @test MW[1, 1] ≈ a + *(b, c, d)

    a = randn()
    B = randn(1, n)
    C = randn(n, n)
    D = randn(n, 1)
    W = Woodbury(a, B, C, D)
    MW = Matrix(W)
    @test size(MW) == (1, 1)
    @test MW[1, 1] ≈ a + dot(B, C, D)

    @test inv(W) isa Real
    @test factorize(W) isa Real
end

end # TestWoodburyIdentity
