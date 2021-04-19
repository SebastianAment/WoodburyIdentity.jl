module WoodburyIdentity
using LinearAlgebra
using LinearAlgebra: checksquare

# IDEA: can optimize threshold constant c which makes WoodburyIdentity computationally beneficial
# using LazyArrays: applied, ApplyMatrix
import LazyInverse: inverse, Inverse

using LinearAlgebraExtensions
using LinearAlgebraExtensions: AbstractMatOrFac, LowRank

export Woodbury

# represents A + αUCV
# things that prevent C from being a scalar: checkdims, factorize, ...
# the α is beneficial to preserve p.s.d.-ness during inversion (see inverse)
# IDEA: try allowing rank 1 correct to be expressed with vector / adjoint
struct Woodbury{T, AT, UT, CT, VT, F, L, TT} <: Factorization{T}
    A::AT
    U::UT
    C::CT
    V::VT
	α::F # scalar, TODO: test if not ±1
	logabsdet::L
	temporaries::TT
end

function Woodbury(A::AbstractMatOrFac, U::AbstractMatOrFac,
				  C::AbstractMatOrFac, V::AbstractMatOrFac,
				  α::Real = 1, logabsdet::Union{NTuple{2, Real}, Nothing} = nothing,
				  temporaries = allocate_temporaries(U, V);
				  check::Bool = true)
	(α == 1 || α == -1) || throw(DomainError("α ≠ ±1 not yet tested: α = $α"))
	check && checkdims(A, U, C, V)
	T = promote_type(typeof(α), eltype.((A, U, C, V))...)	# check promote_type
	F = typeof(α)
	AT, UT, CT, VT, LT = typeof.((A, U, C, V, logabsdet))
	TT = typeof(temporaries)
	Woodbury{T, AT, UT, CT, VT, F, LT, TT}(A, U, C, V, α, logabsdet, temporaries)
end

function Woodbury(A::AbstractMatOrFac, U::AbstractVector,
				  C::Real, V::Adjoint{<:Any, <:AbstractVector}, α::Real = 1,
				  logabsdet::Union{NTuple{2, Real}, Nothing} = nothing,
				  temporaries = allocate_temporaries(U, V);
				  check::Bool = true)
	(α == 1 || α == -1) || throw(DomainError("α ≠ ±1 not yet tested: α = $α"))
	check && checkdims(A, U, C, V)
	T = promote_type(typeof(α), eltype.((A, U, C, V))...)	# check promote_type
	F = typeof(α)
	AT, UT, CT, VT, LT = typeof.((A, U, C, V, logabsdet))
	TT = typeof(temporaries)
	Woodbury{T, AT, UT, CT, VT, F, LT, TT}(A, U, C, V, α, logabsdet, temporaries)
end

const RankOneCorrection = Woodbury{<:Any, <:Any, <:AbstractVector, <:Any, <:Adjoint{<:Any, <:AbstractVector}}

# casts all inputs to an equivalent Matrix
matrix(x::Real) = fill(x, (1, 1)) # could in principle extend Matrix
matrix(x::AbstractVector) = reshape(x, (:, 1))
matrix(x::Adjoint{<:Any, <:AbstractVector}) = reshape(x, (1, :))
matrix(x::AbstractMatOrFac) = x

function Woodbury(A, U, C, V, α::Real = 1, abslogdet = nothing)
	Woodbury(matrix.((A, U, C, V))..., α, abslogdet)
end

# low rank correction
# NOTE: cannot rid of type restriction on A without introducing ambiguities
function Woodbury(A::AbstractMatOrFac, U::AbstractVector, V::Adjoint = U',
	 			  α::Real = 1, logabsdet = nothing)
	Woodbury(A, U, 1., V, α, logabsdet)
end
function Woodbury(A::AbstractMatOrFac, U::AbstractMatrix, V::AbstractMatrix = U',
	 			  α::Real = 1, logabsdet = nothing)
	Woodbury(A, U, 1.0I(size(U, 2)), V, α, logabsdet)
end
function Woodbury(A, L::LowRank, α::Real = 1, logabsdet = nothing)
    Woodbury(A, L.U, 1.0I(size(L.U, 2)), L.V, α, logabsdet)
end
function Woodbury(A, C::CholeskyPivoted, α::Real = 1, logabsdet = nothing)
	Woodbury(A, LowRank(C), α, logabsdet)
end
LinearAlgebra.checksquare(::Number) = 1 # since size(::Number, ::Int) = 1
# checks if the dimensions of A, U, C, V are consistent to form a Woodbury factorization
function checkdims(A, U, C, V)
	n, m = A isa Number ? (1, 1) : size(A)
	k, l = C isa Number ? (1, 1) : size(C)
	s = "is inconsistent with A ($(size(A))) and C ($(size(C)))"
	!(size(U, 1) == n && size(U, 2) == k) && throw(DimensionMismatch("Size of U ($(size(U))) "*s))
	!(size(V, 1) == l && size(V, 2) == m) && throw(DimensionMismatch("Size of V ($(size(V))) "*s))
	return true
end

# pseudo-constructor?
# c is used to set threshold for conversion to Matrix
function woodbury(A, U, C, V, α = 1, c::Real = 1)
    W = Woodbury(A, U, C, V, α)
	#  only return Woodbury if it is efficient
	if size(W.C, 1) ≥ c * size(W.U, 1) || size(W.C, 2) ≥ c * size(W.V, 2)
		W = Matrix(W)
	end
	return W
end
woodbury(A, L::LowRank, α = 1) = Woodbury(A, L, α)

################################## Base ########################################
Base.size(W::Woodbury) = size(W.A)
Base.size(W::Woodbury, d) = size(W.A, d)
Base.eltype(W::Woodbury{T}) where {T} = T
function Base.AbstractMatrix(W::Woodbury)
	Matrix(W.A) + W.α * *(W.U, W.C, W.V)
end
Base.Matrix(W::Woodbury) = AbstractMatrix(W)
function Base.deepcopy(W::Woodbury)
	U = deepcopy(W.U)
	V = U ≡ V' ? U' : deepcopy(V)
	Woodbury(deepcopy(W.A), U, deepcopy(W.C), V, W.α, W.logabsdet)
end
Base.copy(W::Woodbury) = deepcopy(W)

Base.inv(W::Woodbury) = inv(factorize(W))

# sufficient but not necessary condition, useful for quick check
_ishermitian(A::Union{Real, AbstractMatOrFac}) = ishermitian(A)
function _ishermitian(W::Woodbury)
	(W.U ≡ W.V' || W.U == W.V') && _ishermitian(W.A) && _ishermitian(W.C)
end
# if efficiently-verifiable sufficient condition fails, check matrix
LinearAlgebra.ishermitian(W::Woodbury) = _ishermitian(W) || ishermitian(Matrix(W))
LinearAlgebra.issymmetric(W::Woodbury) = eltype(W) <: Real && ishermitian(W)

LinearAlgebra.logabsdet(x) = log(abs(x)), sign(x)
function LinearAlgebra.isposdef(W::Woodbury)
	W.logabsdet isa Nothing ? isposdef(factorize(W)) : (W.logabsdet[2] > 0)
end
function LinearAlgebra.adjoint(W::Woodbury)
	ishermitian(W) ? W : Woodbury(W.A', W.V', W.C', W.U', W.α, W.logabsdet)
end
function LinearAlgebra.transpose(W::Woodbury)
	issymmetric(W) ? W : Woodbury(transpose.((W.A, W.V, W.C, W.U))..., W.α, W.logabsdet)
end

# indexed by two integers returns scalar
function Base.getindex(W::Woodbury, i, j)
	W.A[i, j] + W.α * *(W.U[i, :], W.C, W.V[:, j])
end
function Base.getindex(W::Woodbury, i::Int, j)
	u = @view W.U[i, :]
	W.A[i, j] + W.α * *(u', W.C, W.V[:, j])
end
# indexed by two vectors other, returns woodbury
function Base.getindex(W::Woodbury, i::AbstractVector, j::AbstractVector)
	A = W.A[i, j]
	U = W.U[i, :]
	V = W.U ≡ W.V' && i == j ? U' : W.V[:, j]
	return Woodbury(A, U, W.C, V, W.α, nothing)
end
function Base.view(W::Woodbury, i, j) # WARNING: this also takes temporaries
	A = view(W.A, i, j)
	U = view(W.U, i, :)
	V = W.U ≡ W.V' && i == j ? U' : view(W.V, :, j)
	return Woodbury(A, U, W.C, V, W.α, nothing, W.temporaries)
end
function LinearAlgebra.tr(W::Woodbury)
	n = checksquare(W)
	return tr(W.A) + W.α * tr(LowRank(W.U*W.C, W.V))
end

######################## Linear Algebra primitives #############################
Base.:*(a::Number, W::Woodbury) = Woodbury(a*W.A, W.U, a*W.C, W.V, W.α) # IDEA: could take over logabsdet efficiently
Base.:*(W::Woodbury, a::Number) = a*W

Base.:*(W::Woodbury, x::AbstractVecOrMat) = mul!(similar(x), W, x)
Base.:*(B::AbstractMatrix, W::Woodbury) = (W'*B')'
function LinearAlgebra.mul!(y::AbstractVector, W::Woodbury, x::AbstractVector, α::Real = 1, β::Real = 0)
	s, t = get_temporaries(W, x)
	mul!!(y, W, x, α, β, s, t)
end
function LinearAlgebra.mul!(y::AbstractMatrix, W::Woodbury, x::AbstractMatrix, α::Real = 1, β::Real = 0)
	s, t = get_temporaries(W, x)
	mul!!(y, W, x, α, β, s, t) # Pre-allocate!
end
# allocates s, t arrays for multiplication if not pre-allocated in W
allocate_temporaries(W::Woodbury, T::DataType = Float64) = allocate_temporaries(W.U, W.V, T)
function allocate_temporaries(W::Woodbury, n::Int, T::DataType = Float64)
	allocate_temporaries(W.U, W.V, n, T)
end
function allocate_temporaries(U, V, T::DataType = Float64)
	allocate_temporaries(size(U, 2), size(V, 1), T)
end
function allocate_temporaries(U, V, n::Int, T::DataType = Float64)
	allocate_temporaries(size(U, 2), size(V, 1), n, T)
end
function allocate_temporaries(nu::Int, nv::Int, T::DataType = Float64)
	s = zeros(T, nv)
	t = zeros(T, nu)
	return s, t
end
function allocate_temporaries(nu::Int, nv::Int, n::Int, T::DataType = Float64)
	s = zeros(T, nv, n)
	t = zeros(T, nv, n)
	return s, t
end
function get_temporaries(W::Woodbury, x::AbstractVector)
	T = promote_type(eltype(x), eltype(W))
	W.temporaries isa Tuple{<:AbstractVector{T}} ? W.temporaries : allocate_temporaries(W, eltype(x))
end
function get_temporaries(W::Woodbury, X::AbstractMatrix)
	T = promote_type(eltype(X), eltype(W))
	W.temporaries isa Tuple{<:AbstractMatrix{T}} ? W.temporaries : allocate_temporaries(W, size(X, 2), eltype(X))
end

function mul!!(y::AbstractVecOrMat, W::Woodbury, x::AbstractVecOrMat, α::Real, β::Real, s, t)
	mul!(s, W.V, x)
	mul!(t, W.C, s)
	mul!(y, W.U, t, α * W.α, β)
 	mul!(y, W.A, x, α, 1) # this allocates if D is diagonal due to broadcasting mechanism
end

# special case: rank one correction does not need additional temporaries for MVM
function LinearAlgebra.mul!(y::AbstractVector, W::RankOneCorrection, x::AbstractVector, α::Real = 1, β::Real = 0)
	s = dot(W.V', x)
	t = W.C * s
	@. y = (α * W.α) * W.U * t + β * y
 	mul!(y, W.A, x, α, 1)
end

# ternary dot
function LinearAlgebra.dot(x::AbstractVecOrMat, W::Woodbury, y::AbstractVector)
    Ux = W.U'x     # memory allocation can be avoided (with lazy arrays?)
    Vy = (x ≡ y && W.U ≡ W.V') ? Ux : W.V*y
    dot(x, W.A, y) + W.α * dot(Ux, W.C, Vy)
end

# ternary mul
function Base.:*(x::AbstractMatrix, W::Woodbury, y::AbstractVecOrMat)
    xU = x*W.U
    Vy = (x ≡ y' && W.U ≡ W.V') ? xU' : W.V*y
    *(x, W.A, y) + W.α * *(xU, W.C, Vy) # can avoid two temporaries
end

Base.:(\)(W::Woodbury, B::AbstractVector) = factorize(W)\B
Base.:(\)(W::Woodbury, B::AbstractMatrix) = factorize(W)\B
Base.:(/)(B::AbstractMatrix, W::Woodbury) = B/factorize(W)

function LinearAlgebra.diag(W::Woodbury)
    n = checksquare(W)
    d = zeros(eltype(W), n)
    for i in 1:n
        d[i] = W[i,i]
    end
    return d
end

########################## Matrix inversion lemma ##############################
# figure out constant c for which woodbury is most efficient
# could implement this in WoodburyMatrices and PR
function LinearAlgebra.factorize(W::Woodbury, c::Real = 1,
									compute_logdet::Val{T} = Val(true)) where T
	checksquare(W)
	if size(W.U, 1) < c*size(W.U, 2)
		return factorize(AbstractMatrix(W))
	else # only use Woodbury identity when it is beneficial to do so
		A = factorize(W.A)
		A⁻¹ = inverse(A)
		A⁻¹U = A⁻¹ * W.U
		VA⁻¹ = (W.U ≡ W.V' && ishermitian(A⁻¹)) ? A⁻¹U' : W.V * A⁻¹
		D = factorize_D(W, W.V*A⁻¹U)
		if T
			l, s = _logabsdet(A, W.C, D, W.α)
			W_logabsdet = (-l, s) # -l since the result is packaged in an inverse
		else
			W_logabsdet = nothing
		end
		α = -W.α # switch sign
		return inverse(Woodbury(A⁻¹, A⁻¹U, inverse(D), VA⁻¹, α, W_logabsdet))
	end
end

##################### conveniences for D = C⁻¹ ± V*A⁻¹*U ########################
compute_D(W::Woodbury) = compute_D!(W, *(W.V, inverse(W.A), W.U))
function compute_D!(W::Woodbury, VAU)
	invC = inv(W.C) # because we need the dense inverse matrix
	@. VAU = invC + W.α * VAU # could be made more efficient with 5 arg mul
end
compute_D!(W::Woodbury, VAU::Real) = inv(W.C) + W.α * VAU
factorize_D(W::Woodbury) = factorize_D(W, *(W.V, inverse(W.A), W.U))
function factorize_D(W::Woodbury, VAU)
	D = compute_D!(W, VAU)
	if size(D) == (1, 1)
		return D
	elseif _ishermitian(W)
		try return cholesky(Hermitian(D)) catch end
	end
	return factorize(D)
end

######################## Matrix determinant lemma ##############################
# if W.A = W.C = I, this is Sylvesters determinant theorem
# Determinant lemma for A + α*(UCV)
function LinearAlgebra.det(W::Woodbury)
	l, s = logabsdet(W)
	exp(l) * s
end
function LinearAlgebra.logdet(W::Woodbury)
	l, s = logabsdet(W)
	return s > 0 ? l : throw(DomainError("determinant negative: $(exp(l) * s)"))
end
function LinearAlgebra.logabsdet(W::Woodbury)
	W.logabsdet == nothing ? logabsdet(factorize(W)) : W.logabsdet
end

function _logabsdet(A, C, D, α::Real)
	n, m = checksquare(A), checksquare(D)
	la, sa = logabsdet(A)
	lc, sc = logabsdet(C)
	ld, sd = logabsdet(D)
	sα = (α == -1) && isodd(m) ? -1 : 1
	return +(la, lc, ld), *(sa, sc, sd, sα)
end

end # WoodburyIdentity

# function show(io::IO, W::Woodbury)
#     println(io, "Woodbury factorization:\nA:")
#     show(io, MIME("text/plain"), W.A)
#     print(io, "\nU:\n")
#     Base.print_matrix(IOContext(io, :compact=>true), W.U)
#     if isa(W.C, Matrix)
#         print(io, "\nC:\n")
#         Base.print_matrix(IOContext(io, :compact=>true), W.C)
#     else
#         print(io, "\nC: ", W.C)
#     end
#     print(io, "\nV:\n")
#     Base.print_matrix(IOContext(io, :compact=>true), W.V)
# end
