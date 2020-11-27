module WoodburyIdentity
using LinearAlgebra
using LinearAlgebra: checksquare

# using LazyArrays: applied, ApplyMatrix
import LazyInverse: inverse, Inverse

using LinearAlgebraExtensions
using LinearAlgebraExtensions: AbstractMatOrFac
using LinearAlgebraExtensions: LowRank

export Woodbury

# represents A + αUCV
# things that prevent C from being a scalar: checkdims, factorize, ...
# the α is beneficial to preserve p.s.d.-ness during inversion (see inverse)
struct Woodbury{T, AT, UT, CT, VT, F, L} <: Factorization{T}
    A::AT
    U::UT
    C::CT
    V::VT
	α::F # scalar, TODO: test if not ±1
	logabsdet::L
	function Woodbury(A::AbstractMatOrFac, U::AbstractMatOrFac,
					  C::AbstractMatOrFac, V::AbstractMatOrFac, α::Real = 1,
					  logabsdet::Union{NTuple{2, Real}, Nothing} = nothing;
					  check::Bool = true)
		 (α == 1 || α == -1) || throw(DomainError("α ≠ ±1 not yet tested: α = $α"))
		check && checkdims(A, U, C, V)
		T = promote_type(typeof(α), eltype.((A, U, C, V))...)	# check promote_type
		F = typeof(α)
		AT, UT, CT, VT, LT = typeof.((A, U, C, V, logabsdet))
		new{T, AT, UT, CT, VT, F, LT}(A, U, C, V, α, logabsdet) # tn1, tn2, tm1, tm2)
	end
end

matrix(x::Real) = fill(x, (1, 1)) # could in principle extend Matrix
matrix(x::AbstractVector) = reshape(x, (:, 1))
matrix(x::Adjoint{<:Any, <:AbstractVector}) = reshape(x, (1, :))
matrix(x::AbstractMatOrFac) = x

function Woodbury(A, U, C, V, α::Real = 1, abslogdet = nothing)
	Woodbury(matrix.((A, U, C, V))..., α, abslogdet)
end

# low rank correction
# NOTE: cannot rid of type restriction on A without introducing ambiguities
function Woodbury(A::AbstractMatOrFac, U::AbstractVecOrMat, V::AbstractMatrix = U',
	 			  α::Real = 1, logabsdet = nothing)
	Woodbury(A, LowRank(U, V), α, logabsdet)
end
function Woodbury(A, L::LowRank, α::Real = 1, logabsdet = nothing)
    Woodbury(A, L.U, I(rank(L)), L.V, α, logabsdet)
end
function Woodbury(A, C::CholeskyPivoted, α::Real = 1, logabsdet = nothing)
	Woodbury(A, LowRank(C), α, logabsdet)
end

# checks if the dimensions of A, U, C, V are consistent to form a Woodbury factorization
function checkdims(A, U, C, V)
	n = checksquare(A)
	k = checksquare(C)
	s = "is inconsistent with A ($(size(A))) and C ($(size(C)))"
	size(U) ≠ (n, k) && throw(DimensionMismatch("Size of U ($(size(U)))"*s))
	size(V) ≠ (k, n) && throw(DimensionMismatch("Size of V ($(size(V)))"*s))
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
_ishermitian(A::AbstractMatOrFac) = ishermitian(A)
function _ishermitian(W::Woodbury)
	(W.U ≡ W.V' || W.U == W.V') && _ishermitian(W.A) && _ishermitian(W.C)
end
# if efficiently-verifiable sufficient condition fails, check matrix
function LinearAlgebra.ishermitian(W::Woodbury)
	_ishermitian(W) || ishermitian(Matrix(W))
end
function LinearAlgebra.issymmetric(W::Woodbury)
	eltype(W) <: Real && ishermitian(W)
end

function LinearAlgebra.isposdef(W::Woodbury)
	W.logabsdet isa Nothing ? isposdef(factorize(W)) : (W.logabsdet[2] > 0)
end
function LinearAlgebra.adjoint(W::Woodbury)
	ishermitian(W) ? W : Woodbury(W.A', W.V', W.C', W.U', W.α, W.logabsdet)
end
function LinearAlgebra.transpose(W::Woodbury)
	issymmetric(W) ? W : Woodbury(transpose.((W.A, W.V, W.C, W.U))..., W.α, W.logabsdet)
end

# WARNING: creates views of previous object, so mutating it changes the original object
function Base.getindex(W::Woodbury, i::UnitRange, j::UnitRange)
	A = view(W.A, i, j)
	U = view(W.U, i, :)
	V = U ≡ V' ? U' : view(W.V, :, j)
	return Woodbury(A, U, W.C, V, W.α, nothing)
end
function Base.getindex(W::Woodbury, i::Int, j::Int)
	u = view(W.U, i, :)
	v = view(W.V, :, j)
	return W.A[i,j] + W.α * dot(u, W.C, v)
end
function LinearAlgebra.tr(W::Woodbury)
	n = checksquare(W)
	return tr(W.A) + W.α * tr(LowRank(W.U*W.C, W.V))
end

######################## Linear Algebra primitives #############################
Base.:*(W::Woodbury, x::AbstractVecOrMat) = mul!(similar(x), W, x)
Base.:*(B::AbstractMatrix, W::Woodbury) = (W'*B')'
function LinearAlgebra.mul!(y::AbstractVector, W::Woodbury, x::AbstractVector)
	t = similar(x, size(W.V, 1))
	mul!!(y, W, x, t)
end
function LinearAlgebra.mul!(y::AbstractMatrix, W::Woodbury, x::AbstractMatrix)
	t = similar(x, size(W.V, 1), size(x, 2))
	mul!!(y, W, x, t)
end

# stores result in y, uses temporary t
function mul!!(y::AbstractVecOrMat, W::Woodbury, x::AbstractVecOrMat, t::AbstractVecOrMat)
	k = size(W.V, 1)
	yk = y isa AbstractVector ? view(y, 1:k) : view(y, 1:k, :) # matrix memory alignment could be better
	mul!(yk, W.V, x)
	mul!(t, W.C, yk)
	mul!(y, W.U, t)
 	mul!(y, W.A, x, 1, W.α)
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
    if size(W.U, 1) > c*size(W.U, 2) # only use Woodbury identity when it is beneficial to do so
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
	else
		return factorize(AbstractMatrix(W))
	end
end

##################### conveniences for D = C⁻¹ ± V*A⁻¹*U ########################
compute_D(W::Woodbury) = compute_D!(W, *(W.V, inverse(W.A), W.U))
function compute_D!(W::Woodbury, VAU)
	invC = inv(W.C) # because we need the dense inverse matrix
	@. VAU = invC + W.α * VAU # could be made more efficient with 5 arg mul
	return VAU
end
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
