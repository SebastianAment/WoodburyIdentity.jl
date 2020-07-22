module WoodburyIdentity
using LinearAlgebra
using LinearAlgebra: checksquare

# using LazyArrays: applied, ApplyMatrix
import LazyInverse: inverse, Inverse

using LinearAlgebraExtensions
using LinearAlgebraExtensions: AbstractMatOrFac
using LinearAlgebraExtensions: LowRank

export Woodbury
# things that prevent C from being a scalar: checkdims, factorize, ...
# represents A + αUCV
# the α is beneficial to preserve p.s.d.-ness during inversion (see inverse)
struct Woodbury{T, AT, UT, CT, VT, F, L} <: Factorization{T}
    A::AT
    U::UT
    C::CT
    V::VT
	α::F # this should either be + or -, generalize to scalar?
	logabsdet::L
	function Woodbury(A::AbstractMatOrFac, U::AbstractMatOrFac, C, V::AbstractMatOrFac,
		 α::F = +, logabsdet = nothing) where {F<:Union{typeof(+), typeof(-)}} # logabsdet::Union{NTuple{2, Real}, Nothing} = nothing
		checkdims(A, U, C, V)
		# check promote_type
		T = promote_type(eltype.((A, U, C, V))...)
		AT, UT, CT, VT, LT = typeof.((A, U, C, V, logabsdet))
		new{T, AT, UT, CT, VT, F, LT}(A, U, C, V, α, logabsdet) # tn1, tn2, tm1, tm2)
	end
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
function Woodbury(A::AbstractMatOrFac, L::LowRank, α = +)
    Woodbury(A, L.U, 1.0*I(rank(L)), L.V, α)
end
Woodbury(A::AbstractMatrix, C::CholeskyPivoted, α = +) = Woodbury(A, LowRank(C), α)

woodbury(A, L::LowRank, α = +) = Woodbury(A, L, α)
function woodbury(A, U, C, V, α = +, c::Real = 1)
    W = Woodbury(A, U, C, V, α)
    # size(W.U, 1) > c * size(W.U, 2) ? W : Matrix(W) # only return Woodbury if it is efficient
end
switch_α(α::Union{typeof(+), typeof(-)}) = (α == +) ? (-) : (+)

# rank one correction
Woodbury(A, u::AbstractVector, α = +) = Woodbury(A, u, u', α)
function Woodbury(A, u::AbstractVector, v::Adjoint{<:Number, <:AbstractVector}, α = +)
	Woodbury(A, reshape(u, :, 1), fill(1., (1, 1)), v, α)
end
################################## Base ########################################
import Base: size, eltype, copy, deepcopy
size(W::Woodbury) = size(W.A)
size(W::Woodbury, d) = size(W.A, d)
eltype(W::Woodbury{T}) where {T} = T
function Base.AbstractMatrix(W::Woodbury)
	M = *(W.U, Matrix(W.C), W.V)
	@. M = W.α($Matrix(W.A), M)
	# ishermitian(W) ? Hermitian(M) : M
end
Base.Matrix(W::Woodbury) = AbstractMatrix(W)
function deepcopy(W::Woodbury)
	U = deepcopy(W.U)
	V = U ≡ V' ? U' : deepcopy(V)
	Woodbury(deepcopy(W.A), U, deepcopy(W.C), V, W.α, W.logabsdet)
end
copy(W::Woodbury) = deepcopy(W)

import LinearAlgebra: issymmetric, ishermitian, isposdef, adjoint, transpose, inv

inv(W::Woodbury) = Matrix(inverse(factorize(W)))
# inv(W::Woodbury) = W \ Matrix(1.0I(size(W, 1)))

# WARNING: sufficient but not necessary condition, useful for quick check
_ishermitian(A::AbstractMatOrFac) = ishermitian(A)
function _ishermitian(W::Woodbury)
	W.U ≡ W.V' && _ishermitian(W.A) && _ishermitian(W.C)
end
ishermitian(W::Woodbury) = _ishermitian(W) || ishermitian(Matrix(W))
issymmetric(W::Woodbury) = eltype(W) <: Real && ishermitian(W)

function isposdef(W::Woodbury)
	W.logabsdet isa Nothing ? isposdef(factorize(W)) : (W.logabsdet[2] > 0)
end
function adjoint(W::Woodbury)
	ishermitian(W) ? W : Woodbury(W.A', W.V', W.C', W.U', W.α, W.logabsdet)
end
function transpose(W::Woodbury)
	issymmetric(W) ? W : Woodbury(transpose.((W.A, W.V, W.C, W.U))..., W.α, W.logabsdet)
end

# WARNING: creates views of previous object, so mutating it changes the original object
function Base.getindex(W::Woodbury, i::UnitRange, j::UnitRange)
	A = view(W.A, i, j)
	U = view(W.U, i, :)
	if ishermitian(W)
		Woodbury(A, U, W.C, U', W.α, nothing)
	else
		Woodbury(A, U, W.C, view(W.V, :, j), W.α, nothing)
	end
end
function Base.getindex(W::Woodbury, i::Int, j::Int)
	u = view(W.U, i, :)
	v = view(W.V, :, j)
	W.A[i,j] + dot(u, W.C, v)
end
function LinearAlgebra.tr(W::Woodbury)
	n = checksquare(W)
	tr(W.A) + tr(LowRank(W.U*W.C, W.V))
end

######################## Linear Algebra primitives #############################
import LinearAlgebra: dot, *, \, /, mul!
*(W::Woodbury, x::AbstractVecOrMat) = mul!(similar(x), W, x)
*(B::AbstractMatrix, W::Woodbury) = (W'*B')'
function LinearAlgebra.mul!(y::AbstractVector, W::Woodbury, x::AbstractVector)
	t = similar(x, size(W.V, 1)) # temporary, add to struct?
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
 	mul!(y, W.A, x, 1, W.α(1))
end

# ternary dot
function LinearAlgebra.dot(x::AbstractVecOrMat, W::Woodbury, y::AbstractVector)
    Ux = W.U'x     # memory allocation can be avoided (with lazy arrays?)
    Vy = (x ≡ y && W.U ≡ W.V') ? Ux : W.V*y
    W.α(dot(x, W.A, y), dot(Ux, W.C, Vy))
end

# ternary mul
function *(x::AbstractMatrix, W::Woodbury, y::AbstractVecOrMat)
    xU = x*W.U
    Vy = (x ≡ y' && W.U ≡ W.V') ? xU' : W.V*y
    W.α(*(x, W.A, y), *(xU, W.C, Vy)) # can avoid two temporaries
end

\(W::Woodbury, B::AbstractVector) = factorize(W)\B
\(W::Woodbury, B::AbstractMatrix) = factorize(W)\B
/(B::AbstractMatrix, W::Woodbury) = B/factorize(W)

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
		α = switch_α(W.α)
		return inverse(Woodbury(A⁻¹, A⁻¹U, inverse(D), VA⁻¹, α, W_logabsdet))
	else
		return factorize(Matrix(W))
	end
end

##################### conveniences for D = C⁻¹ ± V*A⁻¹*U ########################
compute_D(W::Woodbury) = compute_D!(W, *(W.V, inverse(W.A), W.U))
function compute_D!(W::Woodbury, VAU)
	invC = AbstractMatrix(inverse(W.C)) # because we need the dense inverse matrix
	@. VAU = W.α(invC, VAU) # could be made more efficient with 5 arg mul
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
import LinearAlgebra: det, logdet, logabsdet
# if W.A = W.C = I, this is Sylvesters determinant theorem
# Determinant lemma for A + α*(UCV)
function det(W::Woodbury)
	l, s = logabsdet(W)
	exp(l) * s
end
function logabsdet(W::Woodbury)
	W.logabsdet == nothing ? logabsdet(factorize(W)) : W.logabsdet
end
# TODO: check this
@inline function _logabsdet(A, C, D, α::Union{typeof(+), typeof(-)})
	n, m = checksquare(A), checksquare(D)
	la, sa = logabsdet(A)
	lc, sc = logabsdet(C)
	ld, sd = logabsdet(D)
	sα = (α == -) && isodd(m) ? -1. : 1.
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
