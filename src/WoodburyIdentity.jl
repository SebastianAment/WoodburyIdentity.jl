module WoodburyIdentity
using LinearAlgebra
using LinearAlgebra: checksquare

# using LazyArrays: applied, ApplyMatrix
import LazyInverse: inverse, Inverse

using LinearAlgebraExtensions
using LinearAlgebraExtensions: AbstractMatOrFac
using LinearAlgebraExtensions: LowRank

export Woodbury

# TODO: pre-allocate intermediate storage
# represents A + αUCV
# the α is beneficial to preserve p.s.d.-ness during inversion (see inverse)
struct Woodbury{T, AT, UT, CT, VT, F, L} <: Factorization{T}
    A::AT
    U::UT
    C::CT
    V::VT
	α::F # this should either be + or -
	logabsdet::L
	# tn1::V # temporary arrays
	# tn2::V
	# tm1::V
	# tm2::V
	function Woodbury(A::AbstractMatOrFac, U, C, V, α::F = +,
					logabsdet = nothing) where {F<:Union{typeof(+), typeof(-)}} # logabsdet::Union{NTuple{2, Real}, Nothing} = nothing
		checkdims(A, U, C, V)
		# check promote_type
		T = promote_type(eltype.((A, U, C, V))...)
		# tn1 = zeros(size(A, 1))
		# tn2 = zeros(size(A, 2))
		# tm1 = zeros(size(C, 1))
		# tm2 = zeros(size(C, 2))
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
	return
end

# pseudo-constructor?
function Woodbury(A::AbstractMatOrFac, L::LowRank, α = +)
    Woodbury(A, L.U, 1.0*I(rank(L)), L.V, α)
end
woodbury(A, L::LowRank, α = +) = Woodbury(A, L, α)
function woodbury(A, U, C, V, α = +, c::Real = 1)
    W = Woodbury(A, U, C, V, α)
    # size(W.U, 1) > c * size(W.U, 2) ? W : Matrix(W) # only return Woodbury if it is efficient
end
switch_α(α::Union{typeof(+), typeof(-)}) = (α == +) ? (-) : (+)

# TODO: make it work if U, V are vectors
# original : Vector , Matrix
# function Woodbury(A, U::Vector, C, V::Adjoint{<:Number, Vector}, α = +)
# 	Woodbury(A, reshape(U, , C, V, α)
# end
# Woodbury(A, U::AbstractVector, C, V::Adjoint) = Woodbury(A, U, C, Matrix(V))
################################## Base ########################################
Base.size(W::Woodbury) = size(W.A)
Base.size(W::Woodbury, d) = size(W.A, d)
Base.eltype(W::Woodbury{T}) where {T} = T
function Base.AbstractMatrix(W::Woodbury)
	M = *(W.U, Matrix(W.C), W.V)
	@. M = W.α($Matrix(W.A), M)
end
Base.Matrix(W::Woodbury) = AbstractMatrix(W)

function Base.deepcopy(W::Woodbury)
	U = deepcopy(W.U)
	V = U ≡ V' ? U' : deepcopy(V)
	Woodbury(deepcopy(W.A), U, deepcopy(W.C), V, W.α, W.logabsdet)
end

import LinearAlgebra: issymmetric, ishermitian, isposdef, adjoint, transpose
function ishermitian(W::Woodbury)
	(W.U ≡ W.V' || W.U == W.V') && ishermitian(W.A) && ishermitian(W.C)
end
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
# TODO: take care of temporaries!
function *(W::Woodbury, x::AbstractVecOrMat)
	# mul!(W.tm2, W.V, x)
	# mul!(W.tm1, W.C)
	Ax = W.A*x
	y = W.U*(W.C*(W.V*x))
	@. y = W.α(Ax, y)
end
*(B::AbstractMatrix, W::Woodbury) = (W'*B')'

# need to temporary arrays for multiplication
# function mul!(t1::T, t2::T, W::Woodbury, x::T) where {T<:AbstractVecOrMat}
# 	return 0
# end

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

\(W::Woodbury, B::AbstractVecOrMat) = factorize(W)\B
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
			W_logabsdet = (-l, s) # since we are taking the inverse
		else
			W_logabsdet = nothing
		end
		inverse(Woodbury(A⁻¹, A⁻¹U, inverse(D), VA⁻¹, switch_α(W.α), W_logabsdet))
	else
		factorize(Matrix(W))
	end
end

##################### conveniences for D = C⁻¹ ± V*A⁻¹*U ########################
compute_D(W::Woodbury) = compute_D!(W, *(W.V, inverse(W.A), W.U))
function compute_D!(W::Woodbury, VAU)
	invC = AbstractMatrix(inverse(W.C)) # because we need the dense inverse matrix
	@. VAU = W.α(VAU, invC) # could be made more efficient with 5 arg mul
	return VAU
end
factorize_D(W::Woodbury) = factorize_D(W, *(W.V, inverse(W.A), W.U))
function factorize_D(W::Woodbury, VAU)
	D = compute_D!(W, VAU)
	if ishermitian(W)
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
	n, m = checksquare.((A, D))
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

# TODO:
# function ldiv!(W::Woodbury, B::AbstractVector)
#     length(B) == size(W, 1) || throw(DimensionMismatch("Vector length $(length(B)) must match matrix size $(size(W,1))"))
#     copyto!(W.tmpN1, B)
#     Alu = lu(W.A) # Note. This makes an allocation (unless A::LU). Alternative is to destroy W.A.
#     ldiv!(Alu, W.tmpN1)
#     mul!(W.tmpk1, W.V, W.tmpN1)
#     mul!(W.tmpk2, W.Cp, W.tmpk1)
#     mul!(W.tmpN2, W.U, W.tmpk2)
#     ldiv!(Alu, W.tmpN2)
#     for i = 1:length(W.tmpN2)
#         @inbounds B[i] = W.tmpN1[i] - W.tmpN2[i]
#     end
#     B
# end
