module WoodburyIdentity
using LinearAlgebra
using LinearAlgebra: checksquare

# using LazyArrays: applied, ApplyMatrix
import LazyInverse: inverse, Inverse

using LinearAlgebraExtensions
using LinearAlgebraExtensions: AbstractMatOrFac
using LinearAlgebraExtensions: LowRank

# TODO: pre-allocate intermediate storage
# represents A + αUCV
# the α is beneficial to preserve p.s.d.-ness during inversion (see inverse)
struct Woodbury{T, AT, UT, CT, VT, F} <: Factorization{T}
    A::AT
    U::UT
    C::CT
    V::VT
	α::F # this should either be + or -
	# tn1::V # temporary arrays
	# tn2::V
	# tm1::V
	# tm2::V
	function Woodbury(A::AT, U::UT, C::CT, V::VT, α::F = +) where {AT<:AbstractMatOrFac,
								UT, CT, VT, F<:Union{typeof(+), typeof(-)}}
		checkdims(A, U, C, V)
		# check promote_type
		T = promote_type(eltype.((A, U, C, V))...)
		# tn1 = zeros(size(A, 1))
		# tn2 = zeros(size(A, 2))
		# tm1 = zeros(size(C, 1))
		# tm2 = zeros(size(C, 2))
		new{T, AT, UT, CT, VT, F}(A, U, C, V, α)
															# tn1, tn2, tm1, tm2)
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
function Base.Matrix(W::Woodbury)
	W.α(Matrix(W.A), *(W.U, Matrix(W.C), W.V))
end

function Base.deepcopy(W::Woodbury)
	U = deepcopy(W.U)
	V = U ≡ V' ? U' : deepcopy(V)
	Woodbury(deepcopy(W.A), U, deepcopy(W.C), V, W.α)
end

import LinearAlgebra: issymmetric, ishermitian
issymmetric(W::Woodbury) = eltype(W) <: Real && ishermitian(W)
ishermitian(W::Woodbury) = (W.U ≡ W.V' || W.U == W.V') && ishermitian(W.A) && ishermitian(W.C)
function LinearAlgebra.adjoint(W::Woodbury)
	ishermitian(W) ? W : Woodbury(W.A', W.V', W.C', W.U', W.α)
end
function LinearAlgebra.transpose(W::Woodbury)
	issymmetric(W) ? W : Woodbury(transpose.((W.A, W.V, W.C, W.U))..., W.α)
end

# WARNING: creates views of previous object, so mutating it changes the original object
function Base.getindex(W::Woodbury, i::UnitRange, j::UnitRange)
	A = view(W.A, i, j)
	U = view(W.U, i, :)
	if ishermitian(W)
		Woodbury(A, U, W.C, U')
	else
		Woodbury(A, U, W.C, view(W.V, :, j))
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
import LinearAlgebra: dot, *, \, /
# TODO: take care of temporaries!
function *(W::Woodbury, x::AbstractVecOrMat)
	# mul!(W.tm2, W.V, x)
	# mul!(W.tm1, W.C)
	W.α(W.A*x, W.U*(W.C*(W.V*x)))
end
*(B::AbstractMatrix, W::Woodbury) = (W'*B')'

# ternary dot
function LinearAlgebra.dot(x::AbstractVecOrMat, W::Woodbury, y::AbstractVecOrMat)
    Ux = W.U'x     # memory allocation can be avoided with lazy arrays
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
# TODO: replace inverse with factorize, and let inverse call Inverse
# figure out constant c for which woodbury is most efficient
# could implement this in WoodburyMatrices and PR
function LinearAlgebra.factorize(W::Woodbury, c::Real = 1)
    if size(W.U, 1) > c*size(W.U, 2) # only use Woodbury identiy when it is beneficial to do so
		A = inverse(factorize(W.A))
		AU = A * W.U
		VA = (W.U ≡ W.V' && ishermitian(A)) ? AU' : W.V * A
		C = factorized_C(W, W.V*AU)
        Inverse(Woodbury(A, AU, C, VA, switch_α(W.α)))
    else
        factorize(Matrix(W))
    end
end

function factorized_C(W::Woodbury, VAU)
	invC = inv(W.C) + VAU # inv(C) because we need the dense inverse matrix
	if ishermitian(W)
		return try inverse(cholesky(Hermitian(invC))) catch end
	end
	return inverse(factorize(invC)) # could be made more efficient with 5 arg mul?
end

######################## Matrix determinant lemma ##############################
# TODO: make this efficient
# if W.A = W.C = I, this is Sylvesters determinant theorem
# Determinant lemma for A + α*(UCV)
function LinearAlgebra.det(W::Woodbury)
	n, m = checksquare.((W, W.C))
	α = (W.α == +) ? 1. : -1.
	invC = W.α(*(W.V, inverse(W.A), W.U), inverse(W.C))
	det(W.A) * det(W.C) * α^m * det(invC)
end

function LinearAlgebra.logdet(W::Woodbury)
	l, s = logabsdet(W)
	s > 0 ? l : error("Matrix is not positive definite")
end
function LinearAlgebra.logabsdet(W::Woodbury)
	n, m = checksquare.((W, W.C))
	la, sa = logabsdet(W.A)
	lc, sc = logabsdet(W.C)
	α = (W.α == +) ? 1. : -1.
	invC = W.α(*(W.V, inverse(W.A), W.U), inverse(W.C))
	ld, sd = logabsdet(invC)
	return +(la, lc, ld, m*log(abs(α))), *(sa, sc, sd, sign(α))
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
