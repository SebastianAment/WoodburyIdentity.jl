module WoodburyBenchmarks
using LinearAlgebra
using BenchmarkTools
using WoodburyIdentity
using WoodburyIdentity: Woodbury

# TODO: figure out parameter c in WoodburyIdentity, which
# controls when the Woodbury representation is more efficient than dense
function woodbury(n = 64, m = 3)
    suite = BenchmarkGroup()
    U = randn(n, m)
    W = Woodbury(I(n), U, I(m), U')
    # factorization
    suite["factorize"] = @benchmarkable factorize($W)
    x = randn(n)
    F = factorize(W)
    suite["solve"] = @benchmarkable $F \ $x
    suite["multiply"] = @benchmarkable $W * $x
    suite["ternary dot"] = @benchmarkable dot($x, $W, $x)
    suite["logdet"] = @benchmarkable logdet($W)
    return suite
end

function judge_factorize_logdet(n = 64, m = 3)
    U = randn(n, m)
    W = Woodbury(I(n), U, I(m), U')
    MW = Matrix(W)
    pooled = @benchmark WoodburyIdentity.factorize_logdet($W)
    separated = @benchmark begin factorize($W); logdet($W) end
    f = minimum
    judge(f(pooled), f(separated))
end

########  to compare
function dense(n = 64, m = 3)
    suite = BenchmarkGroup()
    U = randn(n, m)
    W = Woodbury(I(n), U, I(m), U')
    MW = Matrix(W)
    suite["factorize"] = @benchmarkable factorize($MW)
    MWF = factorize(MW)
    x = randn(n)
    suite["solve"] = @benchmarkable  $MWF \ $x
    suite["multiply"] = @benchmarkable $MW * $x
    suite["ternary dot"] = @benchmarkable dot($x, $MW, $x)
    suite["logdet"] = @benchmarkable logdet($MW)
    return suite
end

# comparing Woodbury identity to dense matrix algebra
function wood_vs_dense(n = 64, m = 3)
    wsuite = woodbury(n, m)
    wr = run(wsuite)
    dsuite = dense(n, m)
    dr = run(dsuite)
    judge(minimum(wr), minimum(dr))
end

end # WoodburyBenchmarks
