# Local hub scan for cnot_primitive_xz.jl (plans/xz_noise_hub_cnot.md, Stage 2).
#
# Design: the same gap k (ordinary hub rounds between CNOTs) at every size, and
# N = c d CNOTs with d = L, so the gate density 1/(k+1) and the circuit's
# aspect ratio are fixed. Use at most 8 threads on the development machine:
#
#   HUB_GAP=1 julia --threads=8 --startup-file=no jobs/cnot_primitive_xz_hub_local.jl
#
# One result file per (k, c, L, p) under
#   results/cnot_primitive_xz/hub/k_<k>/N<c>L/      ("NL" for c = 1)
# Existing files are skipped, so an interrupted run can be resumed. Seeds are
# fixed per point.
#
# Environment (space- or comma-separated lists):
#   HUB_GAP          gap k, a nonnegative integer (default 1)
#   HUB_N_LIST       CNOT counts as multiples of d = L: "L", "2L", ... (default "L 2L")
#   HUB_P_LIST       defaults for k = 0, 1, 3 (thresholds ~0.35%, 0.53%, 0.55-0.6%);
#                    other k must set it
#   HUB_L_LIST       default "5 7 9 11 13"
#   HUB_ACC_ERRORS, HUB_MAX_TRIALS   capped stopping (default 1500 failures or 10^4 trials)
#   HUB_STOP_COUNT   which failures HUB_ACC_ERRORS counts: "hub" (target A,
#                    default) or "any" (failed trials). The two get different
#                    seeds and filenames, so their trials are independent.

include(joinpath(@__DIR__, "..", "cnot_primitive_xz.jl"))

parse_list(T, s) = [parse(T, e) for e in split(strip(s), r"[\s,]+") if !isempty(e)]

const GAP = parse(Int, get(ENV, "HUB_GAP", "1"))
GAP >= 0 || error("HUB_GAP must be a nonnegative integer")
# about 0.5-1.5 times the threshold, steps of ~5-7% of it near p_c; crossing region first
const DEFAULT_P = Dict(
    0 => "0.0035 0.00325 0.00375 0.003 0.004 0.00275 0.00425 0.0025 0.0045 0.002 0.005",
    1 => "0.005 0.0055 0.0045 0.006 0.004 0.0065 0.0035 0.007 0.003 0.008 0.0025",
    3 => "0.0055 0.006 0.005 0.0065 0.0045 0.007 0.004 0.0075 0.0035 0.0085 0.003",
)
haskey(ENV, "HUB_P_LIST") || haskey(DEFAULT_P, GAP) ||
    error("no default p grid for HUB_GAP=$GAP (threshold not measured yet); set HUB_P_LIST")
const P_LIST = parse_list(Float64, get(ENV, "HUB_P_LIST", get(DEFAULT_P, GAP, "")))
const L_LIST = parse_list(Int, get(ENV, "HUB_L_LIST", "5 7 9 11 13"))
const N_LIST = split(strip(get(ENV, "HUB_N_LIST", "L 2L")), r"[\s,]+"; keepempty=false)
all(e -> occursin(r"^([1-9][0-9]*)?L$", e), N_LIST) ||
    error("HUB_N_LIST entries must be multiples of L (L, 2L, ...); got $(N_LIST)")
const ACC_ERRORS = parse(Int, get(ENV, "HUB_ACC_ERRORS", "1500"))
const MAX_TRIALS = parse(Int, get(ENV, "HUB_MAX_TRIALS", "10000"))
const STOP_COUNT = Symbol(lowercase(strip(get(ENV, "HUB_STOP_COUNT", "hub"))))
const ROOT = joinpath(@__DIR__, "..", "results", "cnot_primitive_xz", "hub")

Zof(L) = ceil(Int, log(1.5, L))
resolve_N(entry, L) = (entry == "L" ? 1 : parse(Int, chop(entry))) * L

nthreads() > 8 && @warn "this machine's local runs should use at most 8 threads; got $(nthreads())"

function run_hub_scan()
    for p in P_LIST, L in L_LIST, N_entry in N_LIST
        N = resolve_N(N_entry, L)
        # the :any seeds are those of the first k = 1 scan, so that scan resumes unchanged
        seed = STOP_COUNT === :any ? hash(("hub", string(GAP), N, L, p)) : hash(("hub", string(GAP), N, L, p, STOP_COUNT))
        cfg = XZConfig(
            L=L, Z=Zof(L), p=p, qrat=1.0, p2_ratio=10.0, r=3,
            gate_mode=:cnot, schedule=hub_schedule(L, N, GAP), cleanup_time=2L,
            stop_mode="capped", stop_count=STOP_COUNT, acc_errors=ACC_ERRORS, max_trials=MAX_TRIALS,
            trial_parallel=true, track_frames=true, seed=seed,
        )
        path = joinpath(ROOT, "k_$(GAP)", "N$(N_entry)", result_filename(cfg))
        if isfile(path)
            println("skip  k=$GAP N=$N L=$L p=$p")
            continue
        end
        data = simulate(cfg)
        write_result(path, data, params_record(cfg))
        println("done  k=$GAP N=$N L=$L p=$p  trials=$(data["trials"]) hub=$(data["hub_failures"]) ",
                "partners=$(data["partner_failures"])  $(round(data["wall_seconds"], digits=1)) s")
        flush(stdout)
    end
end

run_hub_scan()
println("hub scan finished")
