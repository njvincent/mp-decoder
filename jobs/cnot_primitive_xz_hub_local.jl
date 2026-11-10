# Local Stage-2 hub scan for cnot_primitive_xz.jl
# (plans/xz_noise_hub_cnot.md, Stage 2). Use at most 8 threads on the
# development machine:
#
#   julia --threads=8 --startup-file=no jobs/cnot_primitive_xz_hub_local.jl
#
# One result file per (k, N, L, p) under
# results/cnot_primitive_xz/hub/k_<label>/N<N entry>/. Existing files are skipped,
# so an interrupted run can be resumed. Seeds are fixed per point.
#
# Environment overrides (space- or comma-separated lists):
#   HUB_P_LIST, HUB_L_LIST, HUB_N_LIST, HUB_GAP ("d" or an integer),
#   HUB_ACC_ERRORS, HUB_MAX_TRIALS
# HUB_N_LIST entries are CNOT counts or multiples of L ("L", "2L"); with a fixed
# integer HUB_GAP the latter keep the gate density and the aspect ratio fixed.

include(joinpath(@__DIR__, "..", "cnot_primitive_xz.jl"))

parse_list(T, s) = [parse(T, e) for e in split(strip(s), r"[\s,]+") if !isempty(e)]

# crossing region first, so a partial run is already useful
const P_LIST = parse_list(Float64, get(ENV, "HUB_P_LIST", "0.008 0.010 0.006 0.009 0.007 0.011 0.005 0.012 0.004"))
const L_LIST = parse_list(Int, get(ENV, "HUB_L_LIST", "5 7 9 11 13"))
const N_LIST = split(strip(get(ENV, "HUB_N_LIST", "1 2 4 8")), r"[\s,]+"; keepempty=false)
const GAP = lowercase(strip(get(ENV, "HUB_GAP", "d")))
const ACC_ERRORS = parse(Int, get(ENV, "HUB_ACC_ERRORS", "1500"))
const MAX_TRIALS = parse(Int, get(ENV, "HUB_MAX_TRIALS", "10000"))
const ROOT = joinpath(@__DIR__, "..", "results", "cnot_primitive_xz", "hub")

Zof(L) = ceil(Int, log(1.5, L))
resolve_N(entry, L) = endswith(entry, "L") ? (entry == "L" ? 1 : parse(Int, chop(entry))) * L : parse(Int, entry)

nthreads() > 8 && @warn "this machine's local runs should use at most 8 threads; got $(nthreads())"

function run_hub_scan()
    k_label = GAP == "d" ? "d" : GAP
    for p in P_LIST, L in L_LIST, N_entry in N_LIST
        N = resolve_N(N_entry, L)
        k = GAP == "d" ? L : parse(Int, GAP)
        cfg = XZConfig(
            L=L, Z=Zof(L), p=p, qrat=1.0, p2_ratio=10.0, r=3,
            gate_mode=:cnot, schedule=hub_schedule(L, N, k), cleanup_time=2L,
            stop_mode="capped", acc_errors=ACC_ERRORS, max_trials=MAX_TRIALS,
            trial_parallel=true, track_frames=true, seed=hash(("hub", k_label, N, L, p)),
        )
        path = joinpath(ROOT, "k_$(k_label)", "N$(N_entry)", result_filename(cfg))
        if isfile(path)
            println("skip  k=$k_label N=$N L=$L p=$p")
            continue
        end
        data = simulate(cfg)
        write_result(path, data, params_record(cfg))
        println("done  k=$k_label N=$N L=$L p=$p  trials=$(data["trials"]) failures=$(data["failures"]) ",
                "hub=$(data["hub_failures"]) partners=$(data["partner_failures"])  ",
                "$(round(data["wall_seconds"], digits=1)) s")
        flush(stdout)
    end
end

run_hub_scan()
println("hub scan finished")
