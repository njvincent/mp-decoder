# Local Stage-1 scan runner for cnot_primitive_xz.jl
# (plans/xz_noise_hub_cnot.md, Stage 1, step 4).
#
#   julia --threads=12 --startup-file=no jobs/cnot_primitive_xz_stage1_local.jl            # all variants
#   julia --threads=12 --startup-file=no jobs/cnot_primitive_xz_stage1_local.jl memory_T1d # selected
#
# One result file per (variant, L, p) under
# results/cnot_primitive_xz/stage1/<variant>/. Existing files are skipped, so an
# interrupted run can be resumed. Seeds are fixed per point.
#
# Variants
#   memory_T{1,2,4}d       one block, T = d, 2d, 4d noisy rounds (memory reference)
#   gadget_cnot_P2r10      one noisy CNOT: d rounds, CNOT round, d rounds (full model)
#   gadget_idle_P2r10      same, gate replaced by an idle round that keeps two-qubit noise
#   gadget_cnot_P2r0       ideal gate (propagation + primitive rule, no two-qubit noise)
#   gadget_idle_P2r0       two independent memories of 2d+1 rounds
#   gadget_cnot_P2r10_hubcontrol   symmetry check: the persistent block is the control

include(joinpath(@__DIR__, "..", "cnot_primitive_xz.jl"))

const L_LIST = [5, 7, 9, 11, 13]
const P_MEMORY = collect(0.008:0.002:0.024)
const P_GATE_NOISY = collect(0.003:0.001:0.013)
const P_GATE_IDEAL = collect(0.008:0.002:0.024)
const ACC_ERRORS = 1500
const MAX_TRIALS = 15_000
const ROOT = joinpath(@__DIR__, "..", "results", "cnot_primitive_xz", "stage1")

Zof(L) = ceil(Int, log(1.5, L))

const VARIANTS = [
    (name="memory_T1d", ps=P_MEMORY, Ls=L_LIST, sched=L -> memory_schedule(L), gate=:cnot, p2r=10.0),
    (name="gadget_cnot_P2r10", ps=P_GATE_NOISY, Ls=L_LIST, sched=L -> consecutive_schedule([L, L]), gate=:cnot, p2r=10.0),
    (name="memory_T2d", ps=P_MEMORY, Ls=L_LIST, sched=L -> memory_schedule(2L), gate=:cnot, p2r=10.0),
    (name="memory_T4d", ps=P_MEMORY, Ls=L_LIST, sched=L -> memory_schedule(4L), gate=:cnot, p2r=10.0),
    (name="gadget_idle_P2r0", ps=P_GATE_IDEAL, Ls=L_LIST, sched=L -> consecutive_schedule([L, L]), gate=:idle, p2r=0.0),
    (name="gadget_cnot_P2r0", ps=P_GATE_IDEAL, Ls=L_LIST, sched=L -> consecutive_schedule([L, L]), gate=:cnot, p2r=0.0),
    (name="gadget_idle_P2r10", ps=P_GATE_NOISY, Ls=L_LIST, sched=L -> consecutive_schedule([L, L]), gate=:idle, p2r=10.0),
    (name="gadget_cnot_P2r10_hubcontrol", ps=[0.006, 0.008, 0.010], Ls=[5, 9],
     sched=L -> consecutive_schedule([L, L]; hub_role=:control), gate=:cnot, p2r=10.0),
]

function run_variants(selected)
    for v in VARIANTS
        isempty(selected) || v.name in selected || continue
        dir = joinpath(ROOT, v.name)
        for p in v.ps, L in v.Ls
            cfg = XZConfig(
                L=L, Z=Zof(L), p=p, qrat=1.0, p2_ratio=v.p2r, r=3,
                gate_mode=v.gate, schedule=v.sched(L), cleanup_time=2L,
                stop_mode="capped", acc_errors=ACC_ERRORS, max_trials=MAX_TRIALS,
                trial_parallel=true, seed=hash((v.name, L, p)),
            )
            path = joinpath(dir, result_filename(cfg))
            if isfile(path)
                println("skip  $(v.name) L=$L p=$p")
                continue
            end
            data = simulate(cfg)
            write_result(path, data, params_record(cfg))
            println("done  $(v.name) L=$L p=$p  trials=$(data["trials"]) failures=$(data["failures"]) ",
                    "rate=$(round(data["fail_rate"], sigdigits=4))  $(round(data["wall_seconds"], digits=1)) s")
            flush(stdout)
        end
    end
end

run_variants(ARGS)
println("all requested Stage-1 variants finished")
