#############################################################################
# Test scripts for ACTIVSg2000 and ACTIVSg10k
#############################################################################

using ProxAL
using DelimitedFiles, Printf
using LinearAlgebra, JuMP, Ipopt
using Logging
using LazyArtifacts
using MPI, CUDA

MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)

solver     = length(ARGS) > 0 ? ARGS[1] : "exatron"
num_sweeps = length(ARGS) > 1 ? parse(Int, ARGS[2]) : (solver == "exatron" ? 1 : 2)
rho0       = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : (solver == "exatron" ? 1e-1 : 1e-2)
obj_scale  = length(ARGS) > 3 ? parse(Float64, ARGS[4]) : 1e-3
t_start    = length(ARGS) > 4 ? parse(Int, ARGS[5]) : 21

# choose case
case = "case9"
resolution = 30 #in minutes

# choose backend
backend = (solver == "ipopt") ? ProxAL.JuMPBackend() : ProxAL.AdmmBackend()

# case file
case_file = joinpath(artifact"ExaData", "ExaData/matpower/$(case).m")

# Model/formulation settings
modelinfo = ModelInfo()
modelinfo.case_name = case
modelinfo.time_horizon_start = t_start
modelinfo.num_time_periods = 1
modelinfo.load_scale = 1.0
if startswith(case, "case_ACTIVSg")
    tfile = Int(24 * 7 * (60 / resolution))
    case_file = joinpath(DATA_DIR, "$(case).m")
    load_file = joinpath(DATA_DIR, "$(case)_Jun_oneweek_$(tfile)_$(resolution)min")
    modelinfo.ramp_scale = Float64(resolution)
else
    load_file = joinpath(artifact"ExaData", "ExaData", "mp_demand", "$(case)_oneweek_168")
    modelinfo.ramp_scale = 0.3
end
modelinfo.corr_scale = 0.8
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = false
modelinfo.time_link_constr_type = :penalty
modelinfo.ctgs_link_constr_type = :corrective_penalty
modelinfo.allow_line_limits = false
modelinfo.num_ctgs = 0
modelinfo.obj_scale = obj_scale

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 1
algparams.tol = 1e-3
algparams.decompCtgs = false
algparams.tron_rho_pq = 3e3
algparams.tron_rho_pa = 3e4
algparams.tron_outer_iterlim = 10
algparams.tron_inner_iterlim = 500
algparams.tron_scale = 1e-5
algparams.tron_outer_eps = 1e-4
algparams.num_sweeps = num_sweeps
# if isa(backend, ProxAL.AdmmBackend)
#     algparams.device = ProxAL.CUDADevice
# end
algparams.optimizer = optimizer_with_attributes(
    Ipopt.Optimizer,
    "print_level" => 0,
    "tol" => 1e-4,
    "linear_solver" => "ma27"
)
algparams.init_opf = false

# dry run to compile everything
algparams.verbose = 0
algparams.iterlim = 1
# redirect_stdout(devnull) do
    global elapsed_t_dry_run_create_problem = @elapsed begin
        nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend)
    end
    global elapsed_t_dry_run_optimize = @elapsed ProxAL.optimize!(nlp)
# end

# Set up and solve problem
algparams.verbose = 1
algparams.iterlim = (solver == "ipopt") ? 100 : 500
algparams.mode = :coldstart
ranks = MPI.Comm_size(MPI.COMM_WORLD)
cur_logger = global_logger(NullLogger())
elapsed_t = @elapsed begin
    # redirect_stdout(devnull) do
        global nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend)
    # end
end
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    global_logger(cur_logger)
    println("Creating problem: $elapsed_t")
    np = MPI.Comm_size(MPI.COMM_WORLD)
    elapsed_t = @elapsed begin
        info = ProxAL.optimize!(nlp; ρ_t_initial = rho0, τ_factor = 2.5)
    end

    # @show(ARGS)
    # @show(info.iter)
    # @show(info.maxviol_d)
    # @show(info.maxviol_t_actual)
    @show(info.objvalue)
    # @show(info.wall_time_elapsed_actual)
    # @show(info.wall_time_elapsed_ideal)
    @show(nlp.problem.x.Pg)
    open("sol_$(solver).log", "w") do io
        write(io, "Pg: $(nlp.problem.x.Pg)\n")
        write(io, "Objective value: $(info.objvalue)\n")
        write(io, "Iterations: $(info.iter)\n")
    end
else
    info = ProxAL.optimize!(nlp; ρ_t_initial = rho0, τ_factor = 2.5)
end


MPI.Finalize()
