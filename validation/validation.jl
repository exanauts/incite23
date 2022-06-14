using Ipopt
using ProxAL
using PowerModels
using LazyArtifacts
using Test

case = "case9"

case_file = joinpath(artifact"ExaData", "ExaData", "matpower", "$(case).m")
load_file = joinpath(artifact"ExaData", "ExaData", "mp_demand", "$(case)_oneweek_168")

modelinfo = ModelInfo()
modelinfo.case_name = case
modelinfo.time_horizon_start = 1
modelinfo.num_time_periods = 1
modelinfo.load_scale = 1.0
modelinfo.ramp_scale = 0.3
modelinfo.corr_scale = 0.8
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = false
modelinfo.time_link_constr_type = :penalty
modelinfo.ctgs_link_constr_type = :corrective_penalty
modelinfo.allow_line_limits = false
modelinfo.num_ctgs = 0
modelinfo.obj_scale = 0.001

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
algparams.num_sweeps = 1
algparams.init_opf = false

algparams.optimizer = optimizer_with_attributes(
    Ipopt.Optimizer,
    "print_level" => 5,
    "tol" => 1e-4,
)

# Load files
nlp = ProxALEvaluator(
    case_file,
    load_file,
    modelinfo,
    algparams,
    ProxAL.JuMPBackend(),
    nothing
)
network = PowerModels.parse_file(case_file)
nbus = length(nlp.opfdata.buses)

pd_proxal = nlp.opfdata.Pd
qd_proxal = nlp.opfdata.Qd

pd_pm = zeros(nbus)
qd_pm = zeros(nbus)

for i in eachindex(network["load"])
    @show i
    @show load = network["load"]["$i"]
    pd_pm[load["load_bus"]] = load["pd"]
    qd_pm[load["load_bus"]] = load["qd"]
end

# Check loads are the same
@test all(pd_proxal[:,1] .== (pd_pm .* baseMVA ))
@test all(qd_proxal[:,1] == (qd_pm .* baseMVA ))

# Start optimization
result_proxal = ProxAL.optimize!(nlp)
result_pm = solve_ac_opf(network, algparams.optimizer)

@test result_pm["objective"] ≈ (result_proxal.objvalue[end] / modelinfo.obj_scale) rtol=1e-3