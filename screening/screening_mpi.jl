using OpenBLAS_jll
using MPI
using PowerModels
using ExaAdmm
using LazyArtifacts
using Ipopt
using Libdl

Libdl.dlopen(OpenBLAS_jll.libopenblas_path, RTLD_NOW | RTLD_GLOBAL ;throw_error=true)
Libdl.dlopen(OpenBLAS_jll.libopenblas_path, RTLD_NOW | RTLD_GLOBAL ;throw_error=true)

abstract type Contingency end
struct LineContingency <: Contingency end
struct GenContingency <: Contingency end

function set_ac_start!(data)
    for (i,bus) in data["bus"]
        bus["vm_start"] = bus["vm"]
        bus["va_start"] = bus["va"]
    end

    for (i,gen) in data["gen"]
        gen["pg_start"] = gen["pg"]
        gen["qg_start"] = gen["qg"]
    end
    for (i,branch) in data["branch"]
        if haskey(branch, "pf")
            println("Setting_branches")
            branch["pf_start"] = branch["pf"]
        end
        if haskey(branch, "qf")
            branch["qf_start"] = branch["qf"]
        end
        if haskey(branch, "pt")
            branch["pt_start"] = branch["pt"]
        end
        if haskey(branch, "qt")
            branch["qt_start"] = branch["qt"]
        end
    end
end

function solve(network, id::Int64, verbose::Int)
    try
        result = solve_ac_opf(network, optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => verbose, "max_iter" => 100, "linear_solver" => "ma27"
            )
        )
        if result["termination_status"] == LOCALLY_SOLVED
            return 1
        else
            return 0
        end
    catch e
        return 0
    end
    return 0
end

function solve_contingency(id::Int64, network, ::LineContingency, verbose::Int)
    cnetwork = deepcopy(network)
    cnetwork["branch"]["$id"]["br_status"] = 0
    set = PowerModels.calc_connected_components(cnetwork)
    # Make sure network is not split
    if length(set) == 1
        return solve(cnetwork, id, verbose)
    else
        println("Line contingency $id creates an island")
        return 0
    end
end

function solve_contingency(id::Int64, network, ::GenContingency, verbose::Int)
    cnetwork = deepcopy(network)
    cnetwork["gen"]["$id"]["gen_status"] = 0
    return solve(cnetwork, id, verbose)
end
MPI.Init()
nprocs = MPI.Comm_size(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)

# case = ARGS[1]
case = "case_ACTIVSg10k"
verbose = 0
# case = "case118"
# case = "case9"
const CASE_PATH = joinpath(artifact"ExaData", "ExaData", "matpower", "$case.m")

network = PowerModels.parse_file(CASE_PATH)
result = solve_ac_opf(network, optimizer_with_attributes(
    Ipopt.Optimizer,
    "print_level" => verbose, "linear_solver" => "ma27"
    )
)

if rank == 0
    PowerModels.export_matpower("export.m", network)
end

set_ac_start!(network)
ngen = length(network["gen"])
nlines = length(network["branch"])
results_gens = zeros(Int64,ngen)
results_lines = zeros(Int64,nlines)
# Generator contingencies
for i in 1:ngen
   if mod(i, nprocs) == rank
       println("Solving generator contingency $i")
       results_gens[i] = solve_contingency(i, network, GenContingency(), verbose)
       println("Solved generator contingency $i with $(results_gens[i])")
   end
end
MPI.Reduce!(results_gens, MPI.SUM, 0, MPI.COMM_WORLD)
if rank == 0
   println("Found $(length(findall(x -> x == 1, results_gens))) feasible generator contingencies of a total of $ngen generators: $(findall(x -> x == 1, results_gens))")
   open("$case.gen", "w") do io
       for res in findall(x -> x == 1, results_gens)
           write(io,"$res\n")
       end
   end
end

# Line contingencies
for i in 1:nlines
    if mod(i, nprocs) == rank
        println("Solving line contingency $i")
        results_lines[i] = solve_contingency(i, network, LineContingency(), verbose)
        println("Solved line contingency $i with $(results_lines[i])")
    end
end
MPI.Reduce!(results_lines, MPI.SUM, 0, MPI.COMM_WORLD)

if rank == 0
    @show nlines
    println("Found $(length(findall(x -> x == 1, results_lines))) feasible line contingencies of a total of $nlines lines: $(findall(x -> x == 1, results_lines))")
    open("$case.lines", "w") do io
        for res in findall(x -> x == 1, results_lines)
            write(io,"$res\n")
        end
    end
end
MPI.Finalize()

